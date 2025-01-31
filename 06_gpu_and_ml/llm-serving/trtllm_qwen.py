# ---
# deploy: true
# ---

# # Serverless TensorRT-LLM (LLaMA 3 8B)

# In this example, we demonstrate how to use the TensorRT-LLM framework to serve Meta's LLaMA 3 8B model
# at very high throughput.

# We achieve a total throughput of over 25,000 output tokens per second on a single NVIDIA H100 GPU.
# At [Modal's on-demand rate](https://modal.com/pricing) of ~$4.50/hr, that's under $0.05 per million tokens --
# on auto-scaling infrastructure and served via a customizable API.

# Additional optimizations like speculative sampling can further improve throughput.

# ## Overview

# This guide is intended to document two things:
# the general process for building TensorRT-LLM on Modal
# and a specific configuration for serving the LLaMA 3 8B model.

# ### Build process

# Any given TensorRT-LLM service requires a multi-stage build process,
# starting from model weights and ending with a compiled engine.
# Because that process touches many sharp-edged high-performance components
# across the stack, it can easily go wrong in subtle and hard-to-debug ways
# that are idiosyncratic to specific systems.
# And debugging GPU workloads is expensive!

# This example builds an entire service from scratch, from downloading weight tensors
# to responding to requests, and so serves as living, interactive documentation of a TensorRT-LLM
# build process that works on Modal.

# ### Engine configuration

# TensorRT-LLM is the Lamborghini of inference engines: it achieves seriously
# impressive performance, but only if you tune it carefully.
# We carefully document the choices we made here and point to additional resources
# so you know where and how you might adjust the parameters for your use case.

# ## Installing TensorRT-LLM

# To run TensorRT-LLM, we must first install it. Easier said than done!

# In Modal, we define [container images](https://modal.com/docs/guide/custom-container) that run our serverless workloads.
# All Modal containers have access to GPU drivers via the underlying host environment,
# but we still need to install the software stack on top of the drivers, from the CUDA runtime up.

# We start from an official `nvidia/cuda` image,
# which includes the CUDA runtime & development libraries
# and the environment configuration necessary to run them.

from typing import Optional

import modal
import pydantic  # for typing, used later

tensorrt_image = modal.Image.from_registry(
    "nvidia/cuda:12.4.1-devel-ubuntu22.04",
    add_python="3.10",  # TRT-LLM requires Python 3.10
).entrypoint([])  # remove verbose logging by base image on entry

# On top of that, we add some system dependencies of TensorRT-LLM,
# including OpenMPI for distributed communication, some core software like `git`,
# and the `tensorrt_llm` package itself.

GIT_HASH = "b0880169d0fb8cd0363049d91aa548e58a41be07"

tensorrt_image = tensorrt_image.apt_install(
    "openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget"
).pip_install(
    "tensorrt_llm==0.14.0",
    "pynvml<12",  # avoid breaking change to pynvml version API
    pre=True,
    extra_index_url="https://pypi.nvidia.com",
# ).run_commands(
    # "git clone https://github.com/NVIDIA/TensorRT-LLM",
    # f"cd TensorRT-LLM && git checkout {GIT_HASH}"
)

# Note that we're doing this by [method-chaining](https://quanticdev.com/articles/method-chaining/)
# a number of calls to methods on the `modal.Image`. If you're familiar with
# Dockerfiles, you can think of this as a Pythonic interface to instructions like `RUN` and `CMD`.

# End-to-end, this step takes five minutes.
# If you're reading this from top to bottom,
# you might want to stop here and execute the example
# with `modal run trtllm_llama.py`
# so that it runs in the background while you read the rest.

# ## Downloading the Model

# Next, we download the model we want to serve. In this case, we're using the instruction-tuned
# version of Meta's LLaMA 3 8B model.
# We use the function below to download the model from the Hugging Face Hub.

MODEL_DIR = "/root/model/model_input"
# MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"  # fork without repo gating
# MODEL_REVISION = "b1532e4dee724d9ba63fe17496f298254d87ca64"  # pin model revisions to prevent unexpected changes!
# MODEL_ID = "unsloth/Qwen2.5-Coder-7B-Instruct"
# MODEL_REVISION = "3fd3aab092612530a892ff49027dfd4f39046ec3"

MODEL_ID =   "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
# MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"


def download_model():
    import os

    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        MODEL_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
        revision=MODEL_REVISION,
    )
    move_cache()


# Just defining that function doesn't actually download the model, though.
# We can run it by adding it to the image's build process with `run_function`.
# The download process has its own dependencies, which we add here.

MINUTES = 60  # seconds
tensorrt_image = (  # update the image by downloading the model we're using
    tensorrt_image.pip_install(  # add utilities for downloading the model
        "hf-transfer==0.1.8",
        "huggingface_hub==0.26.2",
        "requests~=2.31.0",
    )
    .env(  # hf-transfer for faster downloads
        {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
    )
    .run_function(  # download the model
        download_model,
        timeout=20 * MINUTES,
    )
)

# ## Quantization

# The amount of GPU RAM on a single card is a tight constraint for most LLMs:
# RAM is measured in billions of bytes and models have billions of parameters.
# The performance cliff if you need to spill to CPU memory is steep,
# so all of those parameters must fit in the GPU memory,
# along with other things like the KV cache.

# The simplest way to reduce LLM inference's RAM requirements is to make the model's parameters smaller,
# to fit their values in a smaller number of bits, like four or eight. This is known as _quantization_.

# We use a quantization script provided by the TensorRT-LLM team.
# This script takes a few minutes to run.

CONVERSION_SCRIPT_URL = f"https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/{GIT_HASH}/examples/quantization/quantize.py"

# NVIDIA's Ada Lovelace/Hopper chips, like the 4090, L40S, and H100,
# are capable of native calculations in 8bit floating point numbers, so we choose that as our quantization format (`qformat`).
# These GPUs are capable of twice as many floating point operations per second in 8bit as in 16bit --
# about two quadrillion per second on an H100 SXM.

N_GPUS = 1  # Heads up: this example has not yet been tested with multiple GPUs
GPU_CONFIG = modal.gpu.H100(count=N_GPUS)

DTYPE = "float16"  # format we download in, regular fp16
QFORMAT = "fp8"  # format we quantize the weights to
KV_CACHE_DTYPE = "fp8"  # format we quantize the KV cache to

# Quantization is lossy, but the impact on model quality can be minimized by
# tuning the quantization parameters based on target outputs.

CALIB_SIZE = "512"  # size of calibration dataset

# We put that all together with another invocation of `.run_commands`.

QUANTIZATION_ARGS = f"--dtype={DTYPE} --qformat={QFORMAT} --kv_cache_dtype={KV_CACHE_DTYPE} --calib_size={CALIB_SIZE}"

CKPT_DIR = "/root/model/model_ckpt"
tensorrt_image = (  # update the image by quantizing the model
    tensorrt_image.run_commands(  # takes ~2 minutes
        [
            f"wget {CONVERSION_SCRIPT_URL} -O /root/convert.py",
            f"python /root/convert.py --model_dir={MODEL_DIR} --output_dir={CKPT_DIR}"
            + f" --tp_size={N_GPUS}"
            + f" {QUANTIZATION_ARGS}",
        ],
        gpu=GPU_CONFIG,
    )
)

# ## Compiling the engine

# TensorRT-LLM achieves its high throughput primarily by compiling the model:
# making concrete choices of CUDA kernels to execute for each operation.
# These kernels are much more specific than `matrix_multiply` or `softmax` --
# they have names like `maxwell_scudnn_winograd_128x128_ldg1_ldg4_tile148t_nt`.
# They are optimized for the specific types and shapes of tensors that the model uses
# and for the specific hardware that the model runs on.

# That means we need to know all of that information a priori --
# more like the original TensorFlow, which defined static graphs, than like PyTorch,
# which builds up a graph of kernels dynamically at runtime.

# This extra layer of constraint on our LLM service is an important part of
# what allows TensorRT-LLM to achieve its high throughput.

# So we need to specify things like the maximum batch size and the lengths of inputs and outputs.
# The closer these are to the actual values we'll use in production, the better the throughput we'll get.

# Since we want to maximize the throughput, assuming we had a constant workload,
# we set the batch size to the largest value we can fit in GPU RAM.
# Quantization helps us again here, since it allows us to fit more tokens in the same RAM.

MAX_INPUT_LEN, MAX_OUTPUT_LEN = 8192, 1024
MAX_NUM_TOKENS = MAX_INPUT_LEN + MAX_OUTPUT_LEN
MAX_BATCH_SIZE = (
    1024  # better throughput at larger batch sizes, limited by GPU RAM
)
ENGINE_DIR = "/root/model/model_output"

SIZE_ARGS = f"--max_input_len={MAX_INPUT_LEN} --max_num_tokens={MAX_NUM_TOKENS} --max_batch_size={MAX_BATCH_SIZE}"

# There are many additional options you can pass to `trtllm-build` to tune the engine for your specific workload.
# You can find the document we used for LLaMA
# [here](https://github.com/NVIDIA/TensorRT-LLM/tree/b0880169d0fb8cd0363049d91aa548e58a41be07/examples/llama),
# which you can use to adjust the arguments to fit your workloads,
# e.g. adjusting rotary embeddings and block sizes for longer contexts.
# We also recommend the [official TRT-LLM best practices guide](https://nvidia.github.io/TensorRT-LLM/performance/perf-best-practices.html).

# To make best use of our 8bit floating point hardware, and the weights and KV cache we have quantized,
# we activate the 8bit floating point fused multi-head attention plugin.

# Because we are targeting maximum throughput, we do not activate the low latency 8bit floating point matrix multiplication plugin
# or the 8bit floating point matrix multiplication (`gemm`) plugin, which documentation indicates target smaller batch sizes.

PLUGIN_ARGS = "--use_fp8_context_fmha enable"

# We put all of this together with another invocation of `.run_commands`.
def calc_max_draft_len(windows_size, ngram_size, verification_set_size):
    return (
        (0 if (ngram_size == 1) else ngram_size - 2) +
        (windows_size - 1 + verification_set_size) * (ngram_size - 1)
    )

windows_size = 8
ngram_size = 8
verification_set_size = 4
max_draft_len = calc_max_draft_len(windows_size, ngram_size, verification_set_size)

tensorrt_image = (  # update the image by building the TensorRT engine
    tensorrt_image.run_commands(  # takes ~5 minutes
        [
            f"trtllm-build --checkpoint_dir {CKPT_DIR} --output_dir {ENGINE_DIR}"
            + f" --workers={N_GPUS}"
            # + f" --speculative_decoding_mode=lookahead_decoding"
            # + f" --max_draft_len={max_draft_len}"
            + f" {SIZE_ARGS}"
            + f" {PLUGIN_ARGS}"
        ],
        gpu=GPU_CONFIG,  # TRT-LLM compilation is GPU-specific, so make sure this matches production!
    ).env(  # show more log information from the inference engine
        {"TLLM_LOG_LEVEL": "INFO"}
    )
)

# ## Serving inference at tens of thousands of tokens per second

# Now that we have the engine compiled, we can serve it with Modal by creating an `App`.

app = modal.App(
    f"example-trtllm-hack-{MODEL_ID.split('/')[-1].replace('.', '')}", image=tensorrt_image
)

# Thanks to our custom container runtime system even this large, many gigabyte container boots in seconds.

# At container start time, we boot up the engine, which completes in under 30 seconds.
# Container starts are triggered when Modal scales up your infrastructure,
# like the first time you run this code or the first time a request comes in after a period of inactivity.

# Container lifecycles in Modal are managed via our `Cls` interface, so we define one below
# to manage the engine and run inference.
# For details, see [this guide](https://modal.com/docs/guide/lifecycle-functions).


@app.cls(
    gpu=GPU_CONFIG,
    container_idle_timeout=10 * MINUTES,
    image=tensorrt_image,
)
class Model:
    @modal.enter()
    def load(self):
        """Loads the TRT-LLM engine and configures our tokenizer.

        The @enter decorator ensures that it runs only once per container, when it starts."""
        import time

        print(
            f"{COLOR['HEADER']}🥶 Cold boot: spinning up TRT-LLM engine{COLOR['ENDC']}"
        )
        self.init_start = time.monotonic_ns()

        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunner
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        # LLaMA models do not have a padding token, so we use the EOS token
        self.tokenizer.add_special_tokens(
            {"pad_token": self.tokenizer.eos_token}
        )
        # and then we add it from the left, to minimize impact on the output
        self.tokenizer.padding_side = "left"
        self.pad_id = self.tokenizer.pad_token_id
        self.end_id = self.tokenizer.eos_token_id

        runner_kwargs = dict(
            engine_dir=f"{ENGINE_DIR}",
            lora_dir=None,
            rank=tensorrt_llm.mpi_rank(),  # this will need to be adjusted to use multiple GPUs
            max_output_len=MAX_OUTPUT_LEN,
        )

        self.model = ModelRunner.from_dir(**runner_kwargs)

        self.init_duration_s = (time.monotonic_ns() - self.init_start) / 1e9
        print(
            f"{COLOR['HEADER']}🚀 Cold boot finished in {self.init_duration_s}s{COLOR['ENDC']}"
        )

    @modal.method()
    def generate(self, prompts: list[str], settings=None):
        """Generate responses to a batch of prompts, optionally with custom inference settings."""
        import time

        if settings is None or not settings:
            settings = dict(
                temperature=0.1,  # temperature 0 not allowed, so we set top_k to 1 to get the same effect
                top_k=1,
                stop_words_list=None,
                repetition_penalty=1.1,
            )

        settings[
            "max_new_tokens"
        ] = MAX_OUTPUT_LEN  # exceeding this will raise an error
        settings["end_id"] = self.end_id
        settings["pad_id"] = self.pad_id
        # settings["lookahead_config"] = [windows_size, ngram_size, verification_set_size]

        num_prompts = len(prompts)

        if num_prompts > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {num_prompts} exceeds maximum of {MAX_BATCH_SIZE}"
            )

        print(
            f"{COLOR['HEADER']}🚀 Generating completions for batch of size {num_prompts}...{COLOR['ENDC']}"
        )
        start = time.monotonic_ns()

        parsed_prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for prompt in prompts
        ]

        print(
            f"{COLOR['HEADER']}Parsed prompts:{COLOR['ENDC']}",
            *parsed_prompts,
            sep="\n\t",
        )

        inputs_t = self.tokenizer(
            parsed_prompts, return_tensors="pt", padding=True, truncation=False
        )["input_ids"]

        print(
            f"{COLOR['HEADER']}Input tensors:{COLOR['ENDC']}", inputs_t[:, :8]
        )

        outputs_t = self.model.generate(inputs_t, **settings)

        outputs_text = self.tokenizer.batch_decode(
            outputs_t[:, 0]
        )  # only one output per input, so we index with 0

        responses = [
            extract_assistant_response(output_text)
            for output_text in outputs_text
        ]
        duration_s = (time.monotonic_ns() - start) / 1e9

        num_tokens = sum(
            map(lambda r: len(self.tokenizer.encode(r)), responses)
        )

        i = 0
        for prompt, response in zip(prompts, responses):
            print(f"{i} Prompt:\n{prompt}")
            print(f"{i} Response:\n{response}")
            # print(
                # f"{COLOR['HEADER']}{COLOR['GREEN']}{prompt}",
                # f"\n{COLOR['BLUE']}{response}",
                # "\n\n",
                # sep=COLOR["ENDC"],
            # )
            i += 1

        print(
            f"{COLOR['HEADER']}{COLOR['GREEN']}Generated {num_tokens} tokens from {MODEL_ID} in {duration_s:.1f} seconds,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second for batch of size {num_prompts} on {GPU_CONFIG}.{COLOR['ENDC']}"
        )

        return responses


# ## Calling our inference function

# Now, how do we actually run the model?

# There are two basic methods: from Python via our SDK or from anywhere, by setting up an API.

# ### Calling inference from Python

# To run our `Model`'s `.generate` method from Python, we just need to call it --
# with `.remote` appended to run it on Modal.

# We wrap that logic in a `local_entrypoint` so you can run it from the command line with
# ```bash
# modal run trtllm_llama.py
# ```

# For simplicity, we hard-code a batch of 128 questions to ask the model,
# and then bulk it up to a batch size of 1024 by appending seven distinct prefixes.
# These prefixes ensure KV cache misses for the remainder of the generations,
# to keep the benchmark closer to what can be expected in a real workload.

"""
@app.function(
    gpu=GPU_CONFIG,
    container_idle_timeout=10 * MINUTES,
    image=tensorrt_image,
)
def gogo():
    import subprocess
"""
    

@app.local_entrypoint()
def main():
    import glob
    import json
    from pathlib import Path

    prompts = []
    for filepath in glob.glob("reqs/*.json"):
        input_json = json.loads(Path(filepath).read_text())
        prompts.append(input_json["prompt"])
    prompts = prompts[-1:]
        
        
    

    # gogo.remote(prompts)
    model = Model()
    model.generate.remote(prompts)

# ### Calling inference via an API

# We can use `modal.web_endpoint` and `app.function` to turn any Python function into a web API.

# This API wrapper doesn't need all the dependencies of the core inference service,
# so we switch images here to a basic Linux image, `debian_slim`, and add the FastAPI stack.

web_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "fastapi[standard]==0.115.4",
    "pydantic==2.9.2",
    "starlette==0.41.2",
)


# From there, we can take the same remote generation logic we used in `main`
# and serve it with only a few more lines of code.


class GenerateRequest(pydantic.BaseModel):
    prompts: list[str]
    settings: Optional[dict] = None




# To set our function up as a web endpoint, we need to run this file --
# with `modal serve` to create a hot-reloading development server or `modal deploy` to deploy it to production.

# ```bash
# modal serve trtllm_llama.py
# ```

# The URL for the endpoint appears in the output of the `modal serve` or `modal deploy` command.
# Add `/docs` to the end of this URL to see the interactive Swagger documentation for the endpoint.

# You can also test the endpoint by sending a POST request with `curl` from another terminal:

# ```bash
# curl -X POST url-from-output-of-modal-serve-here \
# -H "Content-Type: application/json" \
# -d '{
#     "prompts": ["Tell me a joke", "Describe a dream you had recently", "Share your favorite childhood memory"]
# }' | python -m json.tool # python for pretty-printing, optional
# ```

# And now you have a high-throughput, low-latency, autoscaling API for serving LLM completions!

# ## Footer

# The rest of the code in this example is utility code.


COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}


def extract_assistant_response(output_text):
    """Model-specific code to extract model responses.

    See this doc for LLaMA 3: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/."""
    # Split the output text by the assistant header token
    output_text = output_text.replace("<|im_end|>", "").strip()
    parts = output_text.split("<|start_header_id|>assistant<|end_header_id|>")


    if len(parts) > 1:
        # Join the parts after the first occurrence of the assistant header token
        response = parts[1].split("<|eot_id|>")[0].strip()

        # Remove any remaining special tokens and whitespace
        response = response.replace("<|im_end|>", "").strip()

        return response
    else:
        return output_text
