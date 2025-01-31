import json
from pathlib import Path
from typing import Optional

import modal
import pydantic  # for typing, used later

tensorrt_llm_version = "0.16.0"


if tensorrt_llm_version == "0.17.0":
    GIT_HASH = "71213f726620a259e03ba5a64b9f3c6adee2df17"
if tensorrt_llm_version == "0.16.0":
    GIT_HASH = "42a7b0922fc9e095f173eab9a7efa0bcdceadd0d"
elif tensorrt_llm_version == "0.14.0":
    GIT_HASH = "b0880169d0fb8cd0363049d91aa548e58a41be07"
else:
    raise Exception("e")


tensorrt_image = modal.Image.from_registry(
    "nvidia/cuda:12.6.3-devel-ubuntu22.04",
    add_python="3.12",  # TRT-LLM requires Python 3.10
).entrypoint([])  # remove verbose logging by base image on entry

tensorrt_image = tensorrt_image.apt_install(
    "openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget"
).pip_install(
    f"tensorrt_llm=={tensorrt_llm_version}",
    "pynvml<12",  # avoid breaking change to pynvml version API
    pre=True,
    extra_index_url="https://pypi.nvidia.com",
)

volume = modal.Volume.from_name(
    "example-trtllm-volume", create_if_missing=True
)
VOLUME_PATH = Path("/vol/model") 

MODEL_DIR = TOKENIZER_DIR = VOLUME_PATH / "model_input" 
# MODEL_ID = "unsloth/Qwen2.5-Coder-7B-Instruct"
# MODEL_REVISION = "3fd3aab092612530a892ff49027dfd4f39046ec3"
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"


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
        volumes={VOLUME_PATH: volume},
    )
)

CONVERSION_SCRIPT_URL = f"https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/{GIT_HASH}/examples/qwen/convert_checkpoint.py"


N_GPUS = 1  # TODO
GPU_CONFIG = modal.gpu.H100(count=N_GPUS)

DTYPE = "float16"  # format we download in, regular fp16
QFORMAT = "fp8"  # format we quantize the weights to
KV_CACHE_DTYPE = "fp8"  # format we quantize the KV cache to

CALIB_SIZE = "512"  # size of calibration dataset

# We put that all together with another invocation of `.run_commands`.

# QUANTIZATION_ARGS = f"--dtype={DTYPE} --qformat={QFORMAT} --kv_cache_dtype={KV_CACHE_DTYPE} --calib_size={CALIB_SIZE}"
# INT4 from: https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/qwen/README.md
# TODO KV CACHE quant?
QUANTIZATION_ARGS = f"--dtype={DTYPE} --use_weight_only --weight_only_precision int4"
QUANTIZATION_ARGS = f"--dtype={DTYPE}"
# TODO INT4-GPTQ instructions or quantize instructions like orginal llama example


def quantize_model():
    import subprocess
    import urllib.request
    urllib.request.urlretrieve(CONVERSION_SCRIPT_URL, "/root/convert.py")
    subprocess.run(
        [
            "python", "/root/convert.py", f"--model_dir={MODEL_DIR}",
            f"--output_dir={CKPT_DIR}", f"--tp_size={N_GPUS}",
        ] + QUANTIZATION_ARGS.split(" ")
    )

CKPT_DIR = VOLUME_PATH / "model_ckpt"
# CKPT_DIR = "/root/model/model_ckpt"
tensorrt_image = (  # update the image by quantizing the model
    tensorrt_image.run_function(  # takes ~2 minutes
        quantize_model,
        gpu=GPU_CONFIG,
        volumes={VOLUME_PATH: volume},
    )
)

MAX_INPUT_LEN, MAX_OUTPUT_LEN = 8192, 128
MAX_IO_LEN = MAX_INPUT_LEN + MAX_OUTPUT_LEN
MAX_BATCH_SIZE = 16
ENGINE_DIR = VOLUME_PATH / "model_output" 
# ENGINE_DIR = "/root/model/model_output"

SIZE_ARGS = (
    f"--max_input_len={MAX_INPUT_LEN} --max_num_tokens={MAX_OUTPUT_LEN} "
    f"--max_seq_len={MAX_IO_LEN} --max_batch_size={MAX_BATCH_SIZE}"
)

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

def trtllm_build():
    import subprocess
    subprocess.run(
        [
            "trtllm-build", "--checkpoint_dir", CKPT_DIR, "--output_dir",
            ENGINE_DIR, "--gemm_plugin=float16", f"--workers={N_GPUS}",
            # TODO: broken
            # f"--speculative_decoding_mode=lookahead_decoding",
            # f"--max_draft_len={max_draft_len}",
            # f"--max_beam_width=1",
            # f"--gpt_attention_plugin=float16",
        ] + SIZE_ARGS.split(" ")
    )


tensorrt_image = (  # update the image by building the TensorRT engine
    tensorrt_image.run_function(  # takes ~5 minutes
        trtllm_build,
        gpu=GPU_CONFIG,
        volumes={VOLUME_PATH: volume},
    ).env(  # show more log information from the inference engine
        {"TLLM_LOG_LEVEL": "INFO"}
    )
)


app = modal.App(
    f"example-trtllm-{MODEL_ID.split('/')[-1].replace('.', '')}", image=tensorrt_image
)

@app.cls(
    gpu=GPU_CONFIG,
    container_idle_timeout=10 * MINUTES,
    image=tensorrt_image,
    volumes={VOLUME_PATH: volume},
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

        # Copied from: https://github.com/NVIDIA/TensorRT-LLM/blob/d93a2dde84eada06ae2339b4fb4e6432167a1cfd/examples/utils.py#L173
        self.tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_DIR,
                legacy=False,
                padding_side='left',
                truncation_side='left',
                trust_remote_code=True,
                tokenizer_type=None,
                use_fast=True
        )
        # This seems wrong:
        self.pad_id = self.tokenizer.pad_token_id
        self.end_id = self.tokenizer.eos_token_id
        # This should work but doesn't bc eos_token_id is a list instead of int
        # gen_config = json.loads((TOKENIZER_DIR / "generation_config.json").read_text())
        # self.pad_id = gen_config['pad_token_id']
        # self.end_id = gen_config['eos_token_id']

        runner_kwargs = dict(
            engine_dir=f"{ENGINE_DIR}",
            lora_dir=None,
            rank=tensorrt_llm.mpi_rank(),  # TODO: this will need to be adjusted to use multiple GPUs
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
        import torch

        if settings is None or not settings:
            settings = dict(
                temperature=0.1,  # temperature 0 not allowed, so we set top_k to 1 to get the same effect
                top_k=1,
                stop_words_list=None,
                repetition_penalty=1.1,
            )

        settings["max_new_tokens"] = MAX_OUTPUT_LEN
        settings["end_id"] = self.end_id
        settings["pad_id"] = self.pad_id
        # TODO: Broken in 0.16 with Qwen?
        # settings["lookahead_config"] = [windows_size, ngram_size, verification_set_size]

        assert(len(prompts) <= MAX_BATCH_SIZE)

        start = time.monotonic_ns()

        batch_input_ids = []
        for curr_text in prompts:
            input_ids = self.tokenizer.encode(
                curr_text,
                add_special_tokens=True,
                truncation=True,
                max_length=MAX_INPUT_LEN,
            )
            batch_input_ids.append(input_ids)

        batch_input_ids = [
            torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
        ]

        outputs = self.model.generate(
            batch_input_ids=batch_input_ids,
            **settings
        )
        num_tokens = -1
        outputs_text = self.tokenizer.batch_decode(outputs[:,0])
        # TODO Hack to remove erroneous im_ends:
        outputs_text = [o.replace("<|im_end|>", "") for o in outputs_text]
        for o in outputs_text:
            print('len:', len(o))


        duration_s = (time.monotonic_ns() - start) / 1e9

        i = 0
        for prompt, response in zip(prompts, responses):
            print("\n\n{str(i)*100}\n\n")
            print(
                f"{COLOR['HEADER']}{COLOR['GREEN']}{prompt}",
                f"\n{COLOR['BLUE']}{response}",
                "\n\n",
                sep=COLOR["ENDC"],
            )
            i += 1

        print(
            f"{COLOR['HEADER']}{COLOR['GREEN']}Generated {num_tokens} tokens from {MODEL_ID} in {duration_s:.1f} seconds,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second for batch of size {len(prompts)} on {GPU_CONFIG}.{COLOR['ENDC']}"

        )

        return responses


@app.local_entrypoint()
def main():
    import glob
    import json
    from pathlib import Path

    prompts = []
    for filepath in glob.glob("reqs/*.json"):
        input_json = json.loads(Path(filepath).read_text())
        prompts.append(input_json["prompt"][:4000])
    prompts = prompts[-1:]
        
    model = Model()
    model.generate.remote(prompts)


COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}
