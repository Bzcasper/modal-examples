import modal
import numpy as np
from pathlib import Path

CUDA_VERSION = "12.8.0"

app = modal.App("example-whisper-turbo")

volume = modal.Volume.from_name(
    "example-whisper-turbo", create_if_missing=True
)

VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"

image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_VERSION}-devel-ubuntu22.04",
        add_python="3.12",  # TRT-LLM requires Python 3.10
    ).apt_install(
        "ffmpeg",
    ).pip_install(
        "datasets==3.3.1",
        "hf-transfer==0.1.9",
        "accelerate==1.3.0",
        "huggingface_hub==0.28.1",
        "transformers==4.49.0",
        "torch==2.6.0",
        "librosa==0.10.2.post1",
        "soundfile==0.13.1",
    ).pip_install(
        # This made things slower:
        # "flash-attn==2.7.4.post1", extra_options="--no-build-isolation"
    ).env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": str(MODELS_PATH),
        }
    ).add_local_file(
        "wavs/thirty_strict_compressed.wav", "/root/thirty_strict_compressed.wav"
    )
)

@app.cls(
    image=image,
    volumes={VOLUME_PATH: volume},
    gpu="H100:1",
)
class Model:
    def __init__(self, model_id):
        self.model_id = model_id

    @modal.enter()
    def enter(self):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        from datasets import load_dataset
        import torch

        device = 'cuda'
        torch_dtype = torch.float16

        print ("loading model")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            # attn_implementation="flash_attention_2", (slower)
        )
        model.to(device)

        print ("loading processor")
        processor = AutoProcessor.from_pretrained(self.model_id)

        print ("loading pipeline")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            chunk_length_s=15,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

        print("loading dataset")
        self.dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")

        print('running dry run')
        self.pipe(self.get_truncated_sample())

    def get_truncated_sample(self):
        """Truncate the sample to 30 seconds"""
        sample = self.dataset[0]["audio"]
        Fs = sample['sampling_rate']
        num_seconds = 30
        sample['array'] = sample['array'][:num_seconds*Fs]
        return sample

    @modal.method()
    def inference(self):
        import time

        start_time = time.perf_counter()
        result = self.pipe(self.get_truncated_sample())
        latency_s = time.perf_counter()-start_time

        return (result["text"], latency_s)

@app.local_entrypoint()
def main():
    model_ids = [
        "openai/whisper-large-v3",
        "openai/whisper-large-v3-turbo",
    ]

    n_runs = 40

    for model_id in model_ids:
        print(f"Benchmarking {model_id}")
        model = Model(model_id)
        latencies_s = []
        for i in range(n_runs):
            text, latency_s = model.inference.remote()
            if i == 0:
                print(f"Sanity Check One Output: {text}")
            latencies_s.append(latency_s)

        p50 = np.percentile(latencies_s, 50)
        p90 = np.percentile(latencies_s, 90)
        print(f"Results for {model_id} -- P50: {p50:.2f}, P90: {p90:.2f}")
