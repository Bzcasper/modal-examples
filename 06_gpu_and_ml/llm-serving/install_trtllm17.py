import modal

app = modal.App("trtllm17")

GIT_HASH = "71213f726620a259e03ba5a64b9f3c6adee2df17"

cuda_version = "12.6.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
os = "ubuntu22.04"
cuda_tag = f"{cuda_version}-{flavor}-{os}"

trt_version = "10.7.0"
cuda_version = ".".join(cuda_version.split(".")[:-1])
assert len(cuda_version.split(".")) == 2 # For nv_tag
nv_tag = f"{trt_version}-cuda-{cuda_version}"
nv_name = f"nv-tensorrt-local-repo-{os.replace('.', '')}-{nv_tag}_1.0-1_amd64.deb"

image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{cuda_tag}",
        add_python="3.12",  # TRT requires Python3.12 or Python 3.8
    ).apt_install(
        "openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget", "curl", "cmake", "build-essential"
    ).run_commands(
        f"wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/local_repo/{nv_name}",
        f"dpkg -i {nv_name}",
        f"cp /var/nv-tensorrt-local-repo-*-{nv_tag}/*-keyring.gpg /usr/share/keyrings/",
    ).apt_install(
      "tensorrt",
    ).run_commands(
        "git lfs install",
        "git clone https://github.com/NVIDIA/TensorRT-LLM.git",
        f"cd TensorRT-LLM && git checkout {GIT_HASH}",
        "cd TensorRT-LLM && git submodule update --init --recursive",
        "cd TensorRT-LLM && git lfs pull",
        "cd TensorRT-LLM && python3 ./scripts/build_wheel.py --clean",
    )
)

@app.function(image=image, gpu="A100")
def foo():
    pass

