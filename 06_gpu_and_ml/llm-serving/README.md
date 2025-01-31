Trying to install from source:
```
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/local_repo/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6_1.0-1_amd64.deb
cp /var/nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6/nv-tensorrt-local-F234AD55-keyring.gpg /usr/share/keyrings/
dpkg -i nv-tensorrt-local-repo-ubuntu2204-10.7.0-cuda-12.6_1.0-1_amd64.deb 
sudo apt-get install libnvinfer-dev

# TensorRT-LLM uses git-lfs, which needs to be installed in advance.
apt-get update && apt-get -y install git git-lfs
git lfs install

git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout 71213f726620a259e03ba5a64b9f3c6adee2df17
git submodule update --init --recursive
git lfs pull

python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt
```
