#!/bin/bash

# Check for CUDA and try to install.
# https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/base/Dockerfile

apt-get update && apt-get install -y --no-install-recommends ca-certificates apt-transport-https gnupg-curl && \
        rm -rf /var/lib/apt/lists/* && \
        NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
        NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
        apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
        apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
        echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
        # echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
        echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

export CUDA_VERSION=9.0.176
export CUDA_PKG_VERSION=9-0=$CUDA_VERSION-1

apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION

curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_${CUDA_VERSION}-1_amd64.deb
dpkg -i ./cuda-repo-ubuntu1604_${CUDA_VERSION}-1_amd64.deb
apt-get update && apt-get install -y --no-install-recommends cuda=${CUDA_VERSION}-1
rm ./cuda-repo-ubuntu1604_${CUDA_VERSION}-1_amd64.deb

ln -s cuda-9.0 /usr/local/cuda

echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf
echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64


# NCCL install
# https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/runtime/Dockerfile

export NCCL_VERSION=2.2.13

apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-$CUDA_PKG_VERSION \
        cuda-cublas-9-0=9.0.176.3-1 \
        libnccl2=$NCCL_VERSION-1+cuda9.0


# CUDNN install
# https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/runtime/cudnn7/Dockerfile

export CUDNN_VERSION=7.1.4.18

apt-get update && apt-get install -y --no-install-recommends \
        libcudnn7=$CUDNN_VERSION-1+cuda9.0

rm -rf /var/lib/apt/lists/*


# Tensorflow
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu

apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-0 \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 \
        curl \
        libcudnn7=7.1.4.18-1+cuda9.0 \
        libnccl2=2.2.13-1+cuda9.0 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip

# If E: something occured, please try it like below
# https://itsfoss.com/fix-ubuntu-install-error/
# sudo killall apt-get
# sudo rm /var/lib/apt/lists/lock
# sudo rm /var/cache/apt/archives/lock
# sudo rm /var/lib/dpkg/lock
# sudo dpkg --configure -a

# Default: Anaconda3
# latest version
# https://repo.continuum.io/archive/
apt-get update
curl -O https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh

# recommend install directory
# /usr/bin/anaconda3
bash ./Anaconda3-5.2.0-Linux-x86_64.sh -p /usr/bin/anaconda3 -b

# if you want to change default python
ln -s -f /usr/bin/python3 /usr/bin/python
# ln -s -f /usr/bin/anaconda3/bin/python3 /usr/bin/python
# ln -s -f /usr/bin/anaconda3/bin/pip /usr/bin/pip
# ln -s -f /usr/bin/anaconda3/bin/jupyter /usr/bin/jupyter
# conda set up
ln -s -f /usr/bin/anaconda3/bin/conda /usr/bin/conda

rm ./Anaconda3*.sh

curl -O https://bootstrap.pypa.io/get-pip.py && \
        python get-pip.py && \
        rm get-pip.py

pip --no-cache-dir install \
        msgpack \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        pandas \
        scipy \
        sklearn \
        tensorflow-gpu==1.9.0

python -m ipykernel.kernelspec

# jupyter notebook
jupyter notebook --generate-config
wget https://raw.githubusercontent.com/GzuPark/gcp-ubuntu-gpu/master/jupyter_notebook_config.py
chmod 666 jupyter_notebook_config.py
mv jupyter_notebook_config.py .jupyter/
chmod 777 .jupyter/

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

echo PATH=$PATH > /etc/environment
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH >> /etc/environment

# TODO: nvcc error with root account
echo PATH=$PATH >> .bashrc
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH >> .bashrc
source .bashrc
echo PATH=$PATH >> /root/.bashrc
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH >> /root/.bashrc
source /root/.bashrc

# PyTorch
pip --no-cache-dir install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
pip --no-cache-dir install torchvision

# Screen setting
wget https://raw.githubusercontent.com/GzuPark/gcp-ubuntu-gpu/master/.screenrc
chmod 666 .screen

# download from a package
# https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-from-a-package
# https://download.docker.com/linux/ubuntu/dists/xenial/pool/stable/amd64/
wget https://download.docker.com/linux/ubuntu/dists/xenial/pool/stable/amd64/docker-ce_18.06.0~ce~3-0~ubuntu_amd64.deb
wget https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb

dpkg -i docker-ce_*.deb

# https://github.com/jenkinsci/docker/issues/506#issuecomment-305867728
apt-get update && apt-get install -y libltdl7 && \
        rm -rf /var/lib/apt/lists/*

dpkg -i nvidia-docker*.deb

rm docker*.deb
rm nvidia-docker*.deb

# Test Run: nvidia-docker
# nvidia-docker run --rm nvidia/cuda nvidia-smi
# nvidia-docker rmi nvidia/cuda

apt-get update && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*

python -c 'import tensorflow as tf; \
                print("__TensorFlow GPU device"); \
                tf.test.gpu_device_name()'
python -c 'from tensorflow.python.client import device_lib; \
                print("\n"); \
                print("__TensorFlow device library"); \
                device_lib.list_local_devices()'
python -c 'import torch; import sys; import os;\
                print("\n"); \
                print("__Python VERSION:", sys.version); \
                print("__pyTorch VERSION:", torch.__version__); \
                print("__CUDA VERSION"); \
                os.system("nvcc --version"); \
                print("__CUDNN VERSION:", torch.backends.cudnn.version()); \
                print("__Number CUDA Devices:", torch.cuda.device_count()); \
                print("__Devices"); \
                os.system("nvidia-smi --format=csv --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"); \
                print("Active CUDA Device: GPU", torch.cuda.current_device()); \
                print ("Available devices ", torch.cuda.device_count()); \
                print ("Current cuda device ", torch.cuda.current_device())'

rm setup.sh
