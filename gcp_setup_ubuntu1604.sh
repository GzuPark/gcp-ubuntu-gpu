#!/bin/bash

# Install NVIDIA drivers
sudo apt-get update && sudo add-apt-repository ppa:graphics-drivers/ppa && \
sudo apt-get update && sudo apt-get install nvidia

# Check for CUDA and try to install.
# https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/base/Dockerfile
sudo apt-get update && sudo apt-get install -y --no-install-recommends ca-certificates apt-transport-https gnupg-curl && sudo rm -rf /var/lib/apt/lists/* && \
NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
sudo apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
sudo sh -c 'echo "$NVIDIA_GPGKEY_SUM  cudasign.pub"' | sha256sum -c --strict - && sudo rm cudasign.pub && \
sudo sh -c 'echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list' && \
sudo sh -c 'echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list'

export CUDA_VERSION=9.0.176
export CUDA_PKG_VERSION=9-0=$CUDA_VERSION-1

sudo apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION

curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_${CUDA_VERSION}-1_amd64.deb
sudo dpkg -i ./cuda-repo-ubuntu1604_${CUDA_VERSION}-1_amd64.deb
sudo apt-get update && sudo apt-get install -y --no-install-recommends cuda=${CUDA_VERSION}-1
sudo rm ./cuda-repo-ubuntu1604_${CUDA_VERSION}-1_amd64.deb

ln -s cuda-9.0 /usr/local/cuda

sudo sh -c 'echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf'
sudo sh -c 'echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf'

export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# NCCL install
# https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/runtime/Dockerfile
export NCCL_VERSION=2.2.13

sudo apt-get update && sudo apt-get install -y --no-install-recommends \
        cuda-libraries-$CUDA_PKG_VERSION \
        cuda-cublas-9-0=9.0.176.3-1 \
        libnccl2=$NCCL_VERSION-1+cuda9.0


# CUDNN install
# https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/runtime/cudnn7/Dockerfile
export CUDNN_VERSION=7.1.4.18

sudo apt-get update && sudo apt-get install -y --no-install-recommends \
        libcudnn7=$CUDNN_VERSION-1+cuda9.0

sudo rm -rf /var/lib/apt/lists/*

# Tensorflow
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-0 \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 \
        curl \
        cmake \
        libcudnn7=7.1.4.18-1+cuda9.0 \
        libnccl2=2.2.13-1+cuda9.0 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        python3-dev \
        python-tk3\
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
sudo apt-get update
curl -O https://repo.continuum.io/archive/Anaconda3-5.3.1-Linux-x86_64.sh

# recommend install directory
# /usr/bin/anaconda3
bash ./Anaconda3-5.3.0-Linux-x86_64.sh -p /usr/bin/anaconda3 -b
sudo rm ./Anaconda3*.sh

# if you want to change default python
ln -s -f /usr/bin/python3 /usr/bin/python
# ln -s -f /usr/bin/anaconda3/bin/python3 /usr/bin/python
# ln -s -f /usr/bin/anaconda3/bin/pip /usr/bin/pip
# ln -s -f /usr/bin/anaconda3/bin/jupyter /usr/bin/jupyter
# conda set up
ln -s -f /usr/bin/anaconda3/bin/conda /usr/bin/conda

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
        opencv-python \
        scikit-image \
        tensorflow-gpu \ 
        torch \
        torchvision

python -m ipykernel.kernelspec

# jupyter notebook
jupyter notebook --generate-config
wget https://raw.githubusercontent.com/GzuPark/gcp-ubuntu-gpu/master/jupyter_notebook_config.py
# chmod 666 jupyter_notebook_config.py
mv jupyter_notebook_config.py .jupyter/
# chmod 777 .jupyter/

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

sudo sh -c 'echo PATH=$PATH > /etc/environment'
sudo sh -c 'echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH >> /etc/environment'

echo PATH=$PATH >> .bashrc
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH >> .bashrc
source .bashrc
# echo PATH=$PATH >> /root/.bashrc
# echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH >> /root/.bashrc
# source /root/.bashrc

# Screen setting
wget https://raw.githubusercontent.com/GzuPark/gcp-ubuntu-gpu/master/.screenrc
# chmod 666 .screen

# Install docker-ce
# https://docs.docker.com/install/linux/docker-ce/ubuntu/
sudo apt-get update && sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg2 \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

sudo apt-get update && sudo apt-get install docker-ce

# Install nvidia-docker
# https://github.com/NVIDIA/nvidia-docker/README.md
# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

# Install nvidia-docker2 and reload the Docker daemon configuration
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd

# Manage Docker as a non-root user
# https://docs.docker.com/install/linux/linux-postinstall/
sudo groupadd docker
sudo usermod -aG docker $USER
# it will work after re-login

apt-get update && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*

# Test
sudo docker run --rm nvidia/cuda nvidia-smi
sudo docker rmi nvidia/cuda

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

rm gcp_setup_ubuntu1604.sh
