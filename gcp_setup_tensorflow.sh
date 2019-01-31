#!/bin/sh

# Default: Miniconda3
# latest version
# https://conda.io/en/latest/miniconda.html
sudo apt-get update
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# recommend install directory
# /usr/bin/anaconda3
sudo bash ./Miniconda3-latest-Linux-x86_64.sh -p /usr/bin/miniconda3 -b
conda list && sudo rm ./Miniconda3*.sh

# if you want to change default python
sudo ln -s -f /usr/bin/python3 /usr/bin/python
sudo ln -s -f /usr/bin/miniconda3/bin/conda /usr/bin/conda

curl -O https://bootstrap.pypa.io/get-pip.py && \
        sudo python get-pip.py --user && \
        sudo rm get-pip.py

# Install docker-ce
# https://docs.docker.com/install/linux/docker-ce/ubuntu/
sudo apt-get update && sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg2 \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88

sudo add-apt-repository -y \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

sudo apt-get update && sudo apt-get install -y docker-ce

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
        sudo rm -rf /var/lib/apt/lists/*

# Test
# sudo docker run --rm nvidia/cuda nvidia-smi
# sudo docker rmi nvidia/cuda
python -c 'import tensorflow as tf; \
                print("__TensorFlow GPU device"); \
                tf.test.gpu_device_name()'
python -c 'from tensorflow.python.client import device_lib; \
                print("\n"); \
                print("__TensorFlow device library"); \
                device_lib.list_local_devices()'

rm gcp_setup_tensorflow.sh
