# Quick Set Up NVIDIA GPUs with GCP VM
* OS: `Ubuntu 16.04`
* GPU: `NVIDIA K80, P100, V100`
    * CUDA: `9.0.176`
    * NCCL: `2.2.13`
    * CUDNN: `7.1.4.18`
* Python: `3.5.2`
    * TensorFlow: `1.9.0`
    * PyTorch: `0.4.0`
    * Other ML libraries
* Programs:
    * Anaconda: `5.2.0`, Python: `3.6.2` (is not default)
    * nvidia-docker: `1.0.1`

# How to use
* You __MUST__ run `setup.sh` file with __ROOT__ account.
* Before you run this file, please check your OS __Ubuntu 16.04__.
* You can modify this code if you want to use other OS or CUDA version.
```
wget https://raw.githubusercontent.com/GzuPark/gcp-ubuntu-gpu/master/setup.sh
sudo bash
bash setup.sh
```

# Etc

## Anaconda
* If you want to change from default python interpreter to Anaconda:
```
sudo bash
ln -s -f /usr/bin/anaconda3/bin/python3 /usr/bin/python
ln -s -f /usr/bin/anaconda3/bin/pip /usr/bin/pip
ln -s -f /usr/bin/anaconda3/bin/jupyter /usr/bin/jupyter
```

## nvidia-docker
* If you want to test nvidia-docker:
```
sudo nvidia-docker run --rm nvidia/cuda nvidia-smi
sudo nvidia-docker rmi nvidia/cuda
```

# Issues
Please comment at the [Issues tap](https://github.com/GzuPark/gcp-ubuntu-gpu/issues).

