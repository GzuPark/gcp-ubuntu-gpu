[[한국어 가이드]](https://medium.com/@jijupax/gcp-vm%EC%97%90%EC%84%9C-nvidia-gpu%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%B4%EB%B3%B4%EC%9E%90-d741e70365ac)

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
    * Anaconda: `5.2.0`, Python: `3.6.5` (is not default)
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
* After ending set up, please do like below:
```
# out from the root account
exit
source .bashrc
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

## Jupyter
* First, run this command:
```
jupyter notebook --allow-root
```
* And then, check your __EXTERNAL_IP__ of a instance, enter like below on your web-browser (ex: `10.100.0.8`)
```
10.100.0.8:8888
```

# Issues
Please comment at the [Issues tap](https://github.com/GzuPark/gcp-ubuntu-gpu/issues).

