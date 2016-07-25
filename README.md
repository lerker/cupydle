# cupydle [![Build Status](https://travis-ci.org/lerker/cupydle.svg?branch=master)](https://travis-ci.org/lerker/cupydle) [![GitHub issues](https://img.shields.io/github/issues/lerker/cupydle.svg?style=plastic)](https://github.com/lerker/cupydle/issues) [![GitHub license](https://img.shields.io/badge/license-Apache%202-blue.svg?style=plastic)](https://raw.githubusercontent.com/lerker/cupydle/master/LICENSE) [![GitHub forks](https://img.shields.io/github/forks/lerker/cupydle.svg?style=plastic)](https://github.com/lerker/cupydle/network) [![todofy badge](https://todofy.org/b/lerker/cupydle)](https://todofy.org/r/lerker/cupydle)

**CU**da**PY**thon**D**eep**LE**arning Machine Learning

Introduction
============
_cupydle_ is the fastest and easiest way to use Restricted Boltzmann
Machines (RBMs). RBMs are a class of probabilistic models that can discover
hidden patterns in your data. _cupydle_ provides all the necessary methods with
a pythonic interface, and moreover, all methods call blazing fast C code. The
code can also run transparently on GPU thanks to
Theano (http://deeplearning.net/software/theano/).


:white_check_mark: finished

:negative_squared_cross_mark: not done

:interrobang: in progress or not finished

:bangbang: very important and not done



Functionality:

- Restricted Boltzmann Machine Training
  - [x] With n-step Contrastive Divergence
  - [ ] With persistent Contrastive Divergence
  - [ ] Weight decay, momentum, batch-learning
  - [ ] Binary or gaussian visible nodes

- Restricted Boltzmann Machine Evaluation
  - Sampling from the model
  - Visualizing Filters
  - Annealed Importance Sampling for approximating the partition function
  - Calculating the partition function exactly
  - Visualization and saving of hidden representations

- Stacking RBMs to Deep Belief Networks
  - :negative_squared_cross_mark: Sampling from DBNs


- Neural Network Traing
  - Backpropagation of error
  - RPROP
  - Weight decay, momentum, batch-learning
  - Variable number of layers
  - Cross entropy training

- Finetuning
  - Initalizing a Neural Network with an RBM
  - All of the above functionality can be used

- Training on Image Data
  - Visualization of input, filters and samples from the model
  - on-the-fly modifications to trainingset via gaussian noise or translations

GPU Notes
=========
In host must be installed nvidia's driver correcpond to device model (e.i Tesla 1060 == _nvidia driver 340_)
In ubuntu
- sudo apt-get install nvidia-340 nvidia-340-uvm libcuda1-340
 - wget http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_linux_64.run

Cuda toolkit 6.5 is required as minimum _gcc 4.6_
controlar las variables de entorno
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export PYTHONPATH=/root/cupydle/:${PYTHONPATH}

cat /proc/driver/nvidia/version
nvidia-smi
nvcc --version


Docker Notes
============
# para que cuda funcione se debe correr en el host deviceQuery una vez (carga las dev)
# crear la imagen (evidentemente no funciona con docker solo desde el dockerfile debido a que los devices no son conectados en el build), crear un contenedor de ubuntu 14.04 y correr linea por linea del dockerfile
# docker build -t "ubuntu/cuda/theano/python3" .
## sudo docker run -it --privileged=true --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm --device /dev/nvidia0:/dev/nvidia ubuntu:14.04 /bin/bash
# correr en modo previligiado y cargar los devices (probar con docker-nvidia o docker solo)
# sudo nvidia-docker run -it --privileged=true --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm --device /dev/nvidia0:/dev/nvidia ubuntu/cuda/theano/python3 /bin/bash

# contenedores (-q solo ids)
docker ps -a
# imagenes
docker images -a
# borrar contenedor [ docker rm $(docker ps -aq) ]
docker rm <id>
# borrar imagen
docker rmi <id>
# crear imagen
docker build -t "nombre/de/mi/imagen/:tag" /path:dockerfile
#correr contenedor (iterativo -i, borrar contenedor al salir --rm)
sudo docker run -it --privileged=true --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm --device /dev/nvidia0:/dev/nvidia <imagen_id> /bin/bash
# _salir sin exit_ ctrl p y luego ctrl q
# se sale, buscar el id del docker
docker ps -q
# commitear los cambios
docker commit <id> <nombre>

#volver al container
docker attach <id>

# copiar un archivo del container en ejecucion
docker cp <id>:absolut/path/to/file destination/path/in/host


Authors and Contributors
========================
Ponzoni Cuadra, Nelson E. (@lerker)

Support or Contact
==================
Having trouble? Mail me npcuadra@gmail.com
