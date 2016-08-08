# CUPYDLE
[![Build Status](https://travis-ci.org/lerker/cupydle.svg?branch=master)](https://travis-ci.org/lerker/cupydle) [![GitHub issues](https://img.shields.io/github/issues/lerker/cupydle.svg?style=plastic)](https://github.com/lerker/cupydle/issues) [![GitHub license](https://img.shields.io/badge/license-Apache%202-blue.svg?style=plastic)](https://raw.githubusercontent.com/lerker/cupydle/master/LICENSE) [![GitHub forks](https://img.shields.io/github/forks/lerker/cupydle.svg?style=plastic)](https://github.com/lerker/cupydle/network) [![todofy badge](https://todofy.org/b/lerker/cupydle)](https://todofy.org/r/lerker/cupydle)

**CU**da**PY**thon**D**eep**LE**arning Machine Learning

Introducción
============
_cupydle_ es una libreria simple y sencilla, que provee todas las funcionalidades necesarias para la ejecucion de Redes de Creencia Profunda (DBN) sobre placas de procesamiento grafico para proposito general (GP-GPU).
_cupydle_ posse una inteface 'pythonica', y además, el codigo puede ser ejecutado de forma transparente tanto en GPU como en CPU gracias a Theano (http://deeplearning.net/software/theano/).


:white_check_mark: finished

:negative_squared_cross_mark: not done

:interrobang: in progress or not finished

:bangbang: very important and not done



Funcionalidades:
================

- Maquinas de Boltzmann Restringidas (RBM)
  - Entrenamiento
    - [x] Algoritmo de Divergencia Constrastiva de n pasos de Gibbs (CD_n)
    - [x] Algoritmo de Divergencia Constrastiva Persistente de n pasos de Gibbs (PCD_n)
    - [ ] Weight decay, momento, batch-learning
    - [x] Unidades Ocultas Binarias.
    - [ ] Unidades Ocultas Gausianas.
    - [x] Función de costo, error cuadratico medio (MSE).
    - [x] Función de costo, entropia cruzada.
    - [x] Función de costo, Energia Libre.

  - Evaluación
    - [x] Muestreo desde el modelo.
    - [x] Visialización de Filtros.
    - [x] Almacenamiento y recuperación de las representaciones ocultas.

- Redes de Creencia Profunda (DBN)
  - Entrenamiento (_no supervisado_)
    - [x] Apilado de RBMs en capas.
    - [ ] Almacenamiento.
  - Entrenamiento (_supervisado_)
    - [x] Recuperacion de las capas, pesos ya entrenados.
    - [x] Ajuste de los pesos por medio de un Perceptron Multi-Capa (MLP).
      - [x] Inicialización de los pesos.

- Redes Neuronales Generales (MLP)
  - Entrenamiento
    - [x] Numero variable de capas, capas logisticas, softmax.
    - [x] Funciones de activacion varias, Sigmoid.
    - [x] Algoritmo del Gradiente Descendiente Estocastico (SGD)
      - [x] Calculo de gradientes, retropropagacion del error.
      - [ ] Tecnicas, 'patience'....
    - [x] Weight decay, momentum, batch-learning.
    - [x] Función de costo, entropia cruzada.
    - [x] Función de costo, error cuadratico medio (MSE).

- Pruebas
  - MNIST
    - [x] Preparación de los datos, descarga y manipulación.
    - [x] Visualización de los datos.
    - [x] Visualización de los filtros. :interrobang: (realizar la funcion en rbm que genere los filtro a partir de patches cuadrados...)
    - [x] Muestreo de varios ejemplos a traves de sucesivas cadenas de Markov.

Notas GP-GPU
============
En la maquina Host debe estar instalado el driver correspondiente Nvidia al modelo de la placa, en el caso de pruebas es Nvidia

```bash
~$: /proc/driver/nvidia/gpus/0000\:01\:00.0/information

Model:           Tesla C1060
IRQ:             46
GPU UUID:        GPU-ffb9af25-05ad-7d83-5e0b-a397677ec9fe
Video BIOS:      62.00.7a.00.05
Bus Type:        PCIe
DMA Size:        40 bits
DMA Mask:        0xffffffffff
Bus Location:    0000:01:00.0
```




cat /proc/driver/nvidia/version
NVRM version: NVIDIA UNIX x86_64 Kernel Module  340.29  Thu Jul 31 20:23:19 PDT 2014
GCC version:  gcc version 4.6.4 (Ubuntu/Linaro 4.6.4-6ubuntu2)


In host must be installed nvidia's driver correcpond to device model (e.i Tesla 1060 == _nvidia driver 340_)
In ubuntu
- sudo apt-get install nvidia-340 nvidia-340-uvm libcuda1-340
 - wget http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_linux_64.run

Cuda toolkit 6.5 is required as minimum _gcc 4.6_
./cuda_6.5.14_linux_64.run --silent --driver
./NVIDIA_CUDA-6.5_Samples/bin/x86_64/linux/release/deviceQuery
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



#modificaciones nuevas. solo anda con cuda 6.0

./usr/local/cuda-6.5/bin/uninstall_cuda_6.5.pl

wget http://developer.download.nvidia.com/compute/cuda/6_0/rel/installers/cuda_6.0.37_linux_64.run

chmod a+x cuda_6.0.37_linux_64.run
./cuda_6.0.37_linux_64.run

pip3 uninstall Theano

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export PYTHONPATH=/root/cupydle/:${PYTHONPATH}

