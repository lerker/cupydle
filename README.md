# CUPYDLE
[![Build Status](https://travis-ci.org/lerker/cupydle.svg?branch=master)](https://travis-ci.org/lerker/cupydle) [![GitHub issues](https://img.shields.io/github/issues/lerker/cupydle.svg?style=plastic)](https://github.com/lerker/cupydle/issues) [![GitHub license](https://img.shields.io/badge/license-Apache%202-blue.svg?style=plastic)](https://raw.githubusercontent.com/lerker/cupydle/master/LICENSE) [![GitHub forks](https://img.shields.io/github/forks/lerker/cupydle.svg?style=plastic)](https://github.com/lerker/cupydle/network) [![todofy badge](https://todofy.org/b/lerker/cupydle)](https://todofy.org/r/lerker/cupydle)

**CU**da**PY**thon**D**eep**LE**arning Machine Learning

Introducción
------------
_cupydle_ es una libreria simple y sencilla, que provee todas las funcionalidades necesarias para la ejecucion de Redes de Creencia Profunda (DBN) sobre placas de procesamiento grafico para proposito general (GP-GPU).

_cupydle_ posse una inteface 'pythonica', y además, el codigo puede ser ejecutado de forma transparente tanto en GPU como en CPU gracias a Theano (http://deeplearning.net/software/theano/).


:white_check_mark: finished

:negative_squared_cross_mark: not done

:interrobang: in progress or not finished

:bangbang: very important and not done



Funcionalidades:
----------------
- Generales
  - [x] Optimizacion inteligente en la carga de memoria con los datos, chunk_size automatico
  - [x] Persistencia de los datos optimizada, unico archivo comprimido de dimension variable

- Maquinas de Boltzmann Restringidas (RBM)
  - Entrenamiento
    - [x] MacroBatch, copia por chunks hacia la GPU.
    - [x] Algoritmo de Divergencia Constrastiva de n pasos de Gibbs (CD_n)
    - [x] Algoritmo de Divergencia Constrastiva Persistente de n pasos de Gibbs (PCD_n)
    - [x] Momento, batch-learning
    - [x] Unidades Ocultas Binarias.
    - [x] Unidades Ocultas Gausianas.
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
    - [x] Almacenamiento.
  - Entrenamiento (_supervisado_)
    - [x] Recuperacion de las capas, pesos ya entrenados.
    - [x] Ajuste de los pesos por medio de un Perceptron Multi-Capa (MLP).
      - [x] Inicialización de los pesos.

- Redes Neuronales Generales (MLP)
  - Entrenamiento
    - [x] Numero variable de capas, capas logisticas, softmax.
    - [x] Funciones de activacion varias, Sigmoidea.
    - [x] Algoritmo del Gradiente Descendiente Estocastico (SGD)
      - [x] Calculo de gradientes, retropropagacion del error.
      - [ ] Tecnicas
        - [] patience
        - [x] rapid evaluation
    - [x] momento, batch-learning.
    - [x] Función de costo, entropia cruzada.
    - [x] Función de costo, error cuadratico medio (MSE).

- Validacion y testing
  - [x] Validacion Cruzada
    - [x] K_fold
    - [x] LabelKFold
    - [x] LeaveOneOut
    - [x] StratifiedKFold
  - [] Parameters Searching
    - [] Grid Search
    - [] Random Grid Search

- Pruebas
  - MNIST
    - [x] Preparación de los datos, descarga y manipulación.
    - [x] Visualización de los datos.
    - [x] Visualización de los filtros. :interrobang: (realizar la funcion en rbm que genere los filtro a partir de patches cuadrados...)
    - [x] Muestreo de varios ejemplos a traves de sucesivas cadenas de Markov.
  - RML
    - [x] Pre-procesamiento de datos crudos
      - [x] videos
      - [x] audio
    - [x] Reduccion de dimension de los datos crudos
      - [x] PCA a los videos
      - [x] f0_enegy, coeficientes ceptrales, windowing, etc a los audios
    - [x] Normalizacion de los datos
      - [x] z_score
      - [x] whitening
      - [x] min_max

Notas GP-GPU
------------
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

```bash
~$: cat /proc/driver/nvidia/version

NVRM version: NVIDIA UNIX x86_64 Kernel Module  340.29  Thu Jul 31 20:23:19 PDT 2014
GCC version:  gcc version 4.6.4 (Ubuntu/Linaro 4.6.4-6ubuntu2)
```

Pasos, en ubuntu, fue requerido por cuestiones de compatibilidad por la placa la version de CUDA 6.0, aunque versiones mas recientes (7.5) funcionan con normalidad:

```bash

:$ wget http://developer.download.nvidia.com/compute/cuda/6_0/rel/installers/cuda_6.0.37_linux_64.run

# o bien la 6.5
:$ wget http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_linux_64.run
```

Cuda Toolkit requiere como a lo mas la version de compilador __GCC 4.6__
```bash
:$ chmod a+x cuda_6.0.37_linux_64.run
:$ ./cuda_6.0.37_linux_64.run

# controlar las variables de entorno
:$ export PATH=/usr/local/cuda/bin:${PATH}
:$ export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
:$ export PYTHONPATH=/root/cupydle/:${PYTHONPATH}

# probar el correcto funcionamiento
:$ cd [ruta de los ejemplos]/NVIDIA_CUDA-6.0_Samples/deviceQuery ; make ; ./deviceQuery

:$ cat /proc/driver/nvidia/version
:$ nvidia-smi
:$ nvcc --version
```

Notas Docker
------------
La libreria fue probada sobre un entorno 'virtualizado' del estilo Docker.

A continuacion se detallan algunos pasos para su correcta intalación, los pasos se encuentran en el 'Dockerfile'.

1. Se debe ejecutar siempre antes del inicio del Docker Container el ejecutable deviceQuery para que la unidad GPU sea visible.

2. Crear la imagen a partir del Dockerfile, en el caso de que sea necesario ejecutar si o si el paso anterior este no funcionara, se debera realizar la instalacion manual (debido a que las unidades no son cargadas al inicio con el build). Alguno de los siguientes pasos:
  1. docker build -t "ubuntu/cuda/theano/python3" .
  2. sudo docker run -it --privileged=true --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm --device /dev/nvidia0:/dev/nvidia ubuntu:14.04 /bin/bash

      correr en modo previligiado y cargar los devices (probar con docker-nvidia o docker solo)

      sudo nvidia-docker run -it --privileged=true --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm --device /dev/nvidia0:/dev/nvidia ubuntu/cuda/theano/python3 /bin/bash

3. Cheat Sheet
  + contenedores (-q solo ids)

      docker ps -a

  + imagenes

      docker images -a

  + borrar contenedor [ docker rm $(docker ps -aq) ]

      docker rm <id>

  + borrar imagen

      docker rmi <id>

  + crear imagen

      docker build -t "nombre/de/mi/imagen/:tag" /path:dockerfile

  + correr contenedor (iterativo -i, borrar contenedor al salir --rm)

      sudo docker run -it --privileged=true --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm --device /dev/nvidia0:/dev/nvidia <imagen_id> /bin/bash

  + salir sin exit del contenedor

      ctrl+p y luego ctrl+q

  + commitear los cambios

      docker commit <id> <nombre>

  + volver al container

      docker attach <id>

  + copiar un archivo del container en ejecucion

      docker cp <id>:absolut/path/to/file destination/path/in/host


Autor
------------------------
Ponzoni Cuadra, Nelson E. (@lerker)

Contacto y Soporte
------------------
Algun problema? Mandame un mail a npcuadra@gmail.com
