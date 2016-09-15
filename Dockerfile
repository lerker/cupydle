# para que cuda funcione se debe correr en el host deviceQuery una vez (carga las dev)
# crear la imagen (evidentemente no funciona con docker solo desde el dockerfile debido a que los devices no son conectados en el build), crear un contenedor de ubuntu 14.04 y correr linea por linea del dockerfile
# docker build -t "ubuntu/cuda/theano/python3" .
## sudo docker run -it --privileged=true --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm --device /dev/nvidia0:/dev/nvidia ubuntu:14.04 /bin/bash
# correr en modo previligiado y cargar los devices (probar con docker-nvidia o docker solo)
# sudo nvidia-docker run -it --privileged=true --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm --device /dev/nvidia0:/dev/nvidia ubuntu/cuda/theano/python3 /bin/bash

FROM ubuntu:14.04
MAINTAINER Ponzoni Nelson <npcuadra@gmail.com>
LABEL ubuntu="14.04" python="3" nvidia="340.29" cuda="6.5" theano="0.82"

# Main workdir
WORKDIR /root/

# Kernel, add repo for downloading
RUN echo "deb http://cz.archive.ubuntu.com/ubuntu precise-updates main" >> /etc/apt/sources.list

# Install packages and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends --force-yes \
    wget \
    build-essential \
    python3-numpy \
    python3-scipy \
    python3-dev \
    python3-pip \
    python3-nose \
    python3-matplotlib \
    python3-tk \
    libopenblas-dev \
    git \
    gcc-4.6 \
    g++-4.6 \
    linux-headers-3.2.0-97-generic # linux-headers-`uname -r`

# Install theano with python3
RUN pip3 install Theano

# Install Pillow
RUN pip3 install Pillow

#### probar si desistalo gcc-4.8 anda directamente

# Change the default gcc
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.6 20 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 10 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.6 20 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 10 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30 && \
    update-alternatives --set cc /usr/bin/gcc && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30 && \
    update-alternatives --set c++ /usr/bin/g++

# Setting up .theanorc for CUDA
RUN echo -e "[global]\ndevice=gpu\nfloatX=float32\noptimizer_including=cudnn\n[lib]\ncnmem=1\n[nvcc]\nfastmath=True\nflags=-D_FORCE_INLINES\n" >> /root/.theanorc

# Download driver, toolkit and samples
RUN wget --progress=bar:force http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_linux_64.run && \
    chmod a+x cuda_6.5.14_linux_64.run

# Install them all
# instala el driver nvidia 340.29 por defecto, en mi instalacion (340.96) es la unica forma de que funcione
RUN echo "Instalando driver" && ./cuda_6.5.14_linux_64.run -silent --driver --override && \
    echo "Instalando tooltkit" && ./cuda_6.5.14_linux_64.run -silent --toolkit --override && \
    echo "Instalando samples" && ./cuda_6.5.14_linux_64.run -silent --samples --samplespath=/home/ --override && \
    rm cuda_6.5.14_linux_64.run && \
    ln -s cuda-6.5 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Variables de entorno
# necesarias las variables para el correcto funcionamiento de cuda y cupydle
ENV PATH /usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PYTHONPATH /root/cupydle/

#cleanup
RUN apt-get clean && apt-get autoclean && rm -rf /var/lib/apt/lists/*

# Ejemplo
WORKDIR /root/NVIDIA_CUDA-6.5_Samples/1_Utilities/deviceQuery
RUN make && \
    ./deviceQuery && \
    rm -r /root/NVIDIA_CUDA-6.5_Samples && \
    cat /proc/driver/nvidia/version && \
    ls /dev/* | grep nvidia && \

### FIN
#RUN  && \
#    make && \
#    cd ../.. && ./bin/x86_64/linux/release/deviceQuery && \
#    cd ../ && \
#   rm -r NVIDIA_CUDA-6.5_Samples && \
#    cat /proc/driver/nvidia/version && \
#    ls /dev/* | grep nvidia

# Step 12: download cupydle
WORKDIR /root/
RUN git clone https://github.com/lerker/cupydle.git && \
    chmod a+x cupydle/runner.sh

