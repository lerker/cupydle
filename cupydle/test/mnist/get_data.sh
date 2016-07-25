#!/bin/bash

if [ -f "cupydle/data/DB_mnist/t10k-labels-idx1-ubyte" -a "cupydle/data/DB_mnist/train-labels-idx1-ubyte" -a "cupydle/data/DB_mnist/t10k-images-idx3-ubyte" -a "cupydle/data/DB_mnist/train-images-idx3-ubyte" ]; then
    #echo "DB_mnist directory already present, exiting"
    #echo "Los archivos ya estan en la carpeta DB_mnist http://yann.lecun.com/exdb/mnist/ "
    exit 1
fi

if [ ! -d "cupydle/data/DB_mnist" ]; then
    mkdir -p cupydle/data/DB_mnist
fi
echo "Descargando los archivos desde http://yann.lecun.com/exdb/mnist/"

pushd cupydle/data/DB_mnist
#wget --recursive --level=1 --cut-dirs=3 --no-host-directories \
#  --directory-prefix=DB_mnist --accept '*.gz' http://yann.lecun.com/exdb/mnist/
#pushd DB_mnist
wget --recursive --level=1 --cut-dirs=3 --no-host-directories --accept '*.gz' http://yann.lecun.com/exdb/mnist/

echo "Descomprimiendo..."
gunzip *

echo "Los archivos estan en discos..."
popd
