#!/bin/bash

if [ -d DB_mnist ]; then
    #echo "DB_mnist directory already present, exiting"
    #echo "Los archivos ya estan en la carpeta DB_mnist http://yann.lecun.com/exdb/mnist/ "
    exit 1
fi

mkdir DB_mnist
echo "Descargando los archivos desde http://yann.lecun.com/exdb/mnist/"

wget --recursive --level=1 --cut-dirs=3 --no-host-directories \
  --directory-prefix=DB_mnist --accept '*.gz' http://yann.lecun.com/exdb/mnist/
pushd DB_mnist

echo "Descomprimiendo..."
gunzip *

echo "Los archivos estan en discos..."
popd
