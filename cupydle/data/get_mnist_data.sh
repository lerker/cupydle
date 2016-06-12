#!/bin/bash

if [ -d cupydle/data/mnistDB ]; then
    echo "data directory already present, have files in there? exiting"
    exit 1
fi

mkdir cupydle/data/mnistDB
wget --recursive --level=1 --cut-dirs=3 --no-host-directories \
  --directory-prefix=cupydle/data/mnistDB --accept '*.gz' http://yann.lecun.com/exdb/mnist/
pushd cupydle/data/mnistDB
gunzip *
popd
