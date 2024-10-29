#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# install requirements
pip install -r requirements.txt

# determine if root
USER_IS_ROOT=false
if [ "$EUID" -e 0 ]; then
    USER_IS_ROOT=true
fi

if $USER_IS_ROOT; then
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
    echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" >> /etc/apt/sources.list
    echo "deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" >> /etc/apt/sources.list
    apt-get install llvm-16
else 
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
    echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" | sudo tee /etc/apt/sources.list
    echo "deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" | sudo tee /etc/apt/sources.list
    sudo apt-get install llvm-16
fi

# clone and build tvm
git submodule update --init --recursive

cd 3rdparty/tvm
if [ -d build ]; then
    rm -rf build
fi
mkdir build
cp cmake/config.cmake build
cd build
echo "set(USE_LLVM llvm-config-16)" >> config.cmake && echo "set(USE_ROCM /opt/rocm)" >> config.cmake

cmake .. && make -j && cd ../../..

echo "export TVM_HOME=$(pwd)/3rdparty/tvm" >> ~/.bashrc
echo "export PYTHONPATH=\$TVM_HOME/python:$(pwd):\$PYTHONPATH" >> ~/.bashrc
echo "export CUDA_DEVICE_ORDER=PCI_BUS_ID" >> ~/.bashrc
source ~/.bashrc
