#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# install requirements
pip install -r requirements.txt

# determine if root
USER_IS_ROOT=false
if [ "$EUID" -eq 0 ]; then
    USER_IS_ROOT=true
fi

if $USER_IS_ROOT; then
    # Fetch the GPG key for the LLVM repository and add it to the trusted keys
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc

    # Check if the repository is already present in the sources.list
    if ! grep -q "http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" /etc/apt/sources.list; then
        # Add the LLVM repository to sources.list
        echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" >> /etc/apt/sources.list
        echo "deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" >> /etc/apt/sources.list
    else
        # Print a message if the repository is already added
        echo "The repository is already added."
    fi

    # Update package lists and install llvm-16
    apt-get update
    apt-get install -y llvm-16
else
    # Fetch the GPG key for the LLVM repository and add it to the trusted keys using sudo
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc

    # Check if the repository is already present in the sources.list
    if ! grep -q "http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" /etc/apt/sources.list; then
        # Add the LLVM repository to sources.list using sudo
        echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" | sudo tee -a /etc/apt/sources.list
        echo "deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" | sudo tee -a /etc/apt/sources.list
    else
        # Print a message if the repository is already added
        echo "The repository is already added."
    fi

    # Update package lists and install llvm-16 using sudo
    sudo apt-get update
    sudo apt-get install -y llvm-16
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
