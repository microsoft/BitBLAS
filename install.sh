#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# install requirements
pip install -r requirements.txt

# install llvm
apt-get install llvm-10

# clone and build tvm
git submodule update --init --recursive

cd 3rdparty/tvm
mkdir build
cp cmake/config.cmake build
cd build
echo "set(USE_LLVM llvm-config-10)" >> config.cmake && echo "set(USE_CUDA ON)" >> config.cmake

cmake .. && make -j && cd ../../..

# remove the existing softlink if it exists
rm -f bitblas/tvm
# create softlink for bitblas
ln -s 3rdparty/tvm/python/tvm bitblas/tvm

echo "export TVM_HOME=$(pwd)/3rdparty/tvm" >> ~/.bashrc
echo "export PYTHONPATH=\$TVM_HOME/python:$(pwd):\$PYTHONPATH" >> ~/.bashrc

source ~/.bashrc
