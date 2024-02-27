#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# install torch
pip install torch==2.1.0

# install llvm
apt-get install llvm-dev

# clone and build tvm
git clone https://github.com/LeiWang1999/tvm --recursive -b dev/fast_dlight --depth 1 3rdparty/tvm

cd 3rdparty/tvm
mkdir build
cp cmake/config.cmake build
cd build
echo "set(USE_LLVM ON)" >> config.cmake && echo "set(USE_CUDA ON)" >> config.cmake

cmake .. && make -j && cd ../../..

echo "export TVM_HOME=$(pwd)/3rdparty/tvm" >> ~/.bashrc
echo "export PYTHONPATH=\$TVM_HOME/python:$(pwd)/python" >> ~/.bashrc
