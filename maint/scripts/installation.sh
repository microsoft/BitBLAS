#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# install torch
pip install torch==2.1.0

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

echo "export TVM_HOME=$(pwd)/3rdparty/tvm" >> ~/.bashrc
echo "export PYTHONPATH=\$TVM_HOME/python:$(pwd)/python:\$PYTHONPATH" >> ~/.bashrc

source ~/.bashrc
