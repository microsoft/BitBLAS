#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# clone and build tvm
git submodule update --init --recursive

cd 3rdparty/tvm
mkdir build
cp cmake/config.cmake build
cd build
echo "set(USE_LLVM llvm-config)" >> config.cmake && echo "set(USE_CUDA ON)" >> config.cmake

cmake .. && make -j && cd ../../..
