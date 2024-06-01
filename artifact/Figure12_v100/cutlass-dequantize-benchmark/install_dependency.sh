#!/bin/bash

cd tvm
git submodule update --init --recursive
mkdir build
cd build
cp ../config.cmake .
echo "set(USE_CUDA ON)" >> config.cmake
echo "set(USE_LLVM ON)" >> config.cmake
echo "set(USE_CUBLAS ON)" >> config.cmake
echo "set(USE_CUTLASS ON)" >> config.cmake
cmake ..
make -j

cd ../..

export PYTHONPATH=$(pwd)/tvm/python

mkdir ./tmp
