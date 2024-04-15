#!/bin/bash
rm -rf build
mkdir -p build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
make -j
./cublas_benchmark | tee cublas_benchmark.log
