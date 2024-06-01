#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
rm -rf build
mkdir -p build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=89
make -j
./cublas_benchmark | tee cublas_benchmark.log
cd ..
