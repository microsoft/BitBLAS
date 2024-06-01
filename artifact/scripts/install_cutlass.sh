# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
mkdir -p ./baseline_framework

git clone --recursive https://github.com/NVIDIA/cutlass ./baseline_framework/cutlass

cd ./baseline_framework/cutlass

git checkout 629f4653c3ea3db3264030382956fabe715f3436

git submodule update --init --recursive

mkdir -p build
cd build
cmake .. -DCUTLASS_NVCC_ARCHS='80' -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop* -DCUTLASS_LIBRARY_OPERATIONS=gemm -DCUTLASS_LIBRARY_IGNORE_KERNELS=complex -DCMAKE_BUILD_TYPE=Debug

make -j

cd ../../..
