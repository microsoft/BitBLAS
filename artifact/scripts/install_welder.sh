# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# !/bin/bash


## install tvm environment
apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev

pip3 install --user numpy decorator attrs

apt-get install -y llvm-10

git clone --branch welder https://github.com/nox-410/tvm ./baseline_framework/welder_tvm --recursive

cd ./baseline_framework/welder_tvm
mkdir build
cp cmake/config.cmake build
cd build
echo "set(USE_LLVM llvm-config-10)" >> config.cmake && echo "set(USE_CUDA ON)" >> config.cmake

cmake .. && make -j && cd ../../..

# MemFusion Artifact
git clone https://github.com/LeiWang1999/memfusion_artifact ./baseline_framework/WELDER

# MemFusion NNFusion

git clone --branch lowbit https://github.com/LeiWang1999/nnfusion ./baseline_framework/welder_nnfusion

cd ./baseline_framework/welder_nnfusion
./maint/script/install_dependency.sh 
mkdir build && cd build
cmake .. && make -j && cd ../../..

# MemFusion Cutlass

git clone --branch ladder https://github.com/LeiWang1999/cutlass ./baseline_framework/welder_cutlass --recursive
