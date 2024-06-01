# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
mkdir -p ./baseline_framework

apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev

pip3 install --user numpy decorator attrs synr

apt-get install -y llvm-10

git clone https://github.com/apache/tvm ./baseline_framework/faster_transformer_tvm --recursive

cd ./baseline_framework/faster_transformer_tvm
git checkout 1afbf20129647a35d108152fc6789bc1d029cda5
git submodule update --init --recursive
mkdir build
cp cmake/config.cmake build
cd build
echo "set(USE_CUDA ON)" >> config.cmake
echo "set(USE_LLVM ON)" >> config.cmake
echo "set(USE_CUBLAS ON)" >> config.cmake
echo "set(USE_CUTLASS ON)" >> config.cmake

cmake .. && make -j && cd ../../..
