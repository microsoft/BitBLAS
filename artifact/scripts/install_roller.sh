# !/bin/bash

## install tvm environment
apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev

pip3 install --user numpy decorator attrs

apt-get install -y llvm-10

git clone --branch roller https://github.com/LeiWang1999/tvm ./baseline_framework/roller_tvm --recursive

cd ./baseline_framework/roller_tvm
mkdir build
cp cmake/config.cmake build
cd build
echo "set(USE_LLVM llvm-config-10)" >> config.cmake && echo "set(USE_CUDA ON)" >> config.cmake

cmake .. && make -j && cd ../../..

## install roller

git clone https://github.com/LeiWang1999/Roller --recursive ./baseline_framework/Roller

