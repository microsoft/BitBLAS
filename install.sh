#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

nproc=$(nproc)
if [ -z "$nproc" ]; then
    nproc=1
fi
# max 16 jobs
if [ $nproc -gt 16 ]; then
    nproc=16
fi

# install requirements
pip install -r requirements.txt

# install llvm
LLVM_VERSION="10.0.1"
IS_AARCH64=false
EXTRACT_PATH="3rdparty"

UBUNTU_VERSION="16.04"
if [[ "$LLVM_VERSION" > "16.0.0" ]]; then
    UBUNTU_VERSION="20.04"
elif [[ "$LLVM_VERSION" > "13.0.0" ]]; then
    UBUNTU_VERSION="18.04"
fi

BASE_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}"
if $IS_AARCH64; then
    FILE_NAME="clang+llvm-${LLVM_VERSION}-aarch64-linux-gnu.tar.xz"
else
    FILE_NAME="clang+llvm-${LLVM_VERSION}-x86_64-linux-gnu-ubuntu-${UBUNTU_VERSION}.tar.xz"
fi
DOWNLOAD_URL="${BASE_URL}/${FILE_NAME}"

mkdir -p "$EXTRACT_PATH"

echo "Downloading $FILE_NAME from $DOWNLOAD_URL"
curl -L -o "${EXTRACT_PATH}/${FILE_NAME}" "$DOWNLOAD_URL"

if [ $? -ne 0 ]; then
    echo "Download failed!"
    exit 1
fi

echo "Extracting $FILE_NAME to $EXTRACT_PATH"
tar -xJf "${EXTRACT_PATH}/${FILE_NAME}" -C "$EXTRACT_PATH"

if [ $? -ne 0 ]; then
    echo "Extraction failed!"
    exit 1
fi

echo "Download and extraction completed successfully."

LLVM_CONFIG_PATH="$(realpath ${EXTRACT_PATH}/$(basename ${FILE_NAME} .tar.xz)/bin/llvm-config)"
echo "LLVM config path: $LLVM_CONFIG_PATH"

# update and build tvm
git submodule update --init --recursive

cd 3rdparty/tvm
if [ -d build ]; then
    rm -rf build
fi
mkdir build
cp cmake/config.cmake build
cd build

# get the absolute path of the TVM prebuild path
ABS_TVM_PREBUILD_PATH=$(realpath .)

echo "set(USE_LLVM $LLVM_CONFIG_PATH)" >> config.cmake && echo "set(USE_CUDA /usr/local/cuda)" >> config.cmake

cmake .. && make -j $nproc && cd ../../..

# update and build tile-lang
cd 3rdparty/tile-lang
if [ -d build ]; then
    rm -rf build
fi

mkdir build

cd build

cmake .. -DTVM_PREBUILD_PATH=$ABS_TVM_PREBUILD_PATH && make -j $nproc && cd ../../..

echo "export TVM_HOME=$(pwd)/3rdparty/tvm" >> ~/.bashrc
# For 3rdparty/tile-lang import path
echo "export TVM_IMPORT_PYTHON_PATH=\$TVM_HOME/python" >> ~/.bashrc
echo "export PYTHONPATH=\$TVM_HOME/python:$(pwd):\$PYTHONPATH" >> ~/.bashrc

source ~/.bashrc
