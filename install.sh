#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

echo "Starting installation script..."

# Step 1: Install Python requirements
echo "Installing Python requirements from requirements.txt..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install Python requirements."
    exit 1
else
    echo "Python requirements installed successfully."
fi

# Step 2: Define LLVM version and architecture
LLVM_VERSION="10.0.1"
IS_AARCH64=false
EXTRACT_PATH="3rdparty"
echo "LLVM version set to ${LLVM_VERSION}."
echo "Is AARCH64 architecture: $IS_AARCH64"

# Step 3: Determine the correct Ubuntu version based on LLVM version
UBUNTU_VERSION="16.04"
if [[ "$LLVM_VERSION" > "17.0.0" ]]; then
    UBUNTU_VERSION="22.04"
elif [[ "$LLVM_VERSION" > "16.0.0" ]]; then
    UBUNTU_VERSION="20.04"
elif [[ "$LLVM_VERSION" > "13.0.0" ]]; then
    UBUNTU_VERSION="18.04"
fi
echo "Ubuntu version for LLVM set to ${UBUNTU_VERSION}."

# Step 4: Set download URL and file name for LLVM
BASE_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}"
if $IS_AARCH64; then
    FILE_NAME="clang+llvm-${LLVM_VERSION}-aarch64-linux-gnu.tar.xz"
else
    FILE_NAME="clang+llvm-${LLVM_VERSION}-x86_64-linux-gnu-ubuntu-${UBUNTU_VERSION}.tar.xz"
fi
DOWNLOAD_URL="${BASE_URL}/${FILE_NAME}"
echo "Download URL for LLVM: ${DOWNLOAD_URL}"

# Step 5: Create extraction directory
echo "Creating extraction directory at ${EXTRACT_PATH}..."
mkdir -p "$EXTRACT_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Failed to create extraction directory."
    exit 1
else
    echo "Extraction directory created successfully."
fi

# Step 6: Download LLVM
echo "Downloading $FILE_NAME from $DOWNLOAD_URL..."
curl -L -o "${EXTRACT_PATH}/${FILE_NAME}" "$DOWNLOAD_URL"
if [ $? -ne 0 ]; then
    echo "Error: Download failed!"
    exit 1
else
    echo "Download completed successfully."
fi

# Step 7: Extract LLVM
echo "Extracting $FILE_NAME to $EXTRACT_PATH..."
tar -xJf "${EXTRACT_PATH}/${FILE_NAME}" -C "$EXTRACT_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Extraction failed!"
    exit 1
else
    echo "Extraction completed successfully."
fi

# Step 8: Determine LLVM config path
LLVM_CONFIG_PATH="$(realpath ${EXTRACT_PATH}/$(basename ${FILE_NAME} .tar.xz)/bin/llvm-config)"
echo "LLVM config path determined as: $LLVM_CONFIG_PATH"

# Step 9: Clone and build TVM
echo "Cloning TVM repository and initializing submodules..."
git submodule update --init --recursive
if [ $? -ne 0 ]; then
    echo "Error: Failed to initialize submodules."
    exit 1
else
    echo "Submodules initialized successfully."
fi

# Step 10: Build TVM
echo "Starting TVM build process..."
cd 3rdparty/tvm
if [ -d build ]; then
    echo "Existing build directory found. Removing it..."
    rm -rf build
fi
echo "Creating new build directory for TVM..."
mkdir build
cp cmake/config.cmake build
cd build

echo "Configuring TVM build with LLVM and CUDA paths..."
echo "set(USE_LLVM $LLVM_CONFIG_PATH)" >> config.cmake
echo "set(USE_CUDA /usr/local/cuda)" >> config.cmake

echo "Running CMake for TVM..."
cmake ..
if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed."
    exit 1
fi

echo "Building TVM with make..."
make -j
if [ $? -ne 0 ]; then
    echo "Error: TVM build failed."
    exit 1
else
    echo "TVM build completed successfully."
fi

TVM_PREBUILD_PATH=$(realpath .)

cd ../..

echo "Building TileLang with CMake..."
cd tilelang
mkdir build
cd build

cmake .. -DTVM_PREBUILD_PATH=$TVM_PREBUILD_PATH
if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed."
    exit 1
fi

make -j
if [ $? -ne 0 ]; then
    echo "Error: TileLang build failed."
    exit 1
else
    echo "TileLang build completed successfully."
fi

echo "TileLang build completed successfully."

cd ../../..

# Set environment variables
TVM_HOME_ENV="export TVM_HOME=$(pwd)/3rdparty/tvm"
TVM_EXPORT_ENV="export TVM_IMPORT_PYTHON_PATH=/root/BitBLAS/3rdparty/tvm/python"
TILELANG_HOME_ENV="export TILELANG_HOME=$(pwd)/3rdparty/tilelang"
BITBLAS_PYPATH_ENV="export PYTHONPATH=\$TVM_HOME/python:\$TILELANG_HOME:$(pwd):\$PYTHONPATH"
CUDA_DEVICE_ORDER_ENV="export CUDA_DEVICE_ORDER=PCI_BUS_ID"

# Inject break line if the last line of the file is not empty
if [ -s ~/.bashrc ]; then
    if [ "$(tail -c 1 ~/.bashrc)" != "" ]; then
        echo "" >> ~/.bashrc
    fi
fi

# Check and add the first line if not already present
if ! grep -qxF "$TVM_HOME_ENV" ~/.bashrc; then
    echo "$TVM_HOME_ENV" >> ~/.bashrc
    echo "Added TVM_HOME to ~/.bashrc"
else
    echo "TVM_HOME is already set in ~/.bashrc"
fi

# Check and add the second line if not already present
if ! grep -qxF "$BITBLAS_PYPATH_ENV" ~/.bashrc; then
    echo "$BITBLAS_PYPATH_ENV" >> ~/.bashrc
    echo "Added PYTHONPATH to ~/.bashrc"
else
    echo "PYTHONPATH is already set in ~/.bashrc"
fi

# Check and add the third line if not already present
if ! grep -qxF "$CUDA_DEVICE_ORDER_ENV" ~/.bashrc; then
    echo "$CUDA_DEVICE_ORDER_ENV" >> ~/.bashrc
    echo "Added CUDA_DEVICE_ORDER to ~/.bashrc"
else
    echo "CUDA_DEVICE_ORDER is already set in ~/.bashrc"
fi

# Reload ~/.bashrc to apply the changes
source ~/.bashrc
