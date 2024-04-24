# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

apt install -y wget

mkdir -p ./baseline_framework

# Download and install the NVIDIA repository key
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/9.0.1/tars/TensorRT-9.0.1.4.Linux.x86_64-gnu.cuda-12.2.tar.gz -O ./baseline_framework/TensorRT-9.0.1.4.Linux.x86_64-gnu.cuda-12.2.tar.gz 

# Tar the file
cd ./baseline_framework

tar -xvzf TensorRT-9.0.1.4.Linux.x86_64-gnu.cuda-12.2.tar.gz 

cd ..
