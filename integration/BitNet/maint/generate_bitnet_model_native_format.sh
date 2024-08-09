#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# require git lfs
if ! command -v git-lfs &> /dev/null; then
    echo "Please install git-lfs first by running 'sudo apt install git-lfs'"
    exit 1
fi

mkdir -p models

cd models

# download the model
git clone https://huggingface.co/1bitLLM/bitnet_b1_58-3B bitnet_b1_58-3B --depth 1

# copy quantized config into the model directory
cp ../maint/quantize_config.json bitnet_b1_58-3B

# get the realpath of the model directory
MODEL_DIR=$(realpath bitnet_b1_58-3B)

cd ..

echo "Model has been converted and save to $MODEL_DIR"
