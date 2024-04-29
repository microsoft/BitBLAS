# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

mkdir -p ./baseline_framework

git clone https://github.com/LeiWang1999/vLLM.git --recursive ./baseline_framework/vLLM

cd ./baseline_framework/vLLM

git checkout d1690fc997bbe074f007ffe2a6eb83591f7f8c99

pip install -e .

cd ../..

pip install transformers==4.30.0
