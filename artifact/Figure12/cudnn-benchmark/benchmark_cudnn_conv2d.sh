# !/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

python -u conv2d.py --target cuda --enable_cudnn --number 5 --repeats 5 --begin 0 --dtype FP16 2>&1 | tee logs/cudnn_fp16.log
