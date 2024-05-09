#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

export TVM_HOME=$(pwd)/../../baseline_framework/faster_transformer_tvm
export PYTHONPATH=$TVM_HOME/python

mkdir -p tmp

python -u ./cutlass_fpa_intb.py 2>&1 | tee logs/cutlass_fpa_intb.log
