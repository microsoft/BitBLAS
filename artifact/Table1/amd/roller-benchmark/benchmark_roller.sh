# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# !/bin/bash

export TVM_HOME=$(pwd)/../../../baseline_framework/roller_tvm
export PYTHONPATH=$TVM_HOME/python
export PYTHONPATH=$(pwd)/../../../baseline_framework/Roller/python:$PYTHONPATH

python3 ./fp16xfp16_gemm_nt.py | tee ./fp16xfp16_gemm_nt.log
