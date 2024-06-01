# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


export TVM_HOME=$(pwd)/../../baseline_framework/roller_tvm
export PYTHONPATH=$TVM_HOME/python
export PYTHONPATH=$(pwd)/../../baseline_framework/Roller/python:$PYTHONPATH

python3 ./fp16xfp16_gemm_nt.py | tee ./fp16xfp16_gemm_nt.log

python3 ./fp16xfp16_gemv_nt.py | tee ./fp16xfp16_gemv_nt.log
