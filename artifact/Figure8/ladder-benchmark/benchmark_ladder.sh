# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

export LADDER_HOME=$(pwd)/../../..
export LADDER_TVM_HOME=$LADDER_HOME/3rdparty/tvm
export LADDER_CUTLASS_HOME=$LADDER_HOME/3rdparty/cutlass
export PYTHONPATH=$LADDER_HOME/python
export PYTHONPATH=$LADDER_TVM_HOME/python:$PYTHONPATH
export CPLUS_INCLUDE_PATH=$LADDER_CUTLASS_HOME/include

./0.benchmark_conv.sh $1 $2
./1.benchmark_conv_quantize_b128.sh $1 $2
./2.benchmark_conv_quantize_b1.sh $1 $2
./3.benchmark_llama.sh $1 $2
./4.benchmark_bloom.sh $1 $2
