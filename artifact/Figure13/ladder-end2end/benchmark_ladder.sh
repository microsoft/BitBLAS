# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

export LADDER_HOME=$(pwd)/../../..
export LADDER_TVM_HOME=$LADDER_HOME/3rdparty/tvm
export LADDER_CUTLASS_HOME=$LADDER_HOME/3rdparty/cutlass
export PYTHONPATH=$LADDER_HOME/python
export PYTHONPATH=$LADDER_TVM_HOME/python:$PYTHONPATH
export CPLUS_INCLUDE_PATH=$LADDER_CUTLASS_HOME/include

export CHECKPOINT_PATH=/root/Ladder/artifact/checkpoints/Figure13

echo "[LADDER] Using checkpoint path: $CHECKPOINT_PATH"

./0.benchmark_transform.sh
./1.benchmark_ptx.sh
./2.benchmark_holistic.sh
