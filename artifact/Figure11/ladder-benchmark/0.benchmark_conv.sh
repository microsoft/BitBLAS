# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

export LADDER_HOME=$(pwd)/../../..
export LADDER_TVM_HOME=$LADDER_HOME/3rdparty/tvm
export LADDER_CUTLASS_HOME=$LADDER_HOME/3rdparty/cutlass
export PYTHONPATH=$LADDER_HOME/python
export PYTHONPATH=$LADDER_TVM_HOME/python:$PYTHONPATH
export CPLUS_INCLUDE_PATH=$LADDER_CUTLASS_HOME/include

echo "[LADDER] Using checkpoint path: $CHECKPOINT_PATH"
LADDER_LOG_PATH="$CHECKPOINT_PATH/ladder/logs"

MODEL_PATH=$(pwd)/../../models

python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/resnet-50-b128 2>&1 | tee logs/resnet-50-b128.log
python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/shufflenet-b128 2>&1 | tee logs/shufflenet-b128.log
python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/Conformer-b128 2>&1 | tee logs/Conformer-b128.log 
python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/vi-b128 2>&1 | tee logs/vit-b128.log 

python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/resnet-50-b1 2>&1 | tee logs/resnet-50-b1.log 
python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/shufflenet-b1 2>&1 | tee logs/shufflenet-b1.log 
python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/Conformer-b1 2>&1 | tee logs/Conformer-b1.log
python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/vit-b1 2>&1 | tee logs/vit-b1.log
