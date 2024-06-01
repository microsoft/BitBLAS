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
LADDER_CHECKPOINT_PATH="$CHECKPOINT_PATH/ladder/checkpoints"

force_tune=0
if [[ "$1" == "--force_tune" ]]; then
    force_tune=1
fi

MODEL_PATH=$(pwd)/../../models

if [ $force_tune -eq 1 ]; then
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/resnet-50-b128 2>&1 | tee logs/resnet-50-b128.log
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/shufflenet-b128 2>&1 | tee logs/shufflenet-b128.log
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/Conformer-b128 2>&1 | tee logs/Conformer-b128.log 
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/vi-b128 2>&1 | tee logs/vit-b128.log 

    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/resnet-50-b1 2>&1 | tee logs/resnet-50-b1.log 
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/shufflenet-b1 2>&1 | tee logs/shufflenet-b1.log 
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/Conformer-b1 2>&1 | tee logs/Conformer-b1.log
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/vit-b1 2>&1 | tee logs/vit-b1.log

else
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/resnet-50 2>&1 | tee logs/resnet-50-b128.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/shufflenet 2>&1 | tee logs/shufflenet-b128.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/Conformer 2>&1 | tee logs/Conformer-b128.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/vit 2>&1 | tee logs/vit-b128.log

    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/resnet-50-b1 2>&1 | tee logs/resnet-50-b1.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/shufflenet-b1 2>&1 | tee logs/shufflenet-b1.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/Conformer-b1 2>&1 | tee logs/Conformer-b1.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/vit-b1 2>&1 | tee logs/vit-b1.log
fi