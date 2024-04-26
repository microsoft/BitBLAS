# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


# LADDER_HOME = pwd home + .. + .. + .. + ..

export LADDER_HOME=$(pwd)/../../..
export LADDER_TVM_HOME=$LADDER_HOME/3rdparty/tvm
export LADDER_CUTLASS_HOME=$LADDER_HOME/3rdparty/cutlass
export PYTHONPATH=$LADDER_HOME/python
export PYTHONPATH=$LADDER_TVM_HOME/python:$PYTHONPATH
export CPLUS_INCLUDE_PATH=$LADDER_CUTLASS_HOME/include

force_tune=0
if [[ "$1" == "--force_tune" ]]; then
    force_tune=1
fi

echo "[LADDER] Using checkpoint path: $CHECKPOINT_PATH"
LADDER_LOG_PATH="$CHECKPOINT_PATH/ladder/logs"

if [ -d "$LADDER_LOG_PATH" ] && [ $force_tune -eq 0 ]; then
    echo "[LADDER] Log directory logs already exists in checkpoints directory. Copying to current directory..."
    # if the log directory already exists in current directory, remove it
    if [ -d "./logs" ]; then
        rm -r "./logs"
    fi
    cp "$LADDER_LOG_PATH" "./logs" -r
fi

# if not force_tune, skip tuning


echo "[LADDER] Running benchmark..."

# Function to run benchmark if needed
run_benchmark() {
    local script=$2
    local options=$3
    if [ $force_tune -eq 0 ]; then
        if [ -f "./logs/$1" ]; then
            echo "[LADDER] Log file $1 already exists, skip tuning ..."
        else
            echo "[LADDER] Log file $1 do not exist, start tuning ..."
            python -u $script $options 2>&1 | tee ./logs/$1
        fi  
    else
        echo "[LADDER] Force tuning ..."
        python -u $script $options 2>&1 | tee ./logs/$1
    fi
}


# resnet-50-batch-1 default tuning
run_benchmark "resnet-50-b1.log" "dequantize_tune_from_onnx.py" "--prefix resnet-50-b1"

# resnet-50-batch-128 default tuning
run_benchmark "resnet-50-b128.log" "dequantize_tune_from_onnx.py" "--prefix resnet-50-b128"

# shufflenet-batch-1 default tuning
run_benchmark "shufflenet-b1.log" "dequantize_tune_from_onnx.py" "--prefix shufflenet-b1"

# shufflenet-batch-128 default tuning
run_benchmark "shufflenet-b128.log" "dequantize_tune_from_onnx.py" "--prefix shufflenet-b128"
