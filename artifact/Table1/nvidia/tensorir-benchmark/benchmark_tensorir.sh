# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

export TVM_HOME=$(pwd)/../../../baseline_framework/tvm_v0.14.0
export PYTHONPATH=$TVM_HOME/python
pip install xgboost==1.7.1

force_tune=0
if [[ "$1" == "--force_tune" ]]; then
    force_tune=1
fi

echo "[TensorIR] Using checkpoint path: $CHECKPOINT_PATH"

FP16_LOG_PATH="$CHECKPOINT_PATH/tensorir/logs"

if [ -d "$FP16_LOG_PATH" ] && [ $force_tune -eq 0 ]; then
    echo "[TensorIR] Log directory logs already exists in checkpoints directory. Copying to current directory..."
    # if the log directory already exists in current directory, remove it
    if [ -d "./logs" ]; then
        rm -r "./logs"
    fi
    cp "$FP16_LOG_PATH" "./logs" -r
else
    echo "[TensorIR] Running benchmark..."
    python3 meta_nt.py --M 16384 --N 16384 --K 16384 --trails 1000
fi
