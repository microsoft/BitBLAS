# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# !/bin/bash
# AMOS_HOME = pwd home + .. + .. + .. + 3rdparty + AMOS
export AMOS_HOME=$(pwd)/../../../baseline_framework/AMOS
export PYTHONPATH=$AMOS_HOME/python

force_tune=0
if [[ "$1" == "--force_tune" ]]; then
    force_tune=1
fi

echo "[AMOS] Using checkpoint path: $CHECKPOINT_PATH"

FP16_LOG_PATH="$CHECKPOINT_PATH/amos/gemm_nt_16384_float16.log"

if [ -f "$FP16_LOG_PATH" ] && [ $force_tune -eq 0 ]; then
    echo "[AMOS] Log file gemm_nt_16384_float16.log already exists in checkpoints directory. Copying to current directory..."
    cp "$FP16_LOG_PATH" "./gemm_nt_16384_float16.log"
else
    echo "[AMOS] Running benchmark..."
    python3 ./gemm_nt_16384.py --trials 1000 --simple_mode 1 | tee "./gemm_nt_16384_float16.log"
fi

INT8_LOG_PATH="$CHECKPOINT_PATH/amos/gemm_nt_16384_int8.log"

if [ -f "$INT8_LOG_PATH" ] && [ $force_tune -eq 0 ]; then
    echo "Log file gemm_nt_16384_int8 already exists in checkpoints directory. Copying to current directory..."
    cp "$INT8_LOG_PATH" "./gemm_nt_16384_int8.log"
else
    echo "Running benchmark..."
    python3 ./gemm_nt_16384.py --trials 1000 --simple_mode 1 --in_dtype int8 --out_dtype int32 | tee "./gemm_nt_16384_int8.log"
fi
