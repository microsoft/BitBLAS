# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

export CUDA_VISIBLE_DEVICES=3
export TVM_HOME=$(pwd)/../../baseline_framework/tvm_v0.14.0
export PYTHONPATH=$TVM_HOME/python

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

python3 meta_nt.py --M 1 --N 14336 --K 57344 --trails 1000 2>&1 | tee meta_nt_m1_n14336_k57344.log
python3 meta_nt.py --M 32 --N 14336 --K 57344 --trails 1000 2>&1 | tee meta_nt_m32_n14336_k57344.log
python3 meta_nt.py --M 4096 --N 14336 --K 57344 --trails 1000 2>&1 | tee meta_nt_m4096_n14336_k57344.log
python3 meta_nt.py --M 1 --N 8192 --K 28672 --trails 1000 2>&1 | tee meta_nt_m1_n8192_k28672.log
python3 meta_nt.py --M 32 --N 8192 --K 28672 --trails 1000 2>&1 | tee meta_nt_m32_n8192_k28672.log
python3 meta_nt.py --M 4096 --N 8192 --K 28672 --trails 1000 2>&1 | tee meta_nt_m4096_n8192_k28672.log

fi
