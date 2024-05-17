# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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

# 1. tune the resnet-50 batch size 1

python3 meta_nt.py --M 16384 --N 16384 --K 16384 --trails 1000
python3 meta_nt.py --M 8192 --N 43008 --K 14336 --trails 1000
python3 meta_nt.py --M 8192 --N 14336 --K 14336 --trails 1000
python3 meta_nt.py --M 8192 --N 57344 --K 14336 --trails 1000
python3 meta_nt.py --M 8192 --N 14336 --K 57344 --trails 1000
python3 meta_nt.py --M 8192 --N 9216 --K 9216 --trails 1000
python3 meta_nt.py --M 8192 --N 36864 --K 9216 --trails 1000
python3 meta_nt.py --M 8192 --N 9216 --K 36864 --trails 1000
python3 meta_nt.py --M 8192 --N 22016 --K 8192 --trails 1000
python3 meta_nt.py --M 8192 --N 8192 --K 22016 --trails 1000
python3 meta_nt.py --M 8192 --N 8192 --K 8192 --trails 1000
python3 meta_nt.py --M 8192 --N 28672 --K 8192 --trails 1000
python3 meta_nt.py --M 8192 --N 8192 --K 22016 --trails 1000

python3 meta_nt_int8.py --M 16384 --N 16384 --K 16384 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 43008 --K 14336 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 14336 --K 14336 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 57344 --K 14336 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 14336 --K 57344 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 9216 --K 9216 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 36864 --K 9216 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 9216 --K 36864 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 22016 --K 8192 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 8192 --K 22016 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 8192 --K 8192 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 28672 --K 8192 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 8192 --K 22016 --trails 1000

fi
