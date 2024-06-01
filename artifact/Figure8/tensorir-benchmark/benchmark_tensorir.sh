# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

export TVM_TARGET="nvidia/nvidia-a100"
export TVM_HOME=$(pwd)/../../baseline_framework/tvm_v0.14.0
export PYTHONPATH=$TVM_HOME/python
pip install xgboost==1.5.0

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
    exit 0
fi

echo "[TensorIR] Running benchmark..."

# 1. tune the resnet-50 batch size 1
## if 'logs/resnet-50-b1-(1, 3, 224, 224)' already exists, skip tuning
if [ ! -f "./logs/resnet-50-b1-(1, 3, 224, 224)" ] || [ $force_tune -eq 1 ]; then
    START_TIME=$(date +%s)
    python3 tune_from_onnx.py --workload resnet-50-b1 --batch 1 --trials 20000 2>&1 | tee resnet-50-b1.log
    END_TIME=$(date +%s)
    echo "Compiler tuning time: $(($END_TIME - $START_TIME)) seconds" > "./logs/resnet-50-b1-(1, 3, 224, 224)/tune_time_cost.txt"
else
    echo "[TensorIR] Log directory for resnet-50 batch size 1 already exists. Skipping tuning..."
fi

# 2. tune the resnet-50 batch size 128
## if 'logs/resnet-50-b128-(128, 3, 224, 224)' already exists, skip tuning
if [ ! -f "./logs/resnet-50-b128-(128, 3, 224, 224)" ] || [ $force_tune -eq 1 ]; then
    START_TIME=$(date +%s)
    python3 tune_from_onnx.py --workload resnet-50-b128 --batch 128 --trials 20000 2>&1 | tee resnet-50-b128.log
    END_TIME=$(date +%s)
    echo "Compiler tuning time: $(($END_TIME - $START_TIME)) seconds" > "./logs/resnet-50-b128-(128, 3, 224, 224)/tune_time_cost.txt"
else
    echo "[TensorIR] Log directory for resnet-50 batch size 128 already exists. Skipping tuning..."
fi

# 3. tune the shufflenet-v2 batch size 1
## if 'logs/shufflenet-v2-b1-(1, 3, 224, 224)' already exists, skip tuning
if [ ! -f "./logs/shufflenet-b1-(1, 3, 224, 224)" ] || [ $force_tune -eq 1 ]; then
    START_TIME=$(date +%s)
    python3 tune_from_onnx.py --workload shufflenet-b1 --batch 1 --trials 20000 2>&1 | tee shufflenet-b1.log
    END_TIME=$(date +%s)
    echo "Compiler tuning time: $(($END_TIME - $START_TIME)) seconds" > "./logs/shufflenet-b1-(1, 3, 224, 224)/tune_time_cost.txt"
else
    echo "[TensorIR] Log directory for shufflenet-v2 batch size 1 already exists. Skipping tuning..."
fi

# 4. tune the shufflenet-v2 batch size 128
## if 'logs/shufflenet-v2-b128-(128, 3, 224, 224)' already exists, skip tuning
if [ ! -f "./logs/shufflenet-b128-(128, 3, 224, 224)" ] || [ $force_tune -eq 1 ]; then
    START_TIME=$(date +%s)
    python3 tune_from_onnx.py --workload shufflenet-b128 --batch 128 --trials 20000 2>&1 | tee shufflenet-b128.log
    END_TIME=$(date +%s)
    echo "Compiler tuning time: $(($END_TIME - $START_TIME)) seconds" > "./logs/shufflenet-b128-(128, 3, 224, 224)/tune_time_cost.txt"
else
    echo "[TensorIR] Log directory for shufflenet-v2 batch size 128 already exists. Skipping tuning..."
fi
