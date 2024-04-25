# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# AMOS_HOME = pwd home + .. + .. + .. + 3rdparty + AMOS
export AMOS_HOME=$(pwd)/../../../baseline_framework/AMOS
export PYTHONPATH=$AMOS_HOME/python


force_tune=0
if [[ "$1" == "--force_tune" ]]; then
    force_tune=1
fi

echo "[AMOS] Using checkpoint path: $CHECKPOINT_PATH"

RESENET_B128_LOG_PATH="$CHECKPOINT_PATH/amos/resnet50_b128.log"

if [ -f "$RESENET_B128_LOG_PATH" ] && [ $force_tune -eq 0 ]; then
    echo "[AMOS] Log file resnet50_b128.log already exists in checkpoints directory. Copying to current directory..."
    cp "$RESENET_B128_LOG_PATH" "./resnet50_b128.log"
else
    echo "[AMOS] Running benchmark..."
    CUDA_VISIBLE_DEVICES=0 python3 ./mapping_resnet50_tensorcore.py--batch 1 --dtype float16 --trials 20 | tee resnet50_b128.log
fi

SHUFFLENET_B128_LOG_PATH="$CHECKPOINT_PATH/amos/shufflenet_v2_b128.log"

if [ -f "$SHUFFLENET_B128_LOG_PATH" ] && [ $force_tune -eq 0 ]; then
    echo "Log file shufflenet_v2_b128 already exists in checkpoints directory. Copying to current directory..."
    cp "$SHUFFLENET_B128_LOG_PATH" "./shufflenet_v2_b128.log"
else
    echo "Running benchmark..."
    python3 ./mapping_shufflenet_tensorcore.py --batch 1 --dtype float16 --trials 20 | tee shufflenet_v2_b128.log
fi


RESENET_B1_LOG_PATH="$CHECKPOINT_PATH/amos/resnet50_b1.log"

if [ -f "$RESENET_B1_LOG_PATH" ] && [ $force_tune -eq 0 ]; then
    echo "[AMOS] Log file resnet50_b1.log already exists in checkpoints directory. Copying to current directory..."
    cp "$RESENET_B1_LOG_PATH" "./resnet50_b1.log"
else
    echo "[AMOS] Running benchmark..."
    CUDA_VISIBLE_DEVICES=0 python3 ./mapping_resnet50_tensorcore.py--batch 1 --dtype float16 --trials 20 | tee resnet50_b1.log
fi

SHUFFLENET_B1_LOG_PATH="$CHECKPOINT_PATH/amos/shufflenet_v2_b1.log"

if [ -f "$SHUFFLENET_B1_LOG_PATH" ] && [ $force_tune -eq 0 ]; then
    echo "Log file shufflenet_v2_b1 already exists in checkpoints directory. Copying to current directory..."
    cp "$SHUFFLENET_B1_LOG_PATH" "./shufflenet_v2_b1.log"
else
    echo "Running benchmark..."
    python3 ./mapping_shufflenet_tensorcore.py --batch 1 --dtype float16 --trials 20 | tee shufflenet_v2_b1.log
fi

