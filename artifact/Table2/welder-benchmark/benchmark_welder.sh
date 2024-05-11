# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

force_tune=0
if [[ "$1" == "--force_tune" ]]; then
    force_tune=1
fi
echo "[Welder] Using checkpoint path: $CHECKPOINT_PATH"

WELDER_LOG_PATH="$CHECKPOINT_PATH/welder/compiled_models"

if [ -d "$WELDER_LOG_PATH" ] && [ $force_tune -eq 0 ]; then
    echo "[TensorIR] Log directory logs already exists in checkpoints directory. Copying to current directory..."
    # if the log directory already exists in current directory, remove it
    if [ -d "./compiled_models" ]; then
        rm -r "./compiled_models"
    fi
    cp "$WELDER_LOG_PATH" "./compiled_models" -r
else
    rm -r "./compiled_models"
fi

python nnfusion_benchmark_cudnn.py
