# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# export VLLM_HOME=$(pwd)/../../baseline_framework/vLLM
# export PYTHONPATH=$VLLM_HOME
# echo "[vLLM] Using checkpoint path: $VLLM_HOME"
# # This script benchmarks the VLLM model on a single GPU.
# echo "[vLLM] Using checkpoint path: $CHECKPOINT_PATH"
# # LADDER_LOG_PATH="$CHECKPOINT_PATH/vllm/logs"
# # export LD_LIBRARY_PATH=/home/t-leiwang/Ladder/artifact/baseline_framework/vLLM/vllm:$LD_LIBRARY_PATH
# # mkdir -p logs

python3.10 benchmark_kernel.py | tee logs/vllm_benchmark_kernel.log
