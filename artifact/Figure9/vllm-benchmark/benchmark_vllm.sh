# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

export VLLM_HOME=$(pwd)/../../baseline_framework/vLLM
export PYTHONPATH=$VLLM_HOME
echo "[vLLM] Using checkpoint path: $VLLM_HOME"
# This script benchmarks the VLLM model on a single GPU.
echo "[vLLM] Using checkpoint path: $CHECKPOINT_PATH"
LADDER_LOG_PATH="$CHECKPOINT_PATH/vllm/logs"

mkdir -p logs

# out of memory
# python benchmark_bloom.py --batch_size 1 --seq_length 1 | tee logs/bloom_layer1_batch1_seq1.log

# out of memory
# python benchmark_bloom.py --batch_size 32 --seq_length 1 | tee logs/bloom_layer1_batch32_seq1.log

python benchmark_llama.py --batch_size 1 --seq_length 1 | tee logs/llama_layer1_batch1_seq1.log

python benchmark_llama.py --batch_size 32 --seq_length 1 | tee logs/llama_layer1_batch32_seq1.log

python benchmark_llama.py --batch_size 1 --seq_length 4096 | tee logs/llama_layer1_batch1_seq4096.log


# out of memory
# python benchmark_bloom.py --batch_size 1 --seq_length 1 --int4 | tee logs/bloom_layer1_batch1_seq1_int4.log

# out of memory
# python benchmark_bloom.py --batch_size 32 --seq_length 1 --int4 | tee logs/bloom_layer1_batch32_seq1_int4.log

# out of memory
# python benchmark_bloom.py --batch_size 1 --seq_length 4096 --int4 | tee logs/bloom_layer1_batch1_seq4096_int4.log

# awq doesn't support int4 on sm 70
# python benchmark_llama.py --batch_size 1 --seq_length 1 --int4 | tee logs/llama_layer1_batch1_seq1_int4.log

# python benchmark_llama.py --batch_size 32 --seq_length 1 --int4 | tee logs/llama_layer1_batch32_seq1_int4.log

# python benchmark_llama.py --batch_size 1 --seq_length 4096 --int4 | tee logs/llama_layer1_batch1_seq4096_int4.log
