# !/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

mkdir -p logs


python -u llama_70b.py --batch_size 1 --seq_length 1 | tee logs/llama-70b-layer1-seq1-bs1.log

python -u llama_70b.py --batch_size 32 --seq_length 1 | tee logs/llama-70b-layer1-seq1-bs32.log

python -u llama_70b.py --batch_size 1 --seq_length 4096 | tee logs/llama-70b-layer1-seq4096-bs1.log


python -u bloom_176b.py --batch_size 1 --seq_length 1 | tee logs/bloom-176b-layer1-seq1-bs1.log

python -u bloom_176b.py --batch_size 32 --seq_length 1 | tee logs/bloom-176b-layer1-seq1-bs32.log

python -u bloom_176b.py --batch_size 1 --seq_length 4096 | tee logs/bloom-176b-layer1-seq4096-bs1.log

