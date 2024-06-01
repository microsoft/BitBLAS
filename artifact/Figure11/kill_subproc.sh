# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

pkill -f 'nvidia_measure_memory.sh'
pkill -f 'llama_70b.py'
pkill -f 'bloom-176b.py'
pkill -f 'bloom_176b.py'
pkill -f 'ort_runtime.py'
pkill -f 'trtexec'
pkill -f 'benchmark_llama.py'
pkill -f 'benchmark_bloom.py'
pkill -f 'ladder_with_fake_dense_dequantize.py'
pkill -f 'nvidia-smi'
