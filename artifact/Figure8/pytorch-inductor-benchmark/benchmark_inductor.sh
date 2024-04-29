# !/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

mkdir -p logs

rm model
ln -s ../../models/model ./model

python -u torch2onnx.py --bs 1 --run_torch_inductor --prefix resnet-50-b1 --fp16 resnet50 | tee logs/resnet-50-b1.log
python -u torch2onnx.py --bs 1 --run_torch_inductor --prefix shufflenet-b1 --fp16 shufflenet | tee logs/shufflenet-b1.log
python -u torch2onnx.py --bs 1 --run_torch_inductor --prefix Conformer-b1 --fp16 Conformer | tee logs/Conformer-b1.log
python -u torch2onnx.py --bs 1 --run_torch_inductor --prefix vit-b1 --fp16 vit | tee logs/vit-b1.log


python -u torch2onnx.py --bs 128 --run_torch_inductor --prefix resnet-50-b128 --fp16 resnet50 | tee logs/resnet-50-b128.log
python -u torch2onnx.py --bs 128 --run_torch_inductor --prefix shufflenet-b128 --fp16 shufflenet | tee logs/shufflenet-b128.log
python -u torch2onnx.py --bs 128 --run_torch_inductor --prefix Conformer-b128 --fp16 Conformer | tee logs/Conformer-b128.log
python -u torch2onnx.py --bs 128 --run_torch_inductor --prefix vit-b128 --fp16 vit | tee logs/vit.log | tee logs/vit-b128.log

# # large languange models

python -u llama_70b.py --batch_size 1 --seq_length 1 | tee logs/llama-70b-layer1-seq1-bs1.log

python -u llama_70b.py --batch_size 32 --seq_length 1 | tee logs/llama-70b-layer1-seq1-bs32.log

python -u llama_70b.py --batch_size 1 --seq_length 4096 | tee logs/llama-70b-layer1-seq4096-bs1.log


python -u bloom_176b.py --batch_size 1 --seq_length 1 | tee logs/bloom-176b-layer1-seq1-bs1.log

python -u bloom_176b.py --batch_size 32 --seq_length 1 | tee logs/bloom-176b-layer1-seq1-bs32.log

python -u bloom_176b.py --batch_size 1 --seq_length 4096 | tee logs/bloom-176b-layer1-seq4096-bs1.log

