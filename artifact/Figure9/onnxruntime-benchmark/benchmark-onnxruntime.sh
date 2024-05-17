# !/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

MODEL_PATH=$(pwd)/../../models

echo "[ONNXRUNTIME] Using checkpoint path: $CHECKPOINT_PATH"
LADDER_LOG_PATH="$CHECKPOINT_PATH/onnxruntime/logs"


mkdir -p logs

python -u ort_runtime.py --file $MODEL_PATH/resnet-50-b1/model.onnx | tee logs/resnet-50-b1.log
python -u ort_runtime.py --file $MODEL_PATH/shufflenet-b1/model.onnx | tee logs/shufflenet-b1.log
python -u ort_runtime.py --file $MODEL_PATH/Conformer-b1/model.onnx | tee logs/Conformer-b1.log
python -u ort_runtime.py --file $MODEL_PATH/vit-b1/model.onnx | tee logs/vit-b1.log


python -u ort_runtime.py --file $MODEL_PATH/resnet-50-b128/model.onnx | tee logs/resnet-50-b128.log
python -u ort_runtime.py --file $MODEL_PATH/shufflenet-b128/model.onnx | tee logs/shufflenet-b128.log
python -u ort_runtime.py --file $MODEL_PATH/Conformer-b128/model.onnx | tee logs/Conformer-b128.log
python -u ort_runtime.py --file $MODEL_PATH/vit-b128/model.onnx | tee logs/vit-b128.log

# large languange models
python -u ort_runtime.py --file $MODEL_PATH/bloom_176b/bloom-176b_layer1_seq1_bs1/model.onnx | tee logs/bloom-176b-layer1-seq1-bs1.log
python -u ort_runtime.py --file $MODEL_PATH/bloom_176b/bloom-176b_layer1_seq1_bs32/model.onnx | tee logs/bloom-176b-layer1-seq1-bs32.log

# out of memory
# python -u ort_runtime.py --file $MODEL_PATH/bloom_176b/bloom-176b_layer1_seq4096_bs1/model.onnx | tee logs/bloom-176b-layer1-seq4096-bs1.log

python -u ort_runtime.py --file $MODEL_PATH/llama_70b/llama2_70b_layer1_seq1_bs1/model.onnx | tee logs/llama-70b-layer1-seq1-bs1.log
python -u ort_runtime.py --file $MODEL_PATH/llama_70b/llama2_70b_layer1_seq1_bs32/model.onnx | tee logs/llama-70b-layer1-seq1-bs32.log
python -u ort_runtime.py --file $MODEL_PATH/llama_70b/llama2_70b_layer1_seq4096_bs1/model.onnx | tee logs/llama-70b-layer1-seq4096-bs1.log
