# !/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

MODEL_PATH=$(pwd)/../../models
TRT_EXEC_PATH=$(pwd)/../../baseline_framework/TensorRT-9.0.1.4/bin
export LD_LIBRARY_PATH=$TRT_EXEC_PATH/../lib:$LD_LIBRARY_PATH

echo "[TENSORRT] Using checkpoint path: $CHECKPOINT_PATH"
LADDER_LOG_PATH="$CHECKPOINT_PATH/onnxruntime/logs"


mkdir -p logs

# if engine file exists, skip the conversion
function convert() {
    if [ -f $2 ]; then
        echo "[TENSORRT] Engine file $2 exists, skip conversion"
    else
        echo "[TENSORRT] Converting $1 to $2"
        $TRT_EXEC_PATH/trtexec --onnx=$1 --saveEngine=$2 --fp16 --workspace=8192
    fi
}

convert $MODEL_PATH/bloom_176b/bloom-176b_layer1_seq1_bs1/model.onnx $MODEL_PATH/bloom_176b/bloom-176b_layer1_seq1_bs1/model.trt

convert $MODEL_PATH/bloom_176b/bloom-176b_layer1_seq1_bs32/model.onnx $MODEL_PATH/bloom_176b/bloom-176b_layer1_seq1_bs32/model.trt

convert $MODEL_PATH/bloom_176b/bloom-176b_layer1_seq4096_bs1/model.onnx $MODEL_PATH/bloom_176b/bloom-176b_layer1_seq4096_bs1/model.trt

convert $MODEL_PATH/llama-70b/llama2_70b_layer1_seq1_bs1/model.onnx $MODEL_PATH/llama-70b/llama2_70b_layer1_seq1_bs1/model.trt

convert $MODEL_PATH/llama-70b/llama2_70b_layer1_seq1_bs32/model.onnx $MODEL_PATH/llama-70b/llama2_70b_layer1_seq1_bs32/model.trt

convert $MODEL_PATH/llama-70b/llama2_70b_layer1_seq4096_bs1/model.onnx $MODEL_PATH/llama-70b/llama2_70b_layer1_seq4096_bs1/model.trt
