# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
# model_path = pwd + .. + .. + models
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")

models = {
    "vit-b1" : f"{model_path}/vit-b1/model.onnx",
    "vit-b128" : f"{model_path}/vit-b128/model.onnx",
    "llama2_70b_layer1_seq1_bs1": f"{model_path}/llama_70b/llama2_70b_layer1_seq1_bs1/model.onnx",
    "llama2_70b_layer1_seq1_bs32": f"{model_path}/llama_70b/llama2_70b_layer1_seq1_bs32/model.onnx",
    "bloom-176b_layer1_seq1_bs1": f"{model_path}/bloom_176b/bloom-176b_layer1_seq1_bs1/model.onnx",
    "bloom-176b_layer1_seq1_bs32": f"{model_path}/bloom_176b/bloom-176b_layer1_seq1_bs32/model.onnx",
    "llama2_70b_layer1_seq4096_bs1": f"{model_path}/llama_70b/llama2_70b_layer1_seq4096_bs1/model.onnx",
}
