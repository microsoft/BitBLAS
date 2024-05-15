# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
# model_path = pwd + .. + .. + models
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")

models = {
    "llama2_70b_layer1_seq4096_bs1": f"{model_path}/llama_70b/llama2_70b_layer1_seq4096_bs1/model.onnx",
    "bloom-176b_layer1_seq4096_bs1": f"{model_path}/bloom_176b/bloom-176b_layer1_seq4096_bs1/model.onnx",
}
