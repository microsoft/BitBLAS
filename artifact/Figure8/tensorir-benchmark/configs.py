# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os

# model_path = pwd + .. + .. + models
model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "models"
)

onnx_files = {
    "resnet-50-b128": {
        "path": f"{model_path}/resnet-50-b128/model.onnx",
        "input_name": "input0",
        "input_dtype": "float16",
    },
    "resnet-50-b1": {
        "path": f"{model_path}/resnet-50-b1/model.onnx",
        "input_name": "input0",
        "input_dtype": "float16",
    },
    "shufflenet-b128": {
        "path": f"{model_path}/shufflenet-b128/model.onnx",
        "input_name": "input0",
        "input_dtype": "float16",
    },
    "shufflenet-b1": {
        "path": f"{model_path}/shufflenet-b1/model.onnx",
        "input_name": "input0",
        "input_dtype": "float16",
    },
    "Conformer-b128": {
        "path": f"{model_path}/Conformer-b128/model.onnx",
        "input_name": "input0",
        "input_dtype": "float16",
    },
    "Conformer-b1": {
        "path": f"{model_path}/Conformer-b1/model.onnx",
        "input_name": "input0",
        "input_dtype": "float16",
    },
    "vit-b128": {
        "path": f"{model_path}/vit-b128/model.onnx",
        "input_name": "input0",
        "input_dtype": "float16",
    },
    "vit-b1": {
        "path": f"{model_path}/vit-b1/model.onnx",
        "input_name": "input0",
        "input_dtype": "float16",
    },
    "llama2_70b_layer1_seq1_bs1": {
        "path": f"{model_path}/llama_70b/llama2_70b_layer1_seq1_bs1/model.onnx",
        "input_name": "input0",
        "input_dtype": "float16",
    },
    "llama2_70b_layer1_seq1_bs32": {
        "path": f"{model_path}/llama_70b/llama2_70b_layer1_seq1_bs32/model.onnx",
        "input_name": "input0",
        "input_dtype": "float16",
    },
    "llama2_70b_layer1_seq4096_bs1": {
        "path": f"{model_path}/llama_70b/llama2_70b_layer1_seq4096_bs1/model.onnx",
        "input_name": "input0",
        "input_dtype": "float16",
    },
    "bloom-176b_layer1_seq1_bs1": {
        "path": f"{model_path}/bloom_176b/bloom-176b_layer1_seq1_bs1/model.onnx",
        "input_name": "input0",
        "input_dtype": "float16",
    },
    "bloom-176b_layer1_seq1_bs32": {
        "path": f"{model_path}/bloom_176b/bloom-176b_layer1_seq1_bs32/model.onnx",
        "input_name": "input0",
        "input_dtype": "float16",
    },
    "bloom-176b_layer1_seq4096_bs1": {
        "path": f"{model_path}/bloom_176b/bloom-176b_layer1_seq4096_bs1/model.onnx",
        "input_name": "input0",
        "input_dtype": "float16",
    },
}


input_shape_dict = {
    "resnet-50-b128": (128, 3, 224, 224),
    "resnet-50-b1": (1, 3, 224, 224),
    "shufflenet-b128": (128, 3, 224, 224),
    "shufflenet-b1": (1, 3, 224, 224),
    "Conformer-b128": (128, 512, 512),
    "Conformer-b1": (1, 512, 512),
    "vit-b128": (128, 3, 224, 224),
    "vit-b1": (1, 3, 224, 224),
    "llama2_70b_layer1_seq1_bs1": (1, 1),
    "llama2_70b_layer1_seq1_bs32": (32, 1),
    "llama2_70b_layer1_seq4096_bs1": (1, 4096),
    "bloom-176b_layer1_seq1_bs1": (1, 1),
    "bloom-176b_layer1_seq1_bs32": (32, 1),
    "bloom-176b_layer1_seq4096_bs1": (1, 4096),
}
