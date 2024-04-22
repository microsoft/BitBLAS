# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
# model_path = pwd + .. + .. + models
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")

onnx_files = {
    "resnet-50": {
        "path": f"{model_path}/resnet-50-b128/model.onnx",
        "input_name": "input0",
        "input_dtype": "float16",
    },
    "resnet-50-b1": {
        "path": f"{model_path}/resnet-50-b1/model.onnx",
        "input_name": "input0",
        "input_dtype": "float16",
    },
    "shufflenet": {
        "path": f"{model_path}/shufflenet-b128/model.onnx",
        "input_name": "input0",
        "input_dtype": "float16",
    },
    "shufflenet-b1": {
        "path": f"{model_path}/shufflenet-b1/model.onnx",
        "input_name": "input0",
        "input_dtype": "float16",
    },
}
