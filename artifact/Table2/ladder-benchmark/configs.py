import os
# model_path = pwd + .. + .. + models
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")

models = {
    "resnet-50-b1": f"{model_path}/resnet-50-b1/model.onnx",
    "resnet-50-b128": f"{model_path}/resnet-50-b128/model.onnx",
    "shufflenet-b1": f"{model_path}/shufflenet-b1/model.onnx",
    "shufflenet-b128": f"{model_path}/shufflenet-b128/model.onnx",
}
