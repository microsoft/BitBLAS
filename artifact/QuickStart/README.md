# Quick Start Guide for Model Inference and ONNX Export

This README provides a comprehensive guide to perform inference and ONNX export using different environments and precision settings. Follow the steps below to get started with inference using PyTorch in half precision, compilation and inference using Ladder with ONNX models, and inference with ONNX models containing quantized information.

## Prerequisites

Before beginning, ensure that you have set up the necessary environment variables and installed all required dependencies for the Ladder and AutoGPTQ frameworks. You should have Python installed on your system, along with PyTorch, ONNX, and other dependencies specified in the respective frameworks' documentation.

## Step 1: Inference Using PyTorch in Half Precision

1. Navigate to the directory containing the script `0.torch_inference_and_onnx_export.py`.
2. Run the following command to execute the script:
   ```bash
   python 0.torch_inference_and_onnx_export.py
   ```
   This script performs model inference in half precision (`torch.float16`) and outputs a tensor, such as:
   ```
   tensor([[[-0.7817,  0.9946, -0.1107,  ...,  1.9062,  1.5459,  0.8970]]],
       device='cuda:0', dtype=torch.float16)
   ```
