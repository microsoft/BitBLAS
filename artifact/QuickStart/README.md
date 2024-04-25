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
   tensor([[[-0.0641, -0.1862, -1.0596, ..., -0.9717, 1.0879, -1.3789]]],
   ```

## Step 2: Compile and Run ONNX Model Using Ladder

1. Set up environment variables for Ladder framework paths:
   ```bash
   export LADDER_HOME=$(pwd)/../../
   export LADDER_TVM_HOME=$LADDER_HOME/3rdparty/tvm
   export LADDER_CUTLASS_HOME=$LADDER_HOME/3rdparty/cutlass
   export PYTHONPATH=$LADDER_HOME/python:$LADDER_TVM_HOME/python:$PYTHONPATH
   export CPLUS_INCLUDE_PATH=$LADDER_CUTLASS_HOME/include
   ```
2. Compile the ONNX model using Ladder:
   ```bash
   python ladder_from_onnx.py --prefix ./llama2_70b_single_layer/model.onnx
   ```
   This may take several minutes. After compilation, the compiled model will be saved in `./progress/e2e/ladder_from_onnx`.
3. To run the compiled model, use:
   ```bash
   python ladder_from_onnx.py --prebuilt_path ./progress/e2e/ladder_from_onnx
   ```
   The output tensor will appear, such as:
   ```
   [array([[[-0.072, -0.1818, -1.059, ..., -0.969, 1.083, -1.375]]]],
   ```

## Step 3: Inference with Quantized ONNX Model

1. Update the environment variables for AutoGPTQ:
   ```bash
   export AUTOGPTQ_HOME=$LADDER_HOME/artifact/baseline_framework/AutoGPTQ.tvm
   export PYTHONPATH=$AUTOGPTQ_HOME:$PYTHONPATH
   ```
2. Perform inference using the script `1.auto_gptq_inference_and_onnx_export.py`:
   ```bash
   python 1.auto_gptq_inference_and_onnx_export.py
   ```
   This will output a quantized tensor, like:
   ```
   tensor([[[4.3477, 4.3359, 6.2500, ..., 4.2500, 4.4180, 4.0469]]], device='cuda:0', dtype=torch.float16)
   ```
3. To compile and run the quantized ONNX model using Ladder, follow similar steps as in Step 2, but with the specific ONNX model for quantized inference:
   ```bash
   python ladder_from_onnx.py --prefix ./qmodels/opt-125m-4bit/qmodel_b1s1/qmodel_b1s1.onnx
   ```
   Then execute:
   ```bash
   python ladder_from_onnx.py --prebuilt_path ./progress/e2e/ladder_from_onnx
   ```
   The output should match the tensor provided by the quantization inference script.

