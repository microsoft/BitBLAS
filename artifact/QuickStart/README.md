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
   array([[[-0.7817 ,  0.9946 , -0.11066, ...,  1.906  ,  1.546  ,
          0.897  ]]]
   ```
4. To run the compressed model, use:
   ```bash
   python ladder_from_onnx_int4_compress.py  --prefix ./llama2_70b_single_layer/model.onnx
   python ladder_from_onnx_int4_compress.py  --prebuilt_path ./progress/e2e/ladder_from_onnx
   ```
    The output tensor will appear, such as:
   ```
   array([[[-0.7817 ,  0.9946 , -0.11066, ...,  1.906  ,  1.546  ,
          0.897  ]]]
   ```
