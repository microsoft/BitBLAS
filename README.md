# OSDI'24 Ladder Artifacts Evaluation

## 0. Overview
This code branch is used for OSDI'24 Artifact Evaluation of paper #626, titled "Ladder: Enabling Efficient Low-Precision Deep Learning Computing through Hardware-aware Tensor Transformation".


* Artifacts Available:
    * All Ladder related code are available under Ladder open-source project located in: [https://github.com/microsoft/BitBLAS/tree/osdi24_ladder_artifact](https://github.com/microsoft/BitBLAS/tree/osdi24_ladder_artifact)
* Artifacts Functional:
    * *Documentation*: the following of documents include detailed guidelines on how to build, install, test Ladder and the experiments to compare with other baselines.
    * *Completeness*: the source code under `python/ladder` folder includes all the key components of Ladder. Besides, the single operator part of Ladder has been re-implemented and released as [BitBLAS](https://github.com/microsoft/BitBLAS), a library to support mixed-precision DNN operations. The end2end optimization part is available in this artifact.
    * *Exercisability*: under the *artifacts* folder, we prepare all the script and data to reproduce the experiements in individual folders named by the figure name in paper.
* Results Reproduced:
    * To reproduce the main results presented in our paper, we provide Docker images containing all the environments and baseline software as we used in paper evaluation. We also provide detailed guideline to help reproduce the results step by step. 
    * Hardware requirements:
      * Require NVIDIA A100-80GB-PCIe GPU to reproduce the main results.
      * Require NVIDIA A100-80GB-PCIe GPU, NVIDIA Tesla V100-16GB-PCIe GPU, NVIDIA RTX A6000 GPU and AMD Instinct MI250 GPU to reproduce all the results.

## 1. Environment Preparation
To ease the process of installing all the dependencies, baseline software, and Ladder code, we provide a Dockerfile and a simple guideline to build a Docker image with all of above installed. The Docker image is built on top of Ubuntu 20.04, and it contains all the dependencies required to run the experiments. We only provide the Dockerfile for NVIDIA GPU, and the Dockerfile for AMD GPU will be provided upon request.

```bash
cd docker
# build the image, this may take a while (around 30+ minutes on the author's test machine)
docker build -t ladder_cuda -f Dockerfile.cu120 .
# run the container
nvidia-docker run -it --cap-add=SYS_ADMIN --network=host --gpus all --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --name ladder_test ladder_cuda bash
```

## 2. Reproducing Individual Experiement Results

<!-- Since ladder's paper evaluate different models with different batch-sizes and data-types, leading to more than 50 models to tune to completely reproduce the paper's result. To help reproduce quickly, we have uploaded all ladder's compiled model of A100 GPU at [Checkpoints.tar.gz - Microsoft Onedrive]()

After downloading, it should be extracted under the artifacts/temp folder. You can see a lot of model folders in it. With these pre-compiled models, results can be reproduced more quickly with a few commands. Here is a list of script we provide: -->

| Figure # in paper | Experiment Description                           | Commands                      | Script Location      |
| ----------------- | ------------------------------------------------ | ----------------------------- | ----------------------------- |
| Figure8   | End2End Performance on the NVIDIA A100 GPU               | [Figure8](#Figure8)           | [artifact/Figure8](https://github.com/microsoft/BitBLAS/tree/osdi24_ladder_artifact/artifact/Figure8)           |
| Figure9   | End2End Performance on the NVIDIA V100 GPU               | [Figure9](#Figure9)           | [artifact/Figure9](https://github.com/microsoft/BitBLAS/tree/osdi24_ladder_artifact/artifact/Figure9)           |
| Figure10  | End2End Performance on the NVIDIA RTX A6000 GPU          | [Figure10](#Figure10)         | [artifact/Figure10](https://github.com/microsoft/BitBLAS/tree/osdi24_ladder_artifact/artifact/Figure10)         |
| Figure11  | Memory usage of LLM inference on the NVIDIA A100 GPU     | [Figure11](#Figure11)         | [artifact/Figure11](https://github.com/microsoft/BitBLAS/tree/osdi24_ladder_artifact/artifact/Figure11)         |
| Figure12  | Operator Benchmark on the NVIDIA A100 GPU                | [Figure12](#Figure12)         | [artifact/Figure12](https://github.com/microsoft/BitBLAS/tree/osdi24_ladder_artifact/artifact/Figure12)         |
| Figure13  | Optimization breakdown of LLAMA on the NVIDIA A100 GPU   | [Figure13](#Figure13)         | [artifact/Figure13](https://github.com/microsoft/BitBLAS/tree/osdi24_ladder_artifact/artifact/Figure13)         |
| Figure14  | Scaling the bit width of weight and activation.          | [Figure14](#Figure14)         | [artifact/Figure14](https://github.com/microsoft/BitBLAS/tree/osdi24_ladder_artifact/artifact/Figure14)         |
| Figure15  | End-to-end performance on The AMD MI250 GPU              | [Figure15](#Figure15)         | [artifact/Figure15](https://github.com/microsoft/BitBLAS/tree/osdi24_ladder_artifact/artifact/Figure15)         |
| Table1    | MatMul Support and its Performance of Vendor Libraries   | [Table1](#Table1)             | [artifact/Table1](https://github.com/microsoft/BitBLAS/tree/osdi24_ladder_artifact/artifact/Table1)         |
| Table2    | Compilation time of AMOS, TensorIR Welder and Ladder     | [Table2](#Table2)             | [artifact/Table2](https://github.com/microsoft/BitBLAS/tree/osdi24_ladder_artifact/artifact/Table2)         |

### <a id="Figure8">Figure8 </a>

The Figure 8 is about the end-to-end performance of the selected baselines and the proposed method. The end-to-end performance is measured by the inference time of the model. The inference time is measured in seconds.

Run the following command to generate the results of Figure 8:

```bash
python3 run_all.py
```

The `run_all.py` script has the following options:

- `--reproduce`: bool, whether to reproduce the results, otherwise it will use our logges paper result, default value is `False`.
- `--force_tune`: bool, whether to force tune the op with AMOS/TensorIR, otherwise use the checkpoints if available, default value is `False`.

The result will be saved in the `pdf` and `png` directory, respectively. For example, the reproduced result is:

![Figure 8](./artifact/Figure8/png/end2end_a100.png)

### <a id="Figure9"> Figure9 </a>

The Figure 9 is about the end-to-end performance of the selected baselines and the proposed method. The end-to-end performance is measured by the inference time of the model. The inference time is measured in seconds.

Run the following command to generate the results of Figure 8:

```bash
python3 run_all.py
```

The `run_all.py` script has the following options:

- `--reproduce`: bool, whether to reproduce the results, otherwise it will use our logges paper result, default value is `False`.
- `--force_tune`: bool, whether to force tune the op with AMOS/TensorIR, otherwise use the checkpoints if available, default value is `False`.

The result will be saved in the `pdf` and `png` directory, respectively. For example, the reproduced result is:

![Figure 8](./artifact/Figure9/png/end2end_v100.png)

### <a id="Figure10">Figure10</a>

The Figure 10 is about the end-to-end performance of the selected baselines and the proposed method. The end-to-end performance is measured by the inference time of the model. The inference time is measured in seconds.

Run the following command to generate the results of Figure 10:

```bash
python3 run_all.py
```

The `run_all.py` script has the following options:

- `--reproduce`: bool, whether to reproduce the results, otherwise it will use our logges paper result, default value is `False`.
- `--force_tune`: bool, whether to force tune the op with AMOS/TensorIR, otherwise use the checkpoints if available, default value is `False`.

The result will be saved in the `pdf` and `png` directory, respectively. For example, the reproduced result is:

![Figure 8](./artifact/Figure10/png/end2end_a6000.png)


### <a id="Figure11">Figure11</a>

Figure 11 provides a comparative analysis of memory usage across two machine learning models, LLAMA and BLOOM, using various inference frameworks and precision settings. The memory usage is measured in megabytes (MB) and is benchmarked across batch sizes and sequence lengths (BS1 SEQ1, BS32 SEQ1, BS1 SEQ4096).

Run the following command to generate the results of Figure 11:

```bash
python3 run_all.py
```

The `run_all.py` script has the following options:

- `--reproduce`: bool, whether to reproduce the results, otherwise it will use our logges paper result, default value is `False`.
- `--force_tune`: bool, whether to force tune the op with AMOS/TensorIR, otherwise use the checkpoints if available, default value is `False`.

The result will be saved in the `pdf` and `png` directory, respectively. For example, the reproduced result is:

![Figure 11](./artifact/Figure11/png/memory_usage_a100.png)

### <a id="Figure12"> Figure12</a>

Figure 12 showcases the performance speedup of various computational kernels across different models and configurations. The speedup is measured relative to the baseline performance Bitter-$W_{FP16}A_{FP16}$.

Run the following command to generate the results of Figure 12:

```bash
python3 run_all.py
```

The `run_all.py` script has the following options:

- `--reproduce`: bool, whether to reproduce the results, otherwise it will use our logges paper result, default value is `False`.
- `--force_tune`: bool, whether to force tune the op with AMOS/TensorIR, otherwise use the checkpoints if available, default value is `False`.

The result will be saved in the `pdf` and `png` directory, respectively. For example, the reproduced result is:

![Figure 12](./artifact/Figure12/png/operator_performance_a100.png)

### <a id="Table1">Table1</a>
The Table1 is about the performance of the matrix multiplication of vendor libraries. The performance is measured by the throughput of the matrix multiplication. The throughput is calculated by the number of elements in the matrix divided by the time of the matrix multiplication. The throughput is measured in GFLOPS. 

The Table1 is generated by the following command:

```bash
# reproduce the results of Table1 on A100
cd nvidia
./run_all.sh A100
cd ..

# reproduce the results of Table1 on V100
cd nvidia
./run_all.sh V100
cd ..

# reproduce the results of Table1 on MI250
cd amd
./run_all.sh MI250
cd ..
```

The `run_all.sh` scripts share the following common options:

- `DEVICES`: str, the device to measure, default is `A100`.
- `USE_PAPER`: bool, whether to use the paper's result as the input, default is `True`.
- `FORCE_TUNE`: bool, whether to force tune the op with AMOS/TensorIR, otherwise use the checkpoints if available, default is `False`.

The example output of the Table1 is shown below:

```bash
+---------------------------------------------------------------------------------------------------+
|                                    Performance Overview - A100                                    |
+----------+----------------------+----------------------+--------------------+---------------------+
| Library  | W$_{FP16}$A$_{FP16}$ | W$_{INT8}$A$_{INT8}$ | W$_{FP8}$A$_{FP8}$ | W$_{NF4}$A$_{FP16}$ |
+----------+----------------------+----------------------+--------------------+---------------------+
|  cuBLAS  |         87%          |         52%          |         x          |          x          |
| rocBLAS  |          x           |          x           |         x          |          x          |
|   AMOS   |         38%          |         45%          |         x          |          x          |
| TensorIR |         56%          |          x           |         x          |          x          |
|  Roller  |         70%          |          x           |         x          |          x          |
+----------+----------------------+----------------------+--------------------+---------------------+
```


### <a id="Table2">Table2</a>

The Table2 is about the compilation time of AMOS, TensorIR Welder and Ladder. The compilation time is measured by the time of the compilation.

The Table2 is generated by the following command:

```bash
python3 run_all.py
```

The `run_all.py` script has the following options:

- `--reproduce`: bool, whether to reproduce the results, otherwise it will use our logges paper result, default value is `False`.
- `--force_tune`: bool, whether to force tune the op with AMOS/TensorIR, otherwise use the checkpoints if available, default value is `False`.

```bash
+-----------------------------------------------------+
|          Transposed Compilation Time Table          |
+-----------------+------+----------+--------+--------+
|     Library     | AMOS | TensorIR | Welder | LADDER |
+-----------------+------+----------+--------+--------+
|    ResNet(1)    | 3852 |   156    |   11   |   31   |
|   ResNet(128)   | 3328 |   128    |   13   |   17   |
|  ShuffleNet(1)  | 2191 |   836    |   18   |   44   |
| ShuffleNet(128) | 3121 |   400    |   12   |   29   |
+-----------------+------+----------+--------+--------+
```


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
   python ladder_from_onnx_int4_compress.py  --prefix ./llama2_70b_single_layer/model.onnx --fast_decoding
   python ladder_from_onnx_int4_compress.py  --prebuilt_path ./progress/e2e/ladder_from_onnx_int4_compress
   ```
    The output tensor will appear, such as:
   ```
   array([[[-0.7817 ,  0.9946 , -0.11066, ...,  1.906  ,  1.546  ,
          0.897  ]]]
   ```
