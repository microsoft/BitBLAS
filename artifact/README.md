# OSDI'24 Ladder Artifacts Evaluation

## 0. Overview
This code branch is used for OSDI'24 Artifact Evaluation of paper #626, titled "Ladder: Enabling Efficient Low-Precision Deep Learning Computing through Hardware-aware Tensor Transformation".

### Evaluation Setup
* Artifacts Available:
    * All Ladder related code are available under Ladder open-source project located in: [https://github.com/microsoft/ladder/tree/ladder_artifact](https://github.com/microsoft/nnfusion/tree/ladder_artifact)
* Artifacts Functional:
    * *Documentation*: the following of documents include detailed guidelines on how to build, install, test Ladder and the experiments to compare with other baselines.
    * *Completeness*: the source code under `python/ladder` folder includes all the key components of Ladder. And the single operator part of Ladder has been merged into Microsoft Ladder, and the end2end optimization part is available in this artifact.
    * *Exercisability*: under the *artifacts* folder, we prepare all the script and data to reproduce the experiements in individual folders named by the figure name in paper.
* Results Reproduced:
    * To reproduce the main results presented in our paper, we provide Docker images containing all the environments and baseline software, and machines with the same configurations as we used in paper evaluation. We also provide detailed guideline to help reproduce the results step by step. 

## 1. Environment Preparation
To ease the process of installing all the dependencies, baseline software, and Ladder code, we provide a Dockerfile and a simple guideline to build a Docker image with all of above installed. The Docker image is built on top of Ubuntu 20.04, and it contains all the dependencies required to run the experiments. We only provide the Dockerfile for NVIDIA GPU, and the Dockerfile for AMD GPU will be provided upon request.

```bash
cd docker
# build the image
docker build -t ladder_cuda .
# run the container
nvidia-docker run -it --cap-add=SYS_ADMIN --network=host --name ladder_test ladder_cuda bash
```

## 2. Reproducing Individual Experiement Results
## 2. Paper result quick reproduce

Since ladder's paper evaluate different models with different batch-sizes and data-types, leading to more than 50 models to tune to completely reproduce the paper's result. To help reproduce quickly, we have uploaded all ladder's compiled model of V100 GPU at [temp.tar.gz - Google Cloud Drive](https://drive.google.com/file/d/1xJUk7ZBoe6bjaqMpTI-n9gqGtc01IOWG)

```bash
pip install gdown
gdown https://drive.google.com/u/0/uc?id=1xJUk7ZBoe6bjaqMpTI-n9gqGtc01IOWG
```

After downloading, it should be extracted under the artifacts/temp folder. You can see a lot of model folders in it. With these pre-compiled models, results can be reproduced more quickly with a few commands. Here is a list of script we provide:

| Name      | Description                                              | Commands                      |
| --------- | -------------------------------------------------------- | ----------------------------- |
| Figure8   | End2End Performance on the NVIDIA A100 GPU               | [Figure1](#Figure1)           |
| Figure9   | End2End Performance on the NVIDIA V100 GPU               | [Figure9](#f2)                |
| Figure10  | End2End Performance on the NVIDIA RTX A6000 GPU          | [Figure10](#f2)               |
| Figure11  | Memory usage of LLM inference on the NVIDIA A100 GPU     | [Figure11](#f3)                |
| Figure12  | Operator Benchmark on the NVIDIA A100 GPU                | [Figure12](#f3)                |
| Figure13  | Optimization breakdown of LLAMA on the NVIDIA A100 GPU   | [Figure13](#f4)               |
| Figure14  | Scaling the bit width of weight and activation.          | [Figure14](#f4)               |
| Figure15  | End-to-end performance on The AMD MI250 GPU              | [Figure15](#f4)               |
| Table1    | MatMul Support and its Performance of Vendor Libraries   | [Table1](#f5)                 |
| Table2    | Compilation time of AMOS, TensorIR Welder and Ladder     | [Table5](#f6)                 |

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

![Figure 8](./Figure8/png/end2end_a100.png)

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

![Figure 8](./Figure9/png/end2end_v100.png)

### <a id="f2">Figure10</a>

The Figure 10 is about the end-to-end performance of the selected baselines and the proposed method. The end-to-end performance is measured by the inference time of the model. The inference time is measured in seconds.

Run the following command to generate the results of Figure 10:

```bash
python3 run_all.py
```

The `run_all.py` script has the following options:

- `--reproduce`: bool, whether to reproduce the results, otherwise it will use our logges paper result, default value is `False`.
- `--force_tune`: bool, whether to force tune the op with AMOS/TensorIR, otherwise use the checkpoints if available, default value is `False`.

The result will be saved in the `pdf` and `png` directory, respectively. For example, the reproduced result is:

![Figure 8](./Figure10/png/end2end_a6000.png)


### <a id="f3">Figure11</a>

Figure 11 provides a comparative analysis of memory usage across two machine learning models, LLAMA and BLOOM, using various inference frameworks and precision settings. The memory usage is measured in megabytes (MB) and is benchmarked across batch sizes and sequence lengths (BS1 SEQ1, BS32 SEQ1, BS1 SEQ4096).

Run the following command to generate the results of Figure 11:

```bash
python3 run_all.py
```

The `run_all.py` script has the following options:

- `--reproduce`: bool, whether to reproduce the results, otherwise it will use our logges paper result, default value is `False`.
- `--force_tune`: bool, whether to force tune the op with AMOS/TensorIR, otherwise use the checkpoints if available, default value is `False`.

The result will be saved in the `pdf` and `png` directory, respectively. For example, the reproduced result is:

![Figure 11](./Figure11/png/memory_usage_a100.png)

### <a id="f4"> Figure12</a>

Figure 12 showcases the performance speedup of various computational kernels across different models and configurations. The speedup is measured relative to the baseline performance Bitter-$W_{FP16}A_{FP16}$.

Run the following command to generate the results of Figure 12:

```bash
python3 run_all.py
```

The `run_all.py` script has the following options:

- `--reproduce`: bool, whether to reproduce the results, otherwise it will use our logges paper result, default value is `False`.
- `--force_tune`: bool, whether to force tune the op with AMOS/TensorIR, otherwise use the checkpoints if available, default value is `False`.

The result will be saved in the `pdf` and `png` directory, respectively. For example, the reproduced result is:

![Figure 12](./Figure12/png/operator_performance_a100.png)

### <a id="f5">Table1</a>
The Table1 is about the performance of the matrix multiplication of vendor libraries. The performance is measured by the throughput of the matrix multiplication. The throughput is calculated by the number of elements in the matrix divided by the time of the matrix multiplication. The throughput is measured in GFLOPS. The throughput is calculated by the following formula:

\[
\text{{Throughput}} = \frac{{\text{{Number of elements in the matrix}}}}{{\text{{Time of the matrix multiplication}}}}
\]

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

### <a id="f6">Table5</a>

The run instruction is

```bash
./run_all.sh
```

### <a id="f7">Table6</a>

The run instruction is

```bash
./run_all.sh
```

### <a id="f8">Table2</a>

The Table2 is about the compilation time of AMOS, TensorIR Welder and Ladder. The compilation time is measured by the time of the compilation.

The Table2 is generated by the following command:

```bash
python3 run_all.py
```

The `run_all.py` script has the following options:

- `--reproduce`: bool, whether to reproduce the results, otherwise it will use our logges paper result, default value is `False`.
- `--force_tune`: bool, whether to force tune the op with AMOS/TensorIR, otherwise use the checkpoints if available, default value is `False`.

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

## 3. Getting started with Ladder

Despite using the logs provided above, you can also run ladder from scratch. To compile a model with Ladder, there are several steps.

### Step1: Create a ONNX model file:

```bash
python torch2onnx.py MODEL --prefix PREFIX [--bs BATCHSIZE] [--fp16]
```

To generate an ONNX model, we first use the script torch2onnx.py to generate an onnx file under the PREFIX folder. It is recommended to create a new PREFIX folder for every model.

The MODEL parameter can be one of the ten models evaluated in the paper (bert, vit, swin_transformer, BSRN, NAFNet, Restormer, mobilevit, Conformer, mobilenet and NeRF).

Default batchsize is 1, it can be set with --bs flag. The default datatype is float32, if --fp16 is used, the datatype will be float16.

After running this command, The PREFIX folder will be created which contains a model.onnx file. This PREFIX will be used in the following Ladder's compilation steps. Some other baselines will also use this PREFIX as the workspace.

### Step2: Compile ONNX with Ladder

Afther the PREFIX folder is created, run the following command

```bash
python tune_ladder.py PREFIX --topk 20 --arch V100
```

The command will compile the model.onnx under the PREFIX folder. The --topk 20 and --arch V100 indicates that 20 trails is made for each task(subgraph) and V100 GPU is the target.

Specially, when reproducing results in the paper, special flags will be added. The 3 included cases are: bert, fp32, bs=1,64 and swin_transformer, fp16, bs=1. In this three cases, we add an additional compile flag:

```bash
python tune_ladder.py PREFIX --topk 20 --arch V100 --skip_dot
```


Ths flag will lower some Dot kernels to CUDA library (cublas) which performs better than generated kernels in these 3 cases.

### Step3: Evaluate Latency and correctness.

After running the previous command, you can profile the latency of the ladder's generated model:

To check the correctness of Ladder's compiled model, you can run the following command to compare Ladder's output with onnx-runtime's output.

```bash
python3 test_acc.py PREFIX
```

You can also run other baselines on this model:

```bash
# torch
python run_torch.py MODEL [--bs BATCHSIZE] [--fp16]
# TensorRT
python run_trt.py --prefix PREFIX [--fp16]
# onnxruntime
python run_onnxrt.py --prefix PREFIX
# Ansor, note that Ansor requires about one day to tune for a model
python run_ansor.py --prefix PREFIX
# Astitch
python3 run_blade.py MODEL [--bs BATCHSIZE] [--fp16]
```
