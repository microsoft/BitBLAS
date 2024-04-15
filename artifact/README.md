# OSDI'24 Ladder Artifacts Evaluation

## 0. Overview
This code branch is used for OSDI'24 Artifact Evaluation of paper #626, titled "Ladder: Enabling Efficient Low-Precision Deep Learning Computing through Hardware-aware Tensor Transformation".

### Evaluation Setup
* Artifacts Available:
    * All Ladder related code are available under NNFusion open-source project located in: [https://github.com/microsoft/bitblas/tree/ladder_artifact](https://github.com/microsoft/nnfusion/tree/ladder_artifact)
* Artifacts Functional:
    * *Documentation*: the following of documents include detailed guidelines on how to build, install, test Ladder and the experiments to compare with other baselines.
    * *Completeness*: the source code under `python/ladder` folder includes all the key components of Ladder. And the single operator part of Ladder has been merged into Microsoft BitBLAS, and the end2end optimization part is available in this artifact.
    * *Exercisability*: under the *artifacts* folder, we prepare all the script and data to reproduce the experiements in individual folders named by the figure name in paper.
* Results Reproduced:
    * To reproduce the main results presented in our paper, we provide Docker images containing all the environments and baseline software, and machines with the same configurations as we used in paper evaluation. We also provide detailed guideline to help reproduce the results step by step. 

## 1. Environment Preparation
### NVIDIA GPU
To ease the process of installing all the dependencies, baseline software, and Ladder code, we provide a Dockerfile and a simple guideline to build a Docker image with all of above installed.

```bash
cd ladder
# build the image
docker build -t ladder_cuda .
# run the container
nvidia-docker run -it --cap-add=SYS_ADMIN --network=host --name ladder_test ladder_cuda bash
```

### AMD GPU
Please prepare four dockers for running TVM, ONNXRuntime, PyTorch \& Ladder respectively.
* download code
    ```bash

    ```
* Build and run ladder docker
    ```bash

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

Note that results in Figure13/part of Table 6 requires ROCM-GPU/GraphCore IPU environments which is not directly available here.

### <a id="Figure1">Figure1 </a>

The run instruction is

```bash
python run_all.py
```

### <a id="Figure2"> Figure2 </a>

The run instruction is

```bash
python run_all.py
```

### <a id="f2">Figure9-10</a>

This figure includes several baselines. The for Ladder, onnxruntime, pytorch, tensorrt and Rammer are

```bash
python profile_rammer_all.py
python profile_ort_all.py
python profile_torch_all.py
python profile_ladder_all.py
python profile_trt_all.py
```

The run instruction for Ansor is below, it requires additional action before running it.

```bash
# Our tunning log for Ansor only applies for this version.
cd /root/tvm/build && git checkout v0.9.0 && make -j
# after switching branch
cd -
python profile_ansor_all.py
# don't forget to get back
cd /root/tvm/build && git checkout ladder && make -j
```

### <a id="f3">Figure11</a>

The run instruction is

```bash
python profile_ladder_no_tc.py
```
Note that the baseline(Ansor)'s result is already shown in the above section with profile_ansor_all.py.

### <a id="f4"> Figure13</a>

The run instructions are

```bash
# measure latency, IRS and kernel count
python get_IRS.py
# measure memory perf
python get_metrics.py

# measure Ansor's latency, IRS, kernel count and memory perf
cd /root/tvm/build && git checkout v0.9.0 && make -j
python get_ansor_data.py
cd /root/tvm/build && git checkout ladder && make -j
```
Note 1: get_ansor_data.py requires TVM v0.9.0, please switch to that branch following the above instructions.

Note 2: Memory perf (Load/Store trans) from get_ansor_data.py should be halfed because the evaluator actually runs the model twice.

### <a id="f5">Table3</a>

The run instruction is

```bash
python run_ft_cpp_all.py
```

If Faster Transformers is not installed, please follow the following commands:

```bash
git clone https://github.com/NVIDIA/FasterTransformer
cd FasterTransformer
git checkout release/v5.2_bug_fix_tag
# remove line 20 add_definitions("-DENABLE_BF16") in CMakeLists.txt
# we don't use BF16 and this will cause compile error.
mkdir build && cd build
cmake .. -DSM=70 -DCMAKE_BUILD_TYPE=Release
make bert_example bert_gemm vit_example vit_gemm swin_example swin_gemm -j
```

### <a id="f6">Table5</a>

The run instruction is

```bash
python estimate_run_time_ladder.py
python estimate_run_time_ansor.py
```

### <a id="f7">Table6</a>

The run instruction is

```bash
python run_all.py
```

### <a id="f8">Table7</a>

The run instruction is

```bash
bash run_all.sh
```

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

```bash
# to evaluate inference performance, you can directly use an executable
cd PREFIX/nnfusion_rt/cuda_codegen
./build/main_test

# OR use python script which feeds data with pytorch and ctypes
python3 run_ladder.py PREFIX
```

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
