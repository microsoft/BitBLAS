# BitBLAS

BitBLAS is a library to support mixed-precision BLAS operations on GPUs, for example, the $W_{wdtype}A_{adtype}$ mixed-precision matrix multiplication where $C_{cdtype}[M, N] = A_{adtype}[M, K] \times W_{wdtype}[N, K]$.
BitBLAS aims to support efficient mixed-precision DNN model deployment, especially the $W_{wdtype}A_{adtype}$ quantization in large language models (LLMs), for example, the $W_{UINT4}A_{FP16}$ in [GPTQ](https://arxiv.org/abs/2210.17323), the $W_{INT2}A_{FP16}$ in [BitDistiller](https://arxiv.org/abs/2402.10631), the $W_{INT2}A_{INT8}$ in [BitNet-b1.58](https://arxiv.org/abs/2402.17764). BitBLAS is based on techniques from our paper ["Ladder: Enabling Efficient Low-Precision Deep Learning Computing through Hardware-aware Tensor Transformation"](https://www.usenix.org/conference/osdi24/presentation/wang-lei) at OSDI'24.


Some of the key features of BitBLAS include:
  - High performance matrix multiplication for both GEMV (e.g., the single batch auto-regressive decode phase in LLM) and GEMM (e.g., the batched auto-regressive decode phase and the prefill phase in LLM):
    - $W_{wdtype}A_{adtype}$ mixed-precision matrix multiplication including FP16xFP8/FP4/INT4/2/1, INT8xINT4/2/1, etc. Please checkout [support matrix](#support-matrix) for detailed data types support.
    - Matrix multiplication like FP16xFP16 and INT8xINT8.
  - Auto-Tensorization for TensorCore-like hardware instructions.
  - Implemented [integration](https://github.com/microsoft/BitBLAS/blob/main/integration/) to [PyTorch](https://pytorch.org/), [GPTQModel](https://github.com/ModelCloud/GPTQModel), [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ), [vLLM](https://github.com/vllm-project/vllm) and [BitNet-b1.58](https://huggingface.co/1bitLLM/bitnet_b1_58-3B) for LLM deployment. Please checkout [benchmark summary](#benchmark-summary) for detailed end2end LLM inference performance.
  - BitBLAS first implemented $W_{INT2}A_{INT8}$ GEMV/GEMM in [BitNet-b1.58](https://arxiv.org/abs/2402.17764) with 8x/2x speedup over cuBLAS $W_{FP16}A_{FP16}$ on A100, please checkout [op_benchmark_a100_int2_scaling](https://github.com/microsoft/BitBLAS/blob/main/images/figures/op_benchmark_a100_int2_scaling.png) for detailed benchmark results. Please checkout [BitNet-b1.58 integration](https://github.com/microsoft/BitBLAS/blob/main/integration/BitNet) for the integration with the 3rdparty reproduced BitNet-b1.58 model.
  - Support customizing mixed-precision DNN operations for your specific scenarios via the flexible DSL (TIR Script).

## Latest News

- 07/11/2024 ✨: Ladder is published and presented in OSDI'24. Please find [Ladder paper and presentation](https://www.usenix.org/conference/osdi24/presentation/wang-lei) if you are interested in the technical details of BitBLAS.
- 06/25/2024 🚀🚀: BitBLAS has been integrated into [GPTQModel](https://github.com/ModelCloud/GPTQModel)! You can now use BitBLAS as a backend in GPTQ.
- 05/04/2024 🚀🚀: We’ve added integration examples for the 1.58-bit model! Check out the files under integration/BitNet.
- 04/30/2024 🚀🚀: BitBLAS now supports FP8 TensorCore (E5M2/E4M3 * E4M3/E5M2), providing more combinations beyond the three available in cuBLAS!
- 04/19/2024 ✨: We are excited to announce that BitBLAS, a high-performance library for mixed-precision DNN model deployment, is now open source and available to the public!


## Integration Example of FasterTransformer with BitBLAS
![FasterTransformer Integration](images/gif/FasterTransformer.gif)

## Benchmark Summary

BitBLAS achieves exceptional performance across a variety of computational patterns. Below are selected results showcasing its capabilities:

- End2End Integration with Quantize Inference Kernel for AutoGPTQ and vLLM.

  <div>
    <img src="./images/figures/end2end_llama_13b_auto_gptq.png" alt="AutoGPTQ end2end performance of llama13b on A100" style="width: 24%;" />
    <img src="./images/figures/end2end_llama_70b_auto_gptq.png" alt="AutoGPTQ end2end performance of llama13b on A100" style="width: 24%;" />
    <img src="./images/figures/end2end_llama_13b_vllm.png" alt="vLLM end2end performance of llama13b on A100" style="width: 24%;" />
    <img src="./images/figures/end2end_llama_70B_vllm.png" alt="vLLM end2end performance of llama13b on A100" style="width: 24%;" />
  </div>

- Weight Only Matmul performance on A100

  <div>
    <img src="./images/figures/op_benchmark_a100_wq_gemv_e7.png" alt="gemm weight only performance on A100" style="width: 49%;" />
    <img src="./images/figures/op_benchmark_a100_wq_gemm_e7.png" alt="gemm weight only performance on A100" style="width: 49%;" />
  </div>



- TensorCore FP16/INT8 GEMM Performance Vs. Vendor Library on A100 and RTX4090

  <div>
    <img src="./images/figures/op_benchmark_consistent_gemm_fp16.png" alt="gemm fp16 performance on 4090 and a100" style="width: 49%;" />
    <img src="./images/figures/op_benchmark_consistent_gemm_int8.png" alt="gemm int8 performance on 4090 and a100" style="width: 49%;" />
  </div>

For more detailed information on benchmark sets with other formats (NF4/FP4) and other devices (RTX 3090), please refer to the [benchmark](./benchmark/README.md).

## Support Matrix

| **A_dtype** | **W_dtype** | **Accum_dtype** |     **Out_dtype**    | **BitBLAS Support** |                  **Tested Platform**                 |
|:-----------:|:-----------:|:---------------:|:--------------------:|:-------------------:|:----------------------------------------------------:|
|     FP16    |     FP16    |    FP32/FP16    |         FP16         |        **√**        | V100(SM_70)/A100(SM_80)/A6000(SM_86)/RTX 4090(SM_89) |
|     FP16    |   FP4_E2M1  |    FP32/FP16    |         FP16         |        **√**        | V100(SM_70)/A100(SM_80)/A6000(SM_86)/RTX 4090(SM_89) |
|     FP16    |   FP8_E4M3  |    FP32/FP16    |         FP16         |        **√**        | V100(SM_70)/A100(SM_80)/A6000(SM_86)/RTX 4090(SM_89) |
|     FP16    |     INT8    |    FP32/FP16    |         FP16         |        **√**        | V100(SM_70)/A100(SM_80)/A6000(SM_86)/RTX 4090(SM_89) |
|     FP16    |  UINT4/INT4 |    FP32/FP16    |         FP16         |        **√**        | V100(SM_70)/A100(SM_80)/A6000(SM_86)/RTX 4090(SM_89) |
|     FP16    |  UINT2/INT2 |    FP32/FP16    |         FP16         |        **√**        | V100(SM_70)/A100(SM_80)/A6000(SM_86)/RTX 4090(SM_89) |
|     FP16    |    UINT1    |    FP32/FP16    |         FP16         |        **√**        | V100(SM_70)/A100(SM_80)/A6000(SM_86)/RTX 4090(SM_89) |
|     FP16    |     NF4     |    FP32/FP16    |         FP16         |        **√**        | V100(SM_70)/A100(SM_80)/A6000(SM_86)/RTX 4090(SM_89) |
|     INT8    |     INT8    |      INT32      | FP32/INT32/FP16/INT8 |        **√**        | V100(SM_70)/A100(SM_80)/A6000(SM_86)/RTX 4090(SM_89) |
|     INT8    |  UINT4/INT4 |      INT32      | FP32/INT32/FP16/INT8 |        **√**        | V100(SM_70)/A100(SM_80)/A6000(SM_86)/RTX 4090(SM_89) |
|     INT8    |  UINT2/INT2 |      INT32      | FP32/INT32/FP16/INT8 |        **√**        | V100(SM_70)/A100(SM_80)/A6000(SM_86)/RTX 4090(SM_89) |
|     INT8    |    UINT1    |      INT32      | FP32/INT32/FP16/INT8 |        **√**        | V100(SM_70)/A100(SM_80)/A6000(SM_86)/RTX 4090(SM_89) |
|   FP8_E4M3  |   FP8_E4M3  |       FP32      |       FP32/FP16      |        **√**        |                    RTX 4090(SM_89)                   |
|   FP8_E5M2  |   FP8_E5M2  |       FP32      |       FP32/FP16      |        **√**        |                    RTX 4090(SM_89)                   |

We are continuously expanding the support matrix. If you have any specific requirements, please feel free to open an issue or PR.

## Getting Started with an Example

### Installing with pip

**Prerequisites for installation via wheel or PyPI**
- **Operating System**: Ubuntu 20.04 or later
- **Python Version**: >= 3.8
- **CUDA Version**: >= 11.0

The easiest way to install BitBLAS is direcly from the PyPi using pip. To install the latest version, run the following command in your terminal.

```bash
pip install bitblas
```

After installing BitBLAS, you can verify the installation by running:

```bash
python -c "import bitblas; print(bitblas.__version__)"  
```

**Note**: Currently, BitBLAS whl is only supported on Ubuntu 20.04 or later version as we build the whl files on this platform. Currently we only provide whl files for CUDA>=11.0 and with Python>=3.8. **If you are using a different platform or environment, you may need to [build BitBLAS from source](https://github.com/microsoft/BitBLAS/blob/main/docs/Installation.md#building-from-source).** More installation methods can be found in the [installation document](https://github.com/microsoft/BitBLAS/blob/main/docs/Installation.md).

## Quick Start

BitBLAS provides two Python APIs to perform mixed-precision matrix multiplication:
  - ```bitblas.Matmul``` implements the $W_{wdtype}A_{adtype}$ mixed-precision matrix multiplication of $C_{cdtype}[M, N] = A_{adtype}[M, K] \times W_{wdtype}[N, K]$ where $W_{wdtype}$ indicates the weight of $wtype$, A_{adtype} indicates the activation of $adtype$, and C_{cdtype} indicates the output of $cdtype$.
  - ```bitblas.Linear``` is a PyTorch ```nn.Linear```-like module to support a Linear of mixed-precision.

### Example: $W_{INT4}A_{FP16}$ mixed-precision matrix multiplication

Here is an example for a $W_{INT4}A_{FP16}$ mixed-precision matrix multiplication: $out_{FP16}[M, N] = A_{FP16}[M, K] \times W_{INT4}[N, K]$, the example includes the creation of input matrices, quantization of weight matrices, and execution of the multiplication. The result is then compared against a reference result obtained through conventional methods to ensure accuracy.

```python
import bitblas
import torch

# enabling debug output

bitblas.set_log_level("Debug")
matmul_config = bitblas.MatmulConfig(
    M=1,  # M dimension
    N=1024,  # N dimension
    K=1024,  # K dimension
    A_dtype="float16",  # activation A dtype
    W_dtype="int4",  # weight W dtype
    accum_dtype="float16",  # accumulation dtype
    out_dtype="float16",  # output dtype
    layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
    with_bias=False,  # bias
    # configs for weight only quantization
    group_size=None,  # setting for grouped quantization
    with_scaling=False,  # setting for scaling factor
    with_zeros=False,  # setting for zeros
    zeros_mode=None,  # setting for how to calculating zeros
)

matmul = bitblas.Matmul(config=matmul_config)

# Create input matrices
input_tensor = torch.rand((1, 1024), dtype=torch.float16).cuda()
weight_tensor = torch.randint(0, 7, (1024, 1024), dtype=torch.int8).cuda()

# Transform weight tensor to int4 data type
weight_tensor_int4 = matmul.transform_weight(weight_tensor)

# Perform mixed-precision matrix multiplication
output_tensor = matmul(input_tensor, weight_tensor_int4)

# Reference result using PyTorch matmul for comparison
ref_result = torch.matmul(input_tensor, weight_tensor.t().to(torch.float16))
# Assert that the results are close within a specified tolerance, note that the int4 randint value is a little bigger than the float16 value, so we set the atol to 1.0
print("Ref output:", ref_result)
print("BitBLAS output:", output_tensor)
torch.testing.assert_close(output_tensor, ref_result, rtol=1e-2, atol=1e-0)
```

The same example can be extended to include the quantization of the weight tensor with scaling and zeros. The following code snippet demonstrates how to quantize the weight tensor with scaling and zeros and execute the mixed-precision matrix multiplication.

```python
import bitblas
import torch

in_features = 1024
out_features = 1024
group_size = 128

matmul_config = bitblas.MatmulConfig(
    M=1,  # M dimension
    N=out_features,  # N dimension
    K=in_features,  # K dimension
    A_dtype="float16",  # activation A dtype
    W_dtype="uint4",  # weight W dtype
    accum_dtype="float16",  # accumulation dtype
    out_dtype="float16",  # output dtype
    layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
    with_bias=False,  # bias
    # configs for weight only quantization
    group_size=group_size,  # setting for grouped quantization
    with_scaling=True,  # setting for scaling factor
    with_zeros=True,  # setting for zeros
    zeros_mode="original",  # setting for how to calculating zeros
)
matmul = bitblas.Matmul(config=matmul_config)

# Define shapes for tensors
input_shape = (1, 1024)
weight_shape = (1024, 1024)
scaling_shape = (1024, 1024 // 128)
zeros_shape = (1024, 1024 // 128)
output_shape = (1, 1024)

# Create scaling and zeros tensors for quantization
scaling = torch.rand(scaling_shape, dtype=torch.float16).cuda()
zeros = torch.rand(zeros_shape, dtype=torch.float16).cuda()

# Create input tensor
input_tensor = torch.rand(input_shape, dtype=torch.float16).cuda()

# Create and transform weight tensor
weight_tensor = torch.randint(0, 7, weight_shape, dtype=torch.int8).cuda()
weight_tensor_int4 = matmul.transform_weight(weight_tensor)

# Perform mixed-precision matrix multiplication with quantization
output_tensor = matmul(input_tensor, weight_tensor_int4, scale=scaling, zeros=zeros)

rescaling_tensor = torch.zeros_like(weight_tensor, dtype=torch.float16).cuda()
# Compute reference result with manual scaling and zero-point adjustment
# rescale = (weight - zeros) * scaling
for i in range(in_features // group_size):
    for j in range(group_size):
        rescaling_tensor[:, i * group_size + j] = (
            weight_tensor[:, i * group_size + j].to(torch.float16) - zeros[:, i]
        ) * scaling[:, i]
ref_result = torch.matmul(input_tensor, rescaling_tensor.t().to(torch.float16))
# Assert that the results are close within a specified tolerance
print("Ref output:", ref_result)
print("BitBLAS output:", output_tensor)
torch.testing.assert_close(output_tensor, ref_result, rtol=1e-2, atol=1e-2)
```

The init stage of the ```bitblas.Matmul``` class will take minutes to finish, as it will use hardware informations to do a one-time kernel library initialization.

### Example: bitblas.Linear module for PyTorch

BitBLAS also implemented a variant PyTorch ```nn.Linear``` module, i.e., ```bitblas.Linear```, to support a Linear of mixed-precision. See code [implementation](../python/bitblas/module/__init__.py)

Here is an example to define a ```bitblas.Linear``` of $W_{INT4}A_{FP16}$:

```python
import bitblas
import torch

# enabling debug output
bitblas.set_log_level("Debug")

model = bitblas.Linear(
    in_features=1024,
    out_features=1024,
    bias=False,
    A_dtype="float16",  # activation A dtype
    W_dtype="int4",  # weight W dtype
    accum_dtype="float16",  # accumulation dtype
    out_dtype="float16",  # output dtype
    # configs for weight only quantization
    group_size=None,  # setting for grouped quantization
    with_scaling=False,  # setting for scaling factor
    with_zeros=False,  # setting for zeros
    zeros_mode=None,  # setting for how to calculating zeros
    # Target optimization var for dynamic symbolic.
    # For detailed information please checkout docs/PythonAPI.md
    # By default, the optimization var is [1, 16, 32, 64, 128, 256, 512]
    opt_M=[1, 16, 32, 64, 128],
)

# Create an integer weight tensor
intweight = torch.randint(-7, 7, (1024, 1024), dtype=torch.int8)

# Load and transform weights into the BitBLAS linear module
model.load_and_transform_weight(intweight)

# Save the state of the model
torch.save(model.state_dict(), "./model.pth")

# Load the model state
model.load_state_dict(torch.load("./model.pth"))

# Set the model to evaluation mode
model.eval()

# Create a dummy input tensor
dummpy_input = torch.randn(1, 1024, dtype=torch.float16)

# Perform inference
output = model(dummpy_input)
print("BitBLAS output:", output)
# Please checkout the correctness evaluation code in `testing/python/module/test_bitblas_linear.py`
```

we also provide repack interface to repack the pretrained weight of AutoGPTQ into the format of BitBLAS. Here is an example to repack the pretrained weight of AutoGPTQ:

```python
# !pip install auto-gptq
import bitblas
import torch
from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import (
    QuantLinear as CudaOldQuantLinear,
)

# enabling debug output
bitblas.set_log_level("Debug")

in_features = 1024
out_features = 1024
group_size = 128

original_w, linear, s, qw = bitblas.quantization.gen_quant4(
    in_features, out_features, group_size
)
zeros = torch.full((in_features // group_size, out_features), 7, dtype=torch.int32)

cuda_old_linear = CudaOldQuantLinear(
    bits=4,
    group_size=group_size,
    infeatures=in_features,
    outfeatures=out_features,
    bias=False,
)
cuda_old_linear.pack(linear, s.T, zeros.T, g_idx=None)

bitblas_linear = bitblas.Linear(
    in_features=in_features,
    out_features=out_features,
    bias=False,
    A_dtype="float16",  # activation A dtype
    W_dtype="uint4",  # weight W dtype
    accum_dtype="float16",  # accumulation dtype
    out_dtype="float16",  # output dtype
    # configs for weight only quantization
    group_size=group_size,  # setting for grouped quantization
    with_scaling=True,  # setting for scaling factor
    with_zeros=True,  # setting for zeros
    zeros_mode="quantized",  # setting for how to calculating zeros
)
# Repack weights from CudaOldQuantLinear to BitBLAS linear module
bitblas_linear.repack_from_gptq(cuda_old_linear)

# Prepare input data
m = 1  # Batch size
inp = torch.rand(m, in_features, dtype=torch.float16, device="cuda")

# Move models to CUDA for execution
cuda_old_linear = cuda_old_linear.to("cuda")
bitblas_linear = bitblas_linear.to("cuda")

# Perform inference without gradient calculations
with torch.no_grad():
    res_cuda_old = cuda_old_linear(inp)
    res_bitblas = bitblas_linear(inp)

print("CudaOldQuantLinear output:", res_cuda_old)
print("BitBLAS output:", res_bitblas)

# Verify the outputs are close within specified tolerances
torch.testing.assert_close(res_bitblas, res_cuda_old, rtol=1e-0, atol=1e-1)
```

## Other Documents

- [Python API](https://github.com/microsoft/BitBLAS/blob/main/docs/PythonAPI.md): The Python API doc of BitBLAS.

- [Integration](https://github.com/microsoft/BitBLAS/tree/main/integration): Explore how BitBLAS seamlessly integrates with LLM deployment frameworks through our examples. Discover the ease of integrating BitBLAS with PyTorch, AutoGPTQ, and vLLM in the 3rd-party integration examples.

- [Customization](https://github.com/microsoft/BitBLAS/blob/main/docs/ExtendOperatorsWithDSL.md): BitBLAS supports implementing customized mixed-precision DNN operations rather than matrix multiplication with the flexible DSL (TIR Script).


## Reference

Please cite BitBLAS/Ladder in your publications if it helps your research:
```tex
@inproceedings {ladder-osdi24,
author = {Lei Wang and Lingxiao Ma and Shijie Cao and Quanlu Zhang and Jilong Xue and Yining Shi and Ningxin Zheng and Ziming Miao and Fan Yang and Ting Cao and Yuqing Yang and Mao Yang},
title = {Ladder: Enabling Efficient Low-Precision Deep Learning Computing through Hardware-aware Tensor Transformation},
booktitle = {18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)},
year = {2024},
isbn = {978-1-939133-40-3},
address = {Santa Clara, CA},
pages = {307--323},
url = {https://www.usenix.org/conference/osdi24/presentation/wang-lei},
publisher = {USENIX Association},
month = jul
}
```


## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the Microsoft Open Source Code of Conduct. For more information see the Code of Conduct FAQ or contact opencode@microsoft.com with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
