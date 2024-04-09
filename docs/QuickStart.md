# Quick Start


BitBLAS provides two Python APIs to perform mixed-precision matrix multiplication:
  - ```bitblas.Matmul``` implements the $W_{wdtype}A_{adtype}$ mixed-precision matrix multiplication of $C_{cdtype}[M, N] = A_{adtype}[M, K] \times W_{wdtype}[N, K]$ where $W_{wdtype}$ indicates the weight of $wtype$, A_{adtype} indicates the activation of $adtype$, and C_{cdtype} indicates the output of $cdtype$.
  - ```bitblas.Linear``` is a PyTorch ```nn.Linear```-like module to support a Linear of mixed-precision.

## Example: $W_{INT4}A_{FP16}$ mixed-precision matrix multiplication

Here is an example for a $W_{INT4}A_{FP16}$ mixed-precision matrix multiplication: $out_{FP16}[M, N] = A_{FP16}[M, K] \times W_{INT4}[N, K]$

```python
import bitblas
import torch

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

activation = torch.rand((1024, 1024), dtype=torch.float16).cuda()
weight = torch.randint(-7, 7, (1024, 1024), dtype=torch.int8).cuda()
weight_int4 = matmul.transform_weight(weight)
output = matmul(activation, weight_int4, output)
```

The second example includes the creation of input matrices, quantization of weight matrices, and execution of the multiplication. The result is then compared against a reference result obtained through conventional methods to ensure accuracy.

```python
import bitblas
import torch

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

input_shape = (1, 1024)
weight_shape = (1024, 1024)
output_shape = (1, 1024)

maxq = 7

input_tensor = torch.rand(input_shape, dtype=torch.float16).cuda()
weight_tensor = torch.randint(0, maxq, weight_shape, dtype=torch.int8).cuda()
output_tensor = torch.rand(output_shape, dtype=torch.float16).cuda()
ref_result = torch.matmul(input_tensor, weight_tensor.t().to(torch.float16))

bitblas_inputs = [input_tensor]
weight_tensor_int4 = matmul.transform_weight(weight_tensor)
bitblas_inputs.append(weight_tensor_int4)
output_tensor = matmul(*bitblas_inputs)

torch.testing.assert_close(output_tensor, ref_result, rtol=1e-2, atol=1e-0)
```



To highlight the efficiency gains achievable with BitBLAS, the following code snippet demonstrates how to measure the latency of the mixed-precision matrix multiplication. By profiling the operation, users can quantify the performance benefits of BitBLAS.

```python
import bitblas
import torch

import numpy as np
from bitblas.quantization import general_compress

matmul_config = bitblas.MatmulConfig(
    M=1,  # M dimension
    N=1024,  # N dimension
    K=1024,  # K dimension
    A_dtype="float16",  # activation A dtype
    W_dtype="int4",  # weight W dtype
    accum_dtype="float16",  # accumulation dtype
    out_dtype="float16",  # output dtype
    layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
    with_bias=None,  # bias
    # configs for weight only quantization
    group_size=None,  # setting for grouped quantization
    with_scaling=None,  # setting for scaling factor
    with_zeros=None,  # setting for zeros
    zeros_mode=None,  # setting for how to calculating zeros
)
matmul = bitblas.Matmul(config=matmul_config)
latency = matmul.profile_latency()
print(f"Latency: {latency} ms")
# improve performance through fine tuning
matmul.hardware_aware_finetune()
print(f"Latency After Tuning: {latency} ms")
```

## Example: bitblas.Linear module for PyTorch

BitBLAS also implemented a variant PyTorch ```nn.Linear``` module, i.e., ```bitblas.Linear```, to support a Linear of mixed-precision. (link to code)

Here is an example to define a ```bitblas.Linear``` of $W_{INT4}A_{FP16}$:

```python
import bitblas
import torch

model = bitblas.Linear(
    infeatures=1024,
    outfeatures=1024,
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
)


intweight = torch.randint(-7, 7, (1024, 1024), dtype=torch.int8)

model.load_and_transform_weight(intweight)

# save model
torch.save(model.state_dict(), "./debug/model.pth")

model.load_state_dict(torch.load("./debug/model.pth"))

model.eval()

dummpy_input = torch.randn(1, 1024, dtype=torch.float16)
output = model(dummpy_input)
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

infeatures = 1024
outfeatures = 1024
group_size = 128

original_w, linear, s, qw = bitblas.quantization.gen_quant4(infeatures, outfeatures, group_size)
zeros = torch.full((infeatures // group_size, outfeatures), 7, dtype=torch.int32)

cuda_old_linear = CudaOldQuantLinear(
    bits=4,
    group_size=group_size,
    infeatures=infeatures,
    outfeatures=outfeatures,
    bias=False,
)
cuda_old_linear.pack(linear, s.T, zeros.T, g_idx=None)

bitblas_linear = bitblas.Linear(
    infeatures=infeatures,
    outfeatures=outfeatures,
    bias=False,
    A_dtype="float16",  # activation A dtype
    W_dtype="uint4",  # weight W dtype
    accum_dtype="float16",  # accumulation dtype
    out_dtype="float16",  # output dtype
    # configs for weight only quantization
    group_size=group_size,  # setting for grouped quantization
    with_scaling=True,  # setting for scaling factor
    with_zeros=True,  # setting for zeros
    zeros_mode="original",  # setting for how to calculating zeros
)
bitblas_linear.repack_from_gptq(cuda_old_linear)

m = 1 # batch size
inp = torch.rand(m, infeatures, dtype=torch.float16, device="cuda")

cuda_old_linear = cuda_old_linear.to("cuda")
bitblas_linear = bitblas_linear.to("cuda")
with torch.no_grad():
    res_cuda_old = cuda_old_linear(inp)
    res_bitblas = bitblas_linear(inp)

torch.testing.assert_close(res_bitblas, res_cuda_old, rtol=1e9, atol=1e-2)
```