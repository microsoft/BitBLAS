# Quick Start


BitBLAS provides two Python APIs to perform mixed-precision matrix multiplication:
  - ```bitblas.Matmul``` implements the $W_{wdtype}A_{adtype}$ mixed-precision matrix multiplication of $C_{cdtype}[M, N] = A_{adtype}[M, K] \times W_{wdtype}[N, K]$ where $W_{wdtype}$ indicates the weight of $wtype$, A_{adtype} indicates the activation of $adtype$, and C_{cdtype} indicates the output of $cdtype$.
  - ```bitblas.Linear``` is a PyTorch ```nn.Linear```-like module to support a Linear of mixed-precision.

## Example: $W_{INT4}A_{FP16}$ mixed-precision matrix multiplication

Here is an example for a $W_{INT4}A_{FP16}$ mixed-precision matrix multiplication: $out_{FP16}[M, N] = A_{FP16}[M, K] \times W_{INT4}[N, K]$

```python
import bitblas
import torch
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

activation = torch.rand((1024, 1024), dtype=torch.float16).cuda()
weight = torch.randint(-7, 7, (1024, 1024), dtype=torch.int8).cuda()
intweight = weight.cpu().numpy()
weight_int4 = torch.from_numpy(general_compress(intweight, source_bits=4))
# if weight transform can be applied (e.g., Ladder Layout Propagation or Fast Type Conversion).
if matmul.weight_transform is not None:
    weight_int4 = matmul.weight_transform(weight_int4.cpu()).cuda()
output = torch.empty((1024, 1024), dtype=torch.float16).cuda()
matmul(activation, weight_int4, output)

```

The second example includes the creation of input matrices, quantization of weight matrices, and execution of the multiplication. The result is then compared against a reference result obtained through conventional methods to ensure accuracy.

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
torch_inputs = []
torch_inputs.append(torch.rand(input_shape, dtype=torch.float16).cuda() - 0.5)
source_format, bit = "int", 4
maxq = 2 ** (bit - 1) - 1
zeros = maxq
torch_inputs.append(torch.randint(-maxq, maxq, weight_shape, dtype=torch.int8).cuda())


torch_inputs.append(torch.rand(output_shape, dtype=torch.float16).cuda())

intweight = torch_inputs[1]
intweight = intweight.cpu().numpy().astype(np.int8)
intweight = intweight + maxq

bias = torch.rand((output_shape[-1],), dtype=torch.float16).cuda()
ref_result = torch.matmul(torch_inputs[0], (torch_inputs[1].t()).to(torch.float16))

qw_np = general_compress(intweight, source_bits=bit, storage_dtype=np.int8)
qw_torch = torch.from_numpy(qw_np).cuda()
bitblas_inputs = []
bitblas_inputs.append(torch_inputs[0])
if matmul.weight_transform is not None:
    bitblas_inputs.append(matmul.weight_transform(qw_torch.cpu()).cuda())
else:
    bitblas_inputs.append(qw_torch)

bitblas_inputs.append(torch_inputs[-1])
matmul(*bitblas_inputs)
torch.testing.assert_close(bitblas_inputs[-1], ref_result, rtol=1e2, atol=1e-1)
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

dummpy_inp = torch.rand((1, 1024), dtype=torch.float16).cuda() - 0.5

with torch.no_grad():
    output_bitblas = linear_bitblas(dummpy_inp)

# Please checkout the correctness evaluation code in `testing/python/module/test_bitblas_linear.py`
```