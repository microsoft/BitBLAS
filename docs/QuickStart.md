# Quick Start

BitBLAS provides two Python APIs to perform mixed-precision matrix multiplication:
  - ```bitblas.Matmul``` implements the $W_{wdtype}A_{adtype}$ mixed-precision matrix multiplication of $C_{cdtype}[M, N] = A_{adtype}[M, K] \times W_{wdtype}[N, K]$ where $W_{wdtype}$ indicates the weight of $wtype$, A_{adtype} indicates the activation of $adtype$, and C_{cdtype} indicates the output of $cdtype$.
  - ```bitblas.Linear``` is a PyTorch ```nn.Linear```-like module to support a Linear of mixed-precision.

## Example: $W_{INT4}A_{FP16}$ mixed-precision matrix multiplication

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

## Example: bitblas.Linear module for PyTorch

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
intweight = torch.randint(-7, 7, (1024, 1024), dtype=torch.int8).cuda()

# Load and transform weights into the BitBLAS linear module
model.load_and_transform_weight(intweight)

# Save the state of the model
torch.save(model.state_dict(), "./model.pth")

# Load the model state
model.load_state_dict(torch.load("./model.pth"))

# Set the model to evaluation mode
model.eval()

# Create a dummy input tensor
dummpy_input = torch.randn(1, 1024, dtype=torch.float16).cuda()

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

## Example: bitblas.FlashAtten module

```python
import torch
torch.random.manual_seed(0)
from flash_attn.flash_attn_interface import flash_attn_func

type_convert_map = {
    "float16": torch.float16
}

flashatten_config = FlashAttenConfig(
    batch=1,
    heads=4,
    seq_len=256,
    dim=256,
    Q_dtype="float16",
    K_dtype="float16",
    V_dtype="float16",
    Accu_dtype="float32",
    Out_dtype="float16",
    layout="nnn",
    is_causal=False)
flashatten = FlashAtten(config=flashatten_config, enable_tuning=False, backend="tl")

Q_shape = [batch, seq_len, heads, dim]
V_shape = [batch, seq_len, heads, dim]
if layout == "ntn":
    K_shape = [batch, dim, heads, seq_len]
else:
    K_shape = [batch, seq_len, heads, dim]
Out_shape = [batch, seq_len, heads, dim]
q = torch.rand(Q_shape, dtype=type_convert_map[Q_dtype]).cuda() - 0.5
k = torch.rand(K_shape, dtype=type_convert_map[K_dtype]).cuda() - 0.5
k_ref = k
if layout == "ntn":
    k_ref = k.permute((0, 3, 2, 1))
v = torch.rand(V_shape, dtype=type_convert_map[V_dtype]).cuda() - 0.5
tl_output = torch.rand(Out_shape, dtype=type_convert_map[V_dtype]).cuda()

ref_output = flash_attn_func(q, k_ref, v, causal=is_causal)
flashatten(q, k, v, output=tl_output)
print("Flash Attention lib output:", ref_output)
print("BitBLAS Tilelang output:", tl_output)
torch.testing.assert_close(tl_output, ref_output, rtol=1e-1, atol=1e-1)
```
