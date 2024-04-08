# Quick Start


BitBLAS provides two Python APIs to perform mixed-precision matrix multiplication:
  - ```bitblas.Matmul``` implements the $W_{wdtype}A_{adtype}$ mixed-precision matrix multiplication of $C_{cdtype}[M, N] = A_{adtype}[M, K] \times W_{wdtype}[N, K]$ where $W_{wdtype}$ indicates the weight of $wtype$, A_{adtype} indicates the activation of $adtype$, and C_{cdtype} indicates the output of $cdtype$.
  - ```bitblas.Linear``` is a PyTorch ```nn.Linear```-like module to support a Linear of mixed-precision.

## Example: $W_{INT4}A_{FP16}$ mixed-precision matrix multiplication

Here is an example for a $W_{INT4}A_{FP16}$ mixed-precision matrix multiplication: $out_{FP16}[M, N] = A_{FP16}[M, K] \times W_{INT4}[N, K]$

```python
import bitblas

matmul_config = bitblas.MatmulConfig(
    M=32, # M dimension
    N=1024, # N dimension
    K=1024, # K dimension
    A_dtype="float16", # activation A dtype
    W_dtype="int4", # weight W dtype
    accum_dtype="float16", # accumulation dtype
    out_dtype="float16", # output dtype
    layout="nt", # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
    bias=None, # bias

    # configs for weight only quantization
    group_size=None, # setting for grouped quantization
    with_scaling=None, # setting for scaling factor
    with_zeros=None, # setting for zeros
    zeros_mode=None, # setting for how to calculating zeros
)
matmul = bitblas.Matmul(
    config=matmul_config
)

activation = torch.rand((1024, 1024), dtype=torch.float16).cuda()
weight = torch.rand((1024, 1024), dtype=torch.float16).cuda()
TODO(process weight to int4)
output = matmul(activation, weight_int4)
```

TODO (correctness)
```

```


TODO (show performance)


## Example: bitblas.Linear module for PyTorch

BitBLAS also implemented a variant PyTorch ```nn.Linear``` module, i.e., ```bitblas.Linear```, to support a Linear of mixed-precision. (link to code)

Here is an example to define a ```bitblas.Linear``` of $W_{INT4}A_{FP16}$:

```python
import bitblas

model = bitblas.Linear(
    in_features = 1024,
    out_features = 1024,
    bias = False,
    device = None,
    A_dtype="float16", # activation A dtype
    W_dtype="int4", # weight W dtype
    accum_dtype="float16", # accumulation dtype
    out_dtype="float16", # output dtype

    # configs for weight only quantization
    group_size=None, # setting for grouped quantization
    with_scaling=None, # setting for scaling factor
    with_zeros=None, # setting for zeros
    zeros_mode=None, # setting for how to calculating zeros
)

activation = torch.rand((1024, 1024), dtype=torch.float16).cuda()
TODO(process weight to int4)
model.load_state_dict(xxxxxx)
output = model(activation)
```