from quant_linear import QuantLinear
import copy
import torch
import torch.nn as nn

# !pip install auto-gptq
from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import (
    QuantLinear as CudaOldQuantLinear,
)


def gen_quant4(k, n, groupsize=-1):
    maxq = 2**4 - 1
    w = torch.randn((k, n), dtype=torch.half, device="cpu")

    original_w = w.clone()
    if group_size == -1:
        groupsize = k

    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))

    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq

    # Quantize.
    w = torch.round(w / s).int()

    # Unsigned storage.
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    print("quantize weight is: ")
    print((w - (maxq + 1) // 2))
    # Dequantize.
    ref = (w - (maxq + 1) // 2).half() * s

    if groupsize != -1:

        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((k, n)).contiguous()
            return w

        ref = reshape(ref)
        w = reshape(w)

    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(k, n, bias=False)
    linear.weight.data = ref.t()

    return original_w, linear, s


bits = 4
m = 1
group_size = 1024
infeatures = 1024  # this is k of weight (n, k)
outfeatures = 4096  # this is n of weight (n, k)
bias = False

_, linear, s = gen_quant4(infeatures, outfeatures, group_size)

cuda_old_linear = CudaOldQuantLinear(
    bits=4,
    group_size=group_size,
    infeatures=infeatures,
    outfeatures=outfeatures,
    bias=False,
)
zeros = torch.full((infeatures // group_size, outfeatures), 8, dtype=torch.int32)

cuda_old_linear.pack(linear, s.T, zeros.T, g_idx=None)
linear_module = torch.nn.Linear(
    in_features=infeatures,
    out_features=outfeatures,
    bias=False,
    dtype=torch.float16,
    device="cuda",
)
linear_module.weight.data.copy_(
    linear.weight.data
)  # Not using dequantized_weight to avoid approx

scales = s.to("cuda")
bitblas_qlinear = QuantLinear(bits, group_size, infeatures, outfeatures, bias)

bitblas_qlinear.pack(
    linear_module.to("cuda"),
    scales=scales.T.contiguous().to("cuda"),
)

inp = torch.rand(m, infeatures, dtype=torch.float16, device="cuda")

cuda_old_linear = cuda_old_linear.to("cuda")
bitblas_qlinear = bitblas_qlinear.to("cuda")
with torch.no_grad():
    res_cuda_old = cuda_old_linear(inp)
    res_bitblas = bitblas_qlinear(inp)
print(res_cuda_old)
print(res_bitblas)
