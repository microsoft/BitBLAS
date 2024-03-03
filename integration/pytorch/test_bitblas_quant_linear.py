# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import bitblas
import pytest
import copy
from bitblas_quant_linear import QuantLinear
import torch
import torch.nn as nn

# !pip install auto-gptq
from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import (
    QuantLinear as CudaOldQuantLinear,
)

torch.manual_seed(0)


def gen_quant4(k, n, groupsize=-1):
    maxq = 2**4 - 1
    w = torch.randn((k, n), dtype=torch.half, device="cpu")

    original_w = w.clone()

    if groupsize == -1:
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
    w += (maxq) // 2

    w = torch.clamp(w, 0, maxq)

    # Dequantize.
    ref = (w - (maxq) // 2).half() * s

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

    return original_w, linear, s, (w - (maxq) // 2)


@pytest.mark.parametrize(
    "m, infeatures, outfeatures, bits, group_size, bias",
    [
        (1, 1024, 4096, 4, -1, False),
        (1, 1024, 4096, 4, 128, False),
        (1, 1024, 4096, 4, 128, True),
    ],
)
def test_quantization_accuracy(m, infeatures, outfeatures, bits, group_size, bias):
    original_w, linear, s, qw = gen_quant4(infeatures, outfeatures, group_size)

    if group_size == -1:
        group_size = infeatures
    zeros = torch.full((infeatures // group_size, outfeatures), 7, dtype=torch.int32)

    bitblas_zeros = zeros.clone().T
    cuda_old_linear = CudaOldQuantLinear(
        bits=bits,
        group_size=group_size,
        infeatures=infeatures,
        outfeatures=outfeatures,
        bias=bias,
    )
    cuda_old_linear.pack(linear, s.T, zeros.T, g_idx=None)

    linear_module = torch.nn.Linear(
        in_features=infeatures,
        out_features=outfeatures,
        bias=False,
        dtype=torch.float16,
        device="cuda",
    )
    linear_module.weight.data.copy_(linear.weight.data)

    scales = s.to("cuda")
    bitblas_qlinear = QuantLinear(bits, group_size, infeatures, outfeatures, bias)

    bitblas_qlinear.pack(
        linear_module.to("cuda"),
        scales=scales.T.contiguous().to("cuda"),
        zeros=bitblas_zeros.contiguous().to("cuda"),
    )

    inp = torch.rand(m, infeatures, dtype=torch.float16, device="cuda")

    cuda_old_linear = cuda_old_linear.to("cuda")
    bitblas_qlinear = bitblas_qlinear.to("cuda")
    with torch.no_grad():
        res_original = linear_module(inp)
        res_cuda_old = cuda_old_linear(inp)
        res_bitblas = bitblas_qlinear(inp)
    # Verify the accuracy of the quantized outputs against the original
    torch.testing.assert_close(res_cuda_old, res_original, rtol=1e-0, atol=1e-2)
    torch.testing.assert_close(res_bitblas, res_original, rtol=1e-0, atol=1e-2)


if __name__ == "__main__":
    # bitblas.testing.main()
    test_quantization_accuracy(1, 1024, 4096, 4, -1, False)
