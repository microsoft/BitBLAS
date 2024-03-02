# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas_linear import Linear as BitBLASLinear
import torch
import numpy as np
import torch.nn as nn
import bitblas
import pytest
torch.manual_seed(0)

@pytest.mark.parametrize(
"m, infeatures, outfeatures, bias",
[
    (1, 1024, 1024, False),
    (1, 1024, 1024, True),
    (1024, 1024, 1024, False),
])
def test_correctness_static_shape(m, infeatures, outfeatures, bias):
    linear_torch = nn.Linear(infeatures, outfeatures, bias=bias).to(torch.float16).cuda()
    linear_bitblas = BitBLASLinear(
        infeatures,
        outfeatures,
        bias=bias,
        dtype=torch.float16,
        opt_features=m,
        enable_tuning=False
    ).cuda()

    with torch.no_grad():
        linear_bitblas.weight = nn.Parameter(linear_torch.weight.clone())
        if bias:
            linear_bitblas.bias = nn.Parameter(linear_torch.bias.clone())
    
    with torch.no_grad():
        input_data = torch.randn(m, infeatures, dtype=torch.float16).cuda()
        output_torch = linear_torch(input_data)
        output_bitblas = linear_bitblas(input_data)
    
    np.testing.assert_allclose(output_torch.cpu().numpy(), output_bitblas.cpu().numpy(), rtol=1e-1, atol=1e-2)
    # assert torch.testing.ass allclose(output_torch, output_bitblas, rtol=1e-1, atol=1e-2)

if __name__ == "__main__":
    # bitblas.testing.main()
    test_correctness_static_shape(256, 256, 256, False)
