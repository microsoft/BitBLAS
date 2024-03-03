# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas_linear import Linear as BitBLASLinear
import torch
import time
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
    (1024, 1024, 1024, True),
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
    
    torch.testing.assert_close(output_torch, output_bitblas, rtol=1e-1, atol=1e-2)


def profile(func, *args, **kwargs):
    start_time = time.time()
    func(*args, **kwargs)  # Run the function once
    elapsed_time = time.time() - start_time
    torch.cuda.synchronize()
    num_runs = int(100 / (elapsed_time * 1000))  # Calculate the number of runs for 100ms
    start_time = time.time()
    for _ in range(num_runs):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed_time = (time.time() - start_time) / num_runs
    
    return elapsed_time * 1e3

@pytest.mark.parametrize(
"m, infeatures, outfeatures, bias",
[
    (1, 1024, 1024, False),
    (1024, 1024, 1024, False),
])
def profile_performance(m, infeatures, outfeatures, bias):        
    linear_bitblas = BitBLASLinear(
        infeatures,
        outfeatures,
        bias=bias,
        dtype=torch.float16,
        opt_features=m,
        enable_tuning=False
    ).cuda()

    with torch.no_grad():
        input_data = torch.randn(m, infeatures, dtype=torch.float16).cuda()
        torch_latency = profile(linear_bitblas, input_data)
        bitblas_latency = linear_bitblas.bitblas_matmul.profile_latency()

    assert abs(torch_latency - bitblas_latency) / torch_latency < 0.1
        

if __name__ == "__main__":
    bitblas.testing.main()
