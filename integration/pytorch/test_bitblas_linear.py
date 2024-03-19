# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas_linear import Linear as BitBLASLinear
import torch
import time
import numpy as np
import torch.nn as nn
import pytest

torch.manual_seed(0)


@pytest.mark.parametrize(
    "m, infeatures, outfeatures, bias",
    [
        (1, 1024, 1024, False),
        (1, 1024, 1024, True),
        (1024, 1024, 1024, False),
        (1024, 1024, 1024, True),
    ],
)
def test_correctness_static_shape(m, infeatures, outfeatures, bias):
    linear_torch = (nn.Linear(infeatures, outfeatures, bias=bias).to(torch.float16).cuda())
    linear_bitblas = BitBLASLinear(
        infeatures,
        outfeatures,
        bias=bias,
        dtype=torch.float16,
        opt_features=m,
        enable_tuning=False,
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


def profile(model, input_data):
    model = model.cuda()
    model.eval()
    output = torch.empty(
        input_data.shape[:-1] + (model.outfeatures,),
        dtype=input_data.dtype,
        device=input_data.device,
    )

    def get_runtime(num_repeats=1):
        tic = time.time()
        for _ in range(num_repeats):
            _ = model(input_data, output)
        torch.cuda.synchronize()
        return (time.time() - tic) * 1000 / num_repeats

    with torch.no_grad():
        # print("Warming up ...")
        st = time.time()
        while time.time() - st < 1.0:
            get_runtime()  # warmup
        warmup_runtime = get_runtime()
        num_repeats = max(1, int(1000 / warmup_runtime))
        times = get_runtime(num_repeats)
    return np.mean(times)


@pytest.mark.parametrize(
    "m, infeatures, outfeatures, bias",
    [
        (1, 1024, 1024, False),
        (1024, 1024, 1024, False),
    ],
)
def test_profile_performance(m, infeatures, outfeatures, bias):
    linear_bitblas = BitBLASLinear(
        infeatures,
        outfeatures,
        bias=bias,
        dtype=torch.float16,
        opt_features=m,
        enable_tuning=False,
    ).cuda()
    with torch.no_grad():
        input_data = torch.randn(m, infeatures, dtype=torch.float16).cuda()
        torch_latency = profile(linear_bitblas, input_data)
        bitblas_latency = linear_bitblas.bitblas_matmul.profile_latency()
    print(f"torch_latency: {torch_latency}, bitblas_latency: {bitblas_latency}")
    assert (abs(torch_latency - bitblas_latency) / torch_latency <
            0.1), f"torch_latency: {torch_latency}, bitblas_latency: {bitblas_latency}"


if __name__ == "__main__":
    # bitblas.testing.main()
    test_profile_performance(1, 16384, 16384, False)
