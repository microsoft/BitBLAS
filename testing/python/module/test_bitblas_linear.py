# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import bitblas
from bitblas import Linear as BitBLASLinear
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
        (1024, 1024, 1024, True),
        ([1, 1024], 1024, 1024, True),
    ],
)
def test_correctness_consistent(m, infeatures, outfeatures, bias):
    linear_torch = (nn.Linear(infeatures, outfeatures, bias=bias).to(torch.float16).cuda())
    linear_bitblas = BitBLASLinear(
        infeatures,
        outfeatures,
        bias=bias,
        A_dtype="float16",
        W_dtype="float16",
        accum_dtype="float16",
        out_dtype="float16",
        opt_M=m,
    ).cuda()

    with torch.no_grad():
        linear_bitblas.weight = nn.Parameter(linear_torch.weight.clone())
        if bias:
            linear_bitblas.bias = nn.Parameter(linear_torch.bias.clone())

    with torch.no_grad():
        if not isinstance(m, int):
            # average m
            m = sum(m) // len(m)
        input_data = torch.randn(m, infeatures, dtype=torch.float16).cuda()
        output_torch = linear_torch(input_data)
        output_bitblas = linear_bitblas(input_data)
    torch.testing.assert_close(output_torch, output_bitblas, rtol=1e-1, atol=1e-2)


@pytest.mark.parametrize(
    "m, infeatures, outfeatures, bias, W_dtype, group_size, with_scaling, with_zeros, zeros_mode",
    [
        (1, 1024, 1024, False, "uint4", -1, False, False, None),
        (1, 1024, 1024, False, "uint4", -1, False, False, None),
        (1024, 1024, 1024, True, "uint4", -1, False, False, None),
        (1, 1024, 1024, True, "uint2", -1, True, False, None),
        (1, 1024, 1024, True, "uint2", 128, True, True, "original"),
        (1024, 1024, 1024, True, "uint2", 128, True, True, "original"),
        (1, 1024, 1024, True, "uint2", 128, True, True, "rescale"),
    ],
)
def test_correctness_weight_only_dequantize(
    m,
    infeatures,
    outfeatures,
    bias,
    W_dtype,
    group_size,
    with_scaling,
    with_zeros,
    zeros_mode,
):
    import numpy as np
    from bitblas.quantization.utils import general_compress

    linear_bitblas = BitBLASLinear(
        infeatures,
        outfeatures,
        bias=bias,
        A_dtype="float16",
        W_dtype=W_dtype,
        accum_dtype="float16",
        out_dtype="float16",
        group_size=group_size,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        opt_M=m,
    ).cuda()
    if not isinstance(m, int):
        # average m
        m = sum(m) // len(m)
    input_shape = (m, infeatures)
    weight_shape = (outfeatures, infeatures)
    output_shape = (m, outfeatures)
    inputs = []
    inputs.append(torch.rand(input_shape, dtype=torch.float16).cuda() - 0.5)
    source_format, bit = (
        linear_bitblas.bitblas_matmul.source_format,
        linear_bitblas.bitblas_matmul.bit,
    )

    maxq = 2**(bit - 1) - 1
    zeros = maxq
    if source_format == "uint":
        inputs.append(torch.randint(0, maxq, weight_shape, dtype=torch.int8).cuda())
    elif source_format == "int":
        inputs.append(torch.randint(-maxq, maxq, weight_shape, dtype=torch.int8).cuda())
    else:
        raise NotImplementedError

    inputs.append(torch.rand(output_shape, dtype=torch.float16).cuda())

    intweight = inputs[1]
    intweight = intweight.cpu().numpy().astype(np.int8)
    if source_format == "int":
        intweight = intweight + maxq
    if with_zeros:
        inputs[1] = inputs[1] - zeros
    bias_tensor = torch.rand((output_shape[-1],), dtype=torch.float16).cuda()
    ref_result = torch.matmul(inputs[0], (inputs[1].t()).to(torch.float16))
    if bias:
        ref_result = ref_result + bias_tensor

    with torch.no_grad():
        qw_np = general_compress(intweight, source_bits=bit, storage_dtype=np.int8)
        qw_torch = torch.from_numpy(qw_np).cuda()
        permuted_inputs = []
        if linear_bitblas.bitblas_matmul.input_transform is not None:
            permuted_inputs.append(
                linear_bitblas.bitblas_matmul.input_transform(inputs[0].cpu()).cuda())
        else:
            permuted_inputs.append(inputs[0])
        if linear_bitblas.bitblas_matmul.weight_transform is not None:
            permuted_inputs.append(
                linear_bitblas.bitblas_matmul.weight_transform(qw_torch.cpu()).cuda())
        else:
            permuted_inputs.append(qw_torch)
        linear_bitblas.qweight.data = permuted_inputs[-1].clone()
        if with_scaling:
            if group_size == -1:
                group_size = infeatures
            permuted_inputs.append(
                torch.ones([outfeatures, infeatures // group_size], dtype=torch.float16).cuda())
            linear_bitblas.scales.data = permuted_inputs[-1].clone()
        if with_zeros:
            if zeros_mode == "original":
                permuted_inputs.append(
                    torch.ones([outfeatures, infeatures // group_size], dtype=torch.float16).cuda()
                    * zeros)
            elif zeros_mode == "rescale":
                original_zeros = (
                    torch.ones([outfeatures, infeatures // group_size], dtype=torch.float16).cuda()
                    * zeros)
                scaled_zeros = original_zeros * permuted_inputs[-1]
                permuted_inputs.append(scaled_zeros)
            elif zeros_mode == "quantized":
                original_zeros = (
                    torch.ones([infeatures // group_size, outfeatures], dtype=torch.int8).cuda() *
                    zeros)
                qzeros = general_compress(
                    original_zeros.cpu().numpy(), source_bits=bit, storage_dtype=np.int8)
                permuted_inputs.append(torch.from_numpy(qzeros).cuda())
            else:
                raise NotImplementedError
            linear_bitblas.zeros.data = permuted_inputs[-1].clone()
        if bias:
            permuted_inputs.append(bias_tensor)
            linear_bitblas.bias.data = bias_tensor.clone()

    with torch.no_grad():
        output_bitblas = linear_bitblas(inputs[0])
    torch.testing.assert_close(output_bitblas, ref_result, rtol=1e0, atol=1e0)


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


# @pytest.mark.parametrize(
#     "m, infeatures, outfeatures, bias",
#     [
#         (1, 1024, 1024, False),
#         (1024, 1024, 1024, False),
#     ],
# )
# def test_profile_performance(m, infeatures, outfeatures, bias):
#     linear_bitblas = BitBLASLinear(
#         infeatures,
#         outfeatures,
#         bias=bias,
#         A_dtype=torch.float16,
#         opt_M=m,
#         enable_tuning=False,
#     ).cuda()
#     with torch.no_grad():
#         input_data = torch.randn(m, infeatures, dtype=torch.float16).cuda()
#         torch_latency = profile(linear_bitblas, input_data)
#         bitblas_latency = linear_bitblas.bitblas_matmul.profile_latency()
#     print(f"torch_latency: {torch_latency}, bitblas_latency: {bitblas_latency}")
#     assert (abs(torch_latency - bitblas_latency) / torch_latency <
#             0.1), f"torch_latency: {torch_latency}, bitblas_latency: {bitblas_latency}"

if __name__ == "__main__":
    bitblas.testing.main()
