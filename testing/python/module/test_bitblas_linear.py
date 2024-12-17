# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import bitblas
import bitblas.testing
from bitblas import Linear as BitBLASLinear
import torch
import time
import numpy as np
import torch.nn as nn

torch.manual_seed(0)
bitblas.set_log_level("DEBUG")


def correctness_consistent(m, in_features, out_features, bias):
    linear_torch = (nn.Linear(in_features, out_features, bias=bias).to(torch.float16).cuda())
    linear_bitblas = BitBLASLinear(
        in_features,
        out_features,
        bias=bias,
        A_dtype="float16",
        W_dtype="float16",
        accum_dtype="float16",
        out_dtype="float16",
        opt_M=m,
    ).cuda()

    with torch.no_grad():
        linear_bitblas.load_and_transform_weight(linear_torch.weight.clone())
        if bias:
            linear_bitblas.bias = nn.Parameter(linear_torch.bias.clone())

    with torch.no_grad():
        if not isinstance(m, int):
            # When m is a list, average m
            m = sum(m) // len(m)
        input_data = torch.randn(m, in_features, dtype=torch.float16).cuda()
        output_torch = linear_torch(input_data)
        output_bitblas = linear_bitblas(input_data)
    print(output_torch)
    print(output_bitblas)
    bitblas.testing.torch_assert_close(output_torch, output_bitblas, rtol=1e-1, atol=1e-2)


def test_correctness_consistent():
    correctness_consistent(1, 1024, 1024, False)
    correctness_consistent(1, 1024, 1024, True)
    correctness_consistent(1024, 1024, 1024, True)
    correctness_consistent([1, 1024], 1024, 1024, True)


def correctness_weight_only_dequantize(
    m,
    in_features,
    out_features,
    bias,
    W_dtype,
    group_size,
    with_scaling,
    with_zeros,
    zeros_mode,
):
    import numpy as np
    from bitblas.quantization.utils import general_compress
    from bitblas.cache import global_operator_cache

    global_operator_cache.clear()
    linear_bitblas = BitBLASLinear(
        in_features,
        out_features,
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
    input_shape = (m, in_features)
    weight_shape = (out_features, in_features)
    output_shape = (m, out_features)
    inputs = []
    inputs.append(torch.rand(input_shape, dtype=torch.float16).cuda() - 0.5)
    source_format, bit = (
        linear_bitblas.bitblas_matmul.source_format,
        linear_bitblas.bitblas_matmul.bit,
    )

    maxq = 2**(bit - 1)
    zeros = maxq
    if source_format == "uint":
        inputs.append(torch.randint(0, maxq, weight_shape, dtype=torch.int8).cuda())
    elif source_format == "int":
        inputs.append(torch.randint(-maxq, maxq, weight_shape, dtype=torch.int8).cuda())
    else:
        raise NotImplementedError

    inputs.append(torch.rand(output_shape, dtype=torch.float16).cuda())

    intweight = inputs[1]
    intweight = intweight.cpu().to(torch.int8)
    if source_format == "int":
        intweight = intweight + maxq
    if with_zeros:
        inputs[1] = inputs[1] - zeros
    bias_tensor = torch.rand((output_shape[-1],), dtype=torch.float16).cuda()
    ref_result = torch.matmul(inputs[0], (inputs[1].t()).to(torch.float16))
    if bias:
        ref_result = ref_result + bias_tensor

    with torch.no_grad():
        permuted_inputs = []
        permuted_inputs.append(inputs[0])
        if linear_bitblas.bitblas_matmul.weight_transform is not None:
            permuted_inputs.append(
                linear_bitblas.bitblas_matmul.weight_transform(intweight.cpu()).cuda())
        else:
            permuted_inputs.append(inputs[1])
        linear_bitblas.qweight.data = permuted_inputs[-1].clone()
        if with_scaling:
            if group_size == -1:
                group_size = in_features
            permuted_inputs.append(
                torch.ones([out_features, in_features // group_size], dtype=torch.float16).cuda())
            linear_bitblas.scales.data = permuted_inputs[-1].clone()
        if with_zeros:
            if zeros_mode == "original":
                permuted_inputs.append(
                    torch.ones([out_features, in_features // group_size],
                               dtype=torch.float16).cuda() * zeros)
            elif zeros_mode == "rescale":
                original_zeros = (
                    torch.ones([out_features, in_features // group_size],
                               dtype=torch.float16).cuda() * zeros)
                scaled_zeros = original_zeros * permuted_inputs[-1]
                permuted_inputs.append(scaled_zeros)
            elif zeros_mode == "quantized":
                original_zeros = (
                    torch.ones([in_features // group_size, out_features], dtype=torch.int8).cuda() *
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

    rtol = 1e0
    atol = 1e0
    if zeros_mode == "original":
        rtol = 1e2
        atol = 1e2
    print(output_bitblas)
    print(ref_result)
    torch.testing.assert_close(output_bitblas, ref_result, rtol=rtol, atol=atol)


def test_correctness_weight_only_dequantize():
    correctness_weight_only_dequantize(1, 1024, 1024, False, "uint4", -1, False, False, None)
    correctness_weight_only_dequantize(1, 1024, 1024, False, "uint4", -1, False, False, None)
    correctness_weight_only_dequantize(1024, 1024, 1024, True, "uint4", -1, False, False, None)
    correctness_weight_only_dequantize(1, 1024, 1024, True, "uint2", -1, True, False, None)
    correctness_weight_only_dequantize(1, 1024, 1024, True, "uint2", 128, True, True, "original")
    correctness_weight_only_dequantize(1024, 1024, 1024, True, "uint2", 128, True, True, "original")
    correctness_weight_only_dequantize(1, 1024, 1024, True, "uint2", 128, True, True, "rescale")


def profile(model, input_data):
    model = model.cuda()
    model.eval()

    def get_runtime(num_repeats=1):
        tic = time.time()
        for _ in range(num_repeats):
            _ = model(input_data)
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


if __name__ == "__main__":
    bitblas.testing.main()
