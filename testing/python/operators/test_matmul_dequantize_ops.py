# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pytest
import tvm
import bitblas
from bitblas.utils import get_target_from_env
from bitblas.ops.matmul_dequantize import (
    MatmulWeightOnlyDequantize,
    MatmulWeightOnlyDequantizeConfig,
)
import logging
from bitblas import set_log_level

set_log_level(logging.DEBUG)
target = tvm.target.Target(get_target_from_env())


def get_codegen_result(ops, target):
    code = ops.get_source(target=target)
    return code


# fmt: off
@pytest.mark.parametrize(
    "M,N,K,in_dtype,out_dtype,accum_dtype,bit,storage_dtype,source_format,with_scaling,with_zeros,group_size,fast_decoding,with_bias,propagate_a,propagate_b,layout",
    [
        (1, 1024, 1024, "float16", "float16", "float16", 4, "int8", "uint", False, False, -1, False,
         False, False, False, "nt"),
    ],
)
def test_matmul_dequantize_codegen_default(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    bit,
    storage_dtype,
    source_format,
    with_scaling,
    with_zeros,
    group_size,
    fast_decoding,
    with_bias,
    propagate_a,
    propagate_b,
    layout,
):

    matmul_config = MatmulWeightOnlyDequantizeConfig(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        bit=bit,
        storage_dtype=storage_dtype,
        source_format=source_format,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        group_size=group_size,
        fast_decoding=fast_decoding,
        with_bias=with_bias,
        propagate_a=propagate_a,
        propagate_b=propagate_b,
        layout=layout,
    )
    matmul = MatmulWeightOnlyDequantize(
        config=matmul_config,
        target=target,
    )
    assert get_codegen_result(matmul, target)


@pytest.mark.parametrize(
    "M,N,K,in_dtype,out_dtype,accum_dtype,bit,storage_dtype,source_format,with_scaling,with_zeros,group_size,fast_decoding,with_bias,propagate_a,propagate_b,layout",
    [
        (1, 1024, 1024, "float16", "float16", "float16", 4, "int8", "uint", False, False, -1, False,
         False, False, False, "nt"),
    ],
)
def test_matmul_dequantize_retrieve_weight_shape(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    bit,
    storage_dtype,
    source_format,
    with_scaling,
    with_zeros,
    group_size,
    fast_decoding,
    with_bias,
    propagate_a,
    propagate_b,
    layout,
):

    matmul_config = MatmulWeightOnlyDequantizeConfig(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        bit=bit,
        storage_dtype=storage_dtype,
        source_format=source_format,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        group_size=group_size,
        fast_decoding=fast_decoding,
        with_bias=with_bias,
        propagate_a=propagate_a,
        propagate_b=propagate_b,
        layout=layout,
    )
    matmul = MatmulWeightOnlyDequantize(
        config=matmul_config,
        target=target,
    )
    assert matmul.retrieve_weight_shape()


@pytest.mark.parametrize(
    "M,N,K,in_dtype,out_dtype,accum_dtype,bit,storage_dtype,source_format,with_scaling,with_zeros,group_size,fast_decoding,with_bias,propagate_a,propagate_b,layout",
    [
        (
            1,
            1024,
            1024,
            "float16",
            "float16",
            "float16",
            4,
            "int8",
            "uint",
            False,
            False,
            -1,
            False,
            False,
            False,
            False,
            "nt",
        ),
        (
            1,
            1024,
            1024,
            "float16",
            "float16",
            "float16",
            4,
            "int8",
            "uint",
            False,
            False,
            -1,
            False,
            False,
            False,
            True,
            "nt",
        ),
    ],
)
def test_matmul_dequantize_codegen_finetune(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    bit,
    storage_dtype,
    source_format,
    with_scaling,
    with_zeros,
    group_size,
    fast_decoding,
    with_bias,
    propagate_a,
    propagate_b,
    layout,
):

    matmul_config = MatmulWeightOnlyDequantizeConfig(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        bit=bit,
        storage_dtype=storage_dtype,
        source_format=source_format,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        group_size=group_size,
        fast_decoding=fast_decoding,
        with_bias=with_bias,
        propagate_a=propagate_a,
        propagate_b=propagate_b,
        layout=layout,
    )
    matmul = MatmulWeightOnlyDequantize(
        config=matmul_config,
        target=target,
    )
    matmul.hardware_aware_finetune(topk=20)
    assert get_codegen_result(matmul, target)


@pytest.mark.parametrize(
    "M,N,K,in_dtype,out_dtype,accum_dtype,bit,storage_dtype,source_format,with_scaling,with_zeros,group_size,fast_decoding,with_bias,propagate_a,propagate_b,layout",
    [
        (
            1,
            1024,
            1024,
            "float16",
            "float16",
            "float16",
            4,
            "int8",
            "uint",
            False,
            False,
            -1,
            False,
            False,
            False,
            False,
            "nt",
        ),
        (
            1,
            1024,
            1024,
            "float16",
            "float16",
            "float16",
            4,
            "int8",
            "af",
            False,
            False,
            -1,
            False,
            False,
            False,
            False,
            "nt",
        ),
        (
            1024,
            1024,
            1024,
            "float16",
            "float16",
            "float16",
            4,
            "int8",
            "af",
            False,
            False,
            -1,
            False,
            False,
            False,
            False,
            "nt",
        ),
        (
            1024,
            1024,
            1024,
            "float16",
            "float16",
            "float16",
            4,
            "int8",
            "af",
            False,
            False,
            -1,
            False,
            False,
            False,
            True,
            "nt",
        ),
        (
            1024,
            1024,
            1024,
            "float16",
            "float16",
            "float16",
            4,
            "int8",
            "af",
            False,
            False,
            -1,
            False,
            False,
            True,
            True,
            "nt",
        ),
        (
            1024,
            1024,
            1024,
            "float16",
            "float16",
            "float16",
            4,
            "int8",
            "af",
            True,
            False,
            -1,
            False,
            False,
            True,
            True,
            "nt",
        ),
        (
            1024,
            1024,
            1024,
            "float16",
            "float16",
            "float16",
            4,
            "int8",
            "af",
            True,
            False,
            128,
            False,
            False,
            True,
            True,
            "nt",
        ),
    ],
)
def test_matmul_dequantize_profile_latency(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    bit,
    storage_dtype,
    source_format,
    with_scaling,
    with_zeros,
    group_size,
    fast_decoding,
    with_bias,
    propagate_a,
    propagate_b,
    layout,
):

    matmul_config = MatmulWeightOnlyDequantizeConfig(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        bit=bit,
        storage_dtype=storage_dtype,
        source_format=source_format,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        group_size=group_size,
        fast_decoding=fast_decoding,
        with_bias=with_bias,
        propagate_a=propagate_a,
        propagate_b=propagate_b,
        layout=layout,
    )
    matmul = MatmulWeightOnlyDequantize(
        config=matmul_config,
        target=target,
    )
    matmul.hardware_aware_finetune(topk=20)
    latency = matmul.profile_latency()
    assert latency


@pytest.mark.parametrize(
    "M,N,K,in_dtype,out_dtype,accum_dtype,bit,storage_dtype,source_format,with_scaling,with_zeros,group_size,fast_decoding,with_bias,propagate_a,propagate_b,layout,zeros_type",
    [
        (1, 1024, 1024, "float16", "float16", "float16", 4, "int8", "uint", False, False, -1, False,
         False, False, False, "nt", "rescale"),
        (1, 1024, 1024, "float16", "float16", "float16", 4, "int8", "uint", False, False, -1, True,
         False, False, False, "nt", "rescale"),
        (1, 1024, 1024, "float16", "float16", "float16", 4, "int8", "int", False, False, -1, False,
         False, False, False, "nt", "rescale"),
        (1, 1024, 1024, "float16", "float16", "float16", 4, "int8", "int", False, False, -1, True,
         False, False, False, "nt", "rescale"),
        (1, 1024, 1024, "float16", "float16", "float16", 2, "int8", "int", False, False, -1, True,
         False, False, False, "nt", "rescale"),
        (1, 1024, 1024, "float16", "float16", "float16", 2, "int8", "int", True, False, -1, True,
         False, False, False, "nt", "rescale"),
        (1, 1024, 1024, "float16", "float16", "float16", 2, "int8", "int", True, False, 128, True,
         False, False, False, "nt", "rescale"),
        (1, 1024, 1024, "float16", "float16", "float16", 2, "int8", "uint", True, True, 128, False,
         False, False, False, "nt", "rescale"),
        (1, 1024, 4096, "float16", "float16", "float16", 2, "int8", "uint", True, True, 128, True,
         False, False, False, "nt", "rescale"),
        (1024, 1024, 1024, "float16", "float16", "float16", 2, "int8", "int", True, False, 128,
         False, False, False, False, "nt", "rescale"),
        (1024, 1024, 1024, "float16", "float16", "float16", 2, "int8", "int", True, False, 128,
         False, False, False, True, "nt", "rescale"),
        (1024, 1024, 1024, "float16", "float16", "float16", 2, "int8", "int", True, False, 128,
         False, False, True, True, "nt", "rescale"),
        (1024, 1024, 1024, "float16", "float16", "float16", 2, "int8", "int", True, False, 128,
         False, False, True, True, "nt", "original"),
        ([1, 1024], 1024, 1024, "float16", "float16", "float16", 4, "int8", "uint", False, False,
         -1, False, False, False, False, "nt", "original"),
    ],
)
def test_matmul_dequantize_torch_forward(M, N, K, in_dtype, out_dtype, accum_dtype, bit,
                                         storage_dtype, source_format, with_scaling, with_zeros,
                                         group_size, fast_decoding, with_bias, propagate_a,
                                         propagate_b, layout, zeros_type):
    import torch
    torch.random.manual_seed(0)
    import numpy as np
    from bitblas.quantization.utils import general_compress

    matmul_config = MatmulWeightOnlyDequantizeConfig(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        bit=bit,
        storage_dtype=storage_dtype,
        source_format=source_format,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        group_size=group_size,
        fast_decoding=fast_decoding,
        with_bias=with_bias,
        propagate_a=propagate_a,
        propagate_b=propagate_b,
        layout=layout,
        zeros_type=zeros_type)
    matmul = MatmulWeightOnlyDequantize(
        config=matmul_config,
        target=target,
    )
    if not isinstance(M, int):
        M = 32
    matmul.hardware_aware_finetune(topk=20)
    input_shape = (M, K)
    weight_shape = (N, K) if layout == "nt" else (K, N)
    output_shape = (M, N)
    inputs = []
    inputs.append(torch.rand(input_shape, dtype=torch.float16).cuda() - 0.5)
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
    bias = torch.rand((output_shape[-1],), dtype=torch.float16).cuda()
    ref_result = torch.matmul(inputs[0],
                              (inputs[1].t() if layout == "nt" else inputs[1]).to(torch.float16))
    if with_bias:
        ref_result = ref_result + bias
    qw_np = general_compress(intweight, source_bits=bit, storage_dtype=np.int8)
    qw_torch = torch.from_numpy(qw_np).cuda()
    permuted_inputs = []
    if matmul.input_transform is not None:
        permuted_inputs.append(matmul.input_transform(qw_torch.cpu()).cuda())
    else:
        permuted_inputs.append(inputs[0])
    if matmul.weight_transform is not None:
        permuted_inputs.append(matmul.weight_transform(qw_torch.cpu()).cuda())
    else:
        permuted_inputs.append(qw_torch)
    if with_scaling:
        if group_size == -1:
            group_size = K
        permuted_inputs.append(torch.ones([N, K // group_size], dtype=torch.float16).cuda())
    if with_zeros:
        permuted_inputs.append(torch.ones([N, K // group_size], dtype=torch.float16).cuda() * zeros)
    if with_bias:
        permuted_inputs.append(bias)
    permuted_inputs.append(inputs[2])
    matmul(*permuted_inputs)
    print(permuted_inputs[-1])
    print(ref_result)
    torch.testing.assert_close(permuted_inputs[-1], ref_result, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "M,N,K,in_dtype,out_dtype,accum_dtype,bit,storage_dtype,source_format,with_scaling,with_zeros,group_size,fast_decoding,with_bias,layout,zeros_type",
    [
        (16, 768, 768, "float16", "float16", "float16", 4, "int8", "uint", False, False, -1, True,
         False, "nt", "original"),
        (16, 768, 768, "float16", "float16", "float16", 4, "int8", "uint", False, True, -1, True,
         True, "nt", "original"),
        (16, 3072, 768, "float16", "float16", "float16", 4, "int8", "uint", False, False, -1, True,
         False, "nt", "original"),
        (16, 768, 3072, "float16", "float16", "float16", 4, "int8", "uint", False, False, -1, True,
         False, "nt", "original"),
    ],
)
def test_matmul_dequantize_propgate_comparison(M, N, K, in_dtype, out_dtype, accum_dtype, bit,
                                               storage_dtype, source_format, with_scaling,
                                               with_zeros, group_size, fast_decoding, with_bias,
                                               layout, zeros_type):
    import torch
    torch.random.manual_seed(0)
    original_matmul_config = MatmulWeightOnlyDequantizeConfig(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        bit=bit,
        storage_dtype=storage_dtype,
        source_format=source_format,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        group_size=group_size,
        fast_decoding=False,
        with_bias=with_bias,
        propagate_a=False,
        propagate_b=False,
        layout=layout,
        zeros_type=zeros_type)
    original_matmul = MatmulWeightOnlyDequantize(
        config=original_matmul_config,
        target=target,
    )
    if not isinstance(M, int):
        M = 32

    if group_size == -1:
        group_size = K
    input_shape = (M, K)
    weight_shape = (N, K // 2) if layout == "nt" else (K, N)
    output_shape = (M, N)
    scales_shape = (N, K // group_size)
    zeros_shape = (N, K // group_size)
    bias_shape = (N,)

    inputs = []
    input_tensor = torch.rand(input_shape, dtype=torch.float16).cuda()
    weight_tensor = torch.randint(0, 2**(bit - 1) - 1, weight_shape, dtype=torch.int8).cuda()
    scales_tensor = torch.rand(scales_shape, dtype=torch.float16).cuda()
    zeros_tensor = torch.rand(zeros_shape, dtype=torch.float16).cuda()
    bias_tensor = torch.rand(bias_shape, dtype=torch.float16).cuda()
    output_tensor = torch.zeros(output_shape, dtype=torch.float16).cuda()
    inputs.append(input_tensor)
    inputs.append(weight_tensor)
    if with_scaling:
        inputs.append(scales_tensor)
    if with_zeros:
        inputs.append(zeros_tensor)
    if with_bias:
        inputs.append(bias_tensor)
    inputs.append(output_tensor)
    ref_result = original_matmul(*inputs)

    propagated_matmul_config = MatmulWeightOnlyDequantizeConfig(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        bit=bit,
        storage_dtype=storage_dtype,
        source_format=source_format,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        group_size=group_size,
        fast_decoding=fast_decoding,
        with_bias=with_bias,
        propagate_a=False,
        propagate_b=True,
        layout=layout,
        zeros_type=zeros_type)
    propagated_matmul = MatmulWeightOnlyDequantize(
        config=propagated_matmul_config,
        target=target,
    )

    propagated_matmul.hardware_aware_finetune(topk=20)
    propagated_inputs = []
    propagated_inputs.append(input_tensor)
    if propagated_matmul.weight_transform is not None:
        propagated_inputs.append(propagated_matmul.weight_transform(weight_tensor.cpu()).cuda())
    else:
        propagated_inputs.append(weight_tensor)
    if with_scaling:
        propagated_inputs.append(scales_tensor)
    if with_zeros:
        propagated_inputs.append(zeros_tensor)
    if with_bias:
        propagated_inputs.append(bias_tensor)
    propagated_inputs.append(torch.zeros(output_shape, dtype=torch.float16).cuda())

    propagated_result = propagated_matmul(*propagated_inputs)
    torch.testing.assert_close(ref_result, propagated_result, rtol=1e-2, atol=1e-2)


# fmt: on

if __name__ == "__main__":
    bitblas.testing.main()
