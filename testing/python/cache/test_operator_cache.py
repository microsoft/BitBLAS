# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pytest
import os
import tvm
import torch
import bitblas
from bitblas.ops.matmul import Matmul, MatmulConfig
from bitblas.ops.matmul_dequantize import (
    MatmulWeightOnlyDequantize,
    MatmulWeightOnlyDequantizeConfig,
)
from bitblas.cache import global_operator_cache

target = bitblas.utils.get_target_from_env()


def get_codegen_result(ops, target):
    code = ops.get_source(target=target)
    return code


# fmt: off
@pytest.mark.parametrize(
    "M,N,K,in_dtype,out_dtype,accum_dtype,with_bias,propagate_a,propagate_b,layout,enable_tuning",
    [
        (1, 16384, 16384, "float16", "float16", "float16", False, False, False, "nt", False),
        # dynamic shape
        ([1], 16384, 16384, "float16", "float16", "float16", False, False, False, "nt", False),
        ([1, 32], 16384, 16384, "float16", "float16", "float16", False, False, False, "nt", True),
    ],
)
def test_config_hashable(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    with_bias,
    propagate_a,
    propagate_b,
    layout,
    enable_tuning,
):

    matmul_config = MatmulConfig(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        with_bias=with_bias,
        propagate_a=propagate_a,
        propagate_b=propagate_b,
        layout=layout,
    )
    matmul = Matmul(
        config=matmul_config,
        target=target,
    )
    if enable_tuning:
        matmul.hardware_aware_finetune(topk=20)

    BITBLAS_TUNING_CACHE = {}
    success = False
    try:
        BITBLAS_TUNING_CACHE[matmul.config] = matmul
        success = True
    except Exception as hash_error:
        print(hash_error)
    assert success

@pytest.mark.parametrize(
    "M,N,K,in_dtype,out_dtype,accum_dtype,with_bias,propagate_a,propagate_b,layout,enable_tuning",
    [
        (1, 16384, 16384, "float16", "float16", "float16", False, False, False, "nt", False),
        # dynamic shape
        ([1], 16384, 16384, "float16", "float16", "float16", False, False, False, "nt", False),
        ([1, 32], 16384, 16384, "float16", "float16", "float16", False, False, False, "nt", True),
    ],
)
def test_global_cache_inquery(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    with_bias,
    propagate_a,
    propagate_b,
    layout,
    enable_tuning,
):

    matmul_config = MatmulConfig(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        with_bias=with_bias,
        propagate_a=propagate_a,
        propagate_b=propagate_b,
        layout=layout,
    )
    matmul = Matmul(
        config=matmul_config,
        target=target,
    )
    if enable_tuning:
        matmul.hardware_aware_finetune(topk=20)
    success = False
    try:
        global_operator_cache.add(matmul.config, matmul)
        success = True
    except Exception as hash_error:
        print(hash_error)
    assert success
    
    matmul = global_operator_cache.get(matmul.config)
    assert matmul is not None

@pytest.mark.parametrize(
    "M,N,K,in_dtype,out_dtype,accum_dtype,with_bias,propagate_a,propagate_b,layout,enable_tuning",
    [
        (1, 16384, 16384, "float16", "float16", "float16", False, False, False, "nt", False),
        # dynamic shape
        ([1], 16384, 16384, "float16", "float16", "float16", False, False, False, "nt", False),
        ([1, 32], 16384, 16384, "float16", "float16", "float16", False, False, False, "nt", True),
    ],
)
def test_global_cache_inquery_torch_forward(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    with_bias,
    propagate_a,
    propagate_b,
    layout,
    enable_tuning,
):

    matmul_config = MatmulConfig(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        with_bias=with_bias,
        propagate_a=propagate_a,
        propagate_b=propagate_b,
        layout=layout,
    )
    matmul = Matmul(
        config=matmul_config,
        target=target,
    )
    if enable_tuning:
        matmul.hardware_aware_finetune(topk=20)
    success = False
    try:
        global_operator_cache.add(matmul.config, matmul)
        success = True
    except Exception as hash_error:
        print(hash_error)
    assert success
    
    matmul = global_operator_cache.get(matmul.config)
    assert matmul is not None
    if not isinstance(M, int):
        M = 32
    # convert tensors to torch
    input_shape = (M, K)
    weight_shape = (N, K) if layout == "nt" else (K, N)
    output_shape = (M, N)
    inputs = []
    inputs.append(torch.rand(input_shape, dtype=torch.float16).cuda())
    inputs.append(torch.rand(weight_shape, dtype=torch.float16).cuda())
    inputs.append(torch.rand(output_shape, dtype=torch.float16).cuda())
    ref_result = torch.matmul(inputs[0], inputs[1].t() if layout == "nt" else inputs[1])

    permuted_inputs = []
    if matmul.input_transform is not None:
        permuted_inputs.append(
            matmul.input_transform(inputs[0].cpu())
        ).cuda()
    else:
        permuted_inputs.append(inputs[0])
    if matmul.weight_transform is not None:
        permuted_inputs.append(
            matmul.weight_transform(inputs[1].cpu()).cuda()
        )
    else:
        permuted_inputs.append(inputs[1])
    permuted_inputs.append(inputs[2])
    matmul(*permuted_inputs)
    torch.testing.assert_close(permuted_inputs[-1], ref_result, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "M,N,K,in_dtype,out_dtype,accum_dtype,with_bias,propagate_a,propagate_b,layout,enable_tuning",
    [
        (1, 16384, 16384, "float16", "float16", "float16", False, False, False, "nt", False),
    ],
)
def test_global_cache_save_to_database(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    with_bias,
    propagate_a,
    propagate_b,
    layout,
    enable_tuning,
):

    matmul_config = MatmulConfig(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        with_bias=with_bias,
        propagate_a=propagate_a,
        propagate_b=propagate_b,
        layout=layout,
    )
    matmul = Matmul(
        config=matmul_config,
        target=target,
    )
    if enable_tuning:
        matmul.hardware_aware_finetune(topk=20)
    success = False
    try:
        global_operator_cache.add(matmul.config, matmul)
        success = True
    except Exception as hash_error:
        print(hash_error)
    assert success
    
    database_path = "debug/test_database"
    global_operator_cache.save_into_database(database_path, target=target)
    assert os.path.exists(database_path)
    global_operator_cache.clear()
    assert global_operator_cache.size() == 0
    global_operator_cache.load_from_database(database_path, target=target)
    assert global_operator_cache.size() > 0
    
    matmul = global_operator_cache.get(matmul.config)
    assert matmul is not None
    if not isinstance(M, int):
        M = 32
    # convert tensors to torch
    input_shape = (M, K)
    weight_shape = (N, K) if layout == "nt" else (K, N)
    output_shape = (M, N)
    inputs = []
    inputs.append(torch.rand(input_shape, dtype=torch.float16).cuda())
    inputs.append(torch.rand(weight_shape, dtype=torch.float16).cuda())
    inputs.append(torch.rand(output_shape, dtype=torch.float16).cuda())
    ref_result = torch.matmul(inputs[0], inputs[1].t() if layout == "nt" else inputs[1])

    permuted_inputs = []
    if matmul.input_transform is not None:
        permuted_inputs.append(
            matmul.input_transform(inputs[0].cpu())
        ).cuda()
    else:
        permuted_inputs.append(inputs[0])
    if matmul.weight_transform is not None:
        permuted_inputs.append(
            matmul.weight_transform(inputs[1].cpu()).cuda()
        )
    else:
        permuted_inputs.append(inputs[1])
    permuted_inputs.append(inputs[2])
    matmul(*permuted_inputs)
    torch.testing.assert_close(permuted_inputs[-1], ref_result, rtol=1e-2, atol=1e-2)

@pytest.mark.parametrize(
    "M,N,K,in_dtype,out_dtype,accum_dtype,bit,storage_dtype,source_format,with_scaling,with_zeros,group_size,fast_decoding,with_bias,propagate_a,propagate_b,layout",
    [
        (1, 1024, 1024, "float16", "float16", "float16", 4, "int8", "uint", False, False, -1, False, False, False, False, "nt",),
        (1, 1024, 1024, "float16", "float16", "float16", 4, "int8", "af", False, False, -1, False, False, False, False, "nt",),
        (1024, 1024, 1024, "float16", "float16", "float16", 4, "int8", "af", False, False, -1, False, False, False, False, "nt",),
        (1024, 1024, 1024, "float16", "float16", "float16", 4, "int8", "af", False, False, -1, False, False, False, True, "nt",),
        (1024, 1024, 1024, "float16", "float16", "float16", 4, "int8", "af", False, False, -1, False, False, True, True, "nt",),
        (1024, 1024, 1024, "float16", "float16", "float16", 4, "int8", "af", True, False, -1, False, False, True, True, "nt",),
        (1024, 1024, 1024, "float16", "float16", "float16", 4, "int8", "af", True, False, 128, False, False, True, True, "nt",),
    ],
)
def test_matmul_dequantize_save_into_database(
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
    database_path = "debug/test_database"
    success = False
    
    try:
        global_operator_cache.add(matmul.config, matmul)
        success = True
    except Exception as hash_error:
        print(hash_error)
    assert success
    global_operator_cache.save_into_database(database_path, target=target)
    assert os.path.exists(database_path)
    global_operator_cache.clear()
    assert global_operator_cache.size() == 0
    global_operator_cache.load_from_database(database_path, target=target)
    assert global_operator_cache.size() > 0


# fmt: on
if __name__ == "__main__":
    bitblas.testing.main()
