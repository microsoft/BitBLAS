# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pytest
import tvm
import bitblas
from bitblas.ops.ladder_permutate import LadderPermutate, LadderPermutateConfig

target = tvm.target.Target("llvm")


# fmt: off
@pytest.mark.parametrize(
    "M,N,datatype,dequantize_bits,storage_dtype,propagate_kind,transpose_matrix,transform_kind,target_instruction",
    [
        (1024, 1024, "float16", -1, "float16", "B", True, 0, "nvidia-mma"),
        (1024, 1024, "float16", -1, "float16", "B", True, 1, "nvidia-mma"),
        (1024, 1024, "float16", -1, "float16", "B", True, 2, "nvidia-mma"),
        # dequantize propagation
        (1024, 1024, "float16", 4, "uint32", "B", True, 2, "nvidia-mma"),
    ])
def test_ladder_permutate_profile_latency(
    M,
    N,
    datatype,
    dequantize_bits,
    storage_dtype,
    propagate_kind,
    transpose_matrix,
    transform_kind,
    target_instruction,
):

    ladder_permutate_config = LadderPermutateConfig(
        M=M,
        N=N,
        datatype=datatype,
        dequantize_bits=dequantize_bits,
        storage_dtype=storage_dtype,
        propagate_kind=propagate_kind,
        transpose_matrix=transpose_matrix,
        transform_kind=transform_kind,
        target_instruction=target_instruction,
    )
    ladder_permutate = LadderPermutate(
        config=ladder_permutate_config,
        target=target,
    )
    latency = ladder_permutate.profile_latency()
    assert latency


@pytest.mark.parametrize(
    "M,N,datatype,dequantize_bits,storage_dtype,propagate_kind,transpose_matrix,transform_kind,target_instruction",
    [
        (1024, 1024, "float16", -1, "float16", "A", True, 0, "nvidia-mma"),
        (1024, 1024, "float16", -1, "float16", "A", True, 1, "nvidia-mma"),
        (1024, 1024, "float16", -1, "float16", "A", True, 2, "nvidia-mma"),
        # dequantize propagation
        (1024, 1024, "float16", 4, "uint32", "A", True, 2, "nvidia-mma"),
    ])
def test_ladder_permutate_profile_latency_cuda(
    M,
    N,
    datatype,
    dequantize_bits,
    storage_dtype,
    propagate_kind,
    transpose_matrix,
    transform_kind,
    target_instruction,
):

    ladder_permutate_config = LadderPermutateConfig(
        M=M,
        N=N,
        datatype=datatype,
        dequantize_bits=dequantize_bits,
        storage_dtype=storage_dtype,
        propagate_kind=propagate_kind,
        transpose_matrix=transpose_matrix,
        transform_kind=transform_kind,
        target_instruction=target_instruction,
    )
    ladder_permutate = LadderPermutate(
        config=ladder_permutate_config,
        target="cuda",
    )
    # ladder_permutate.hardware_aware_finetune()
    latency = ladder_permutate.profile_latency()
    print(latency)
    assert latency


# fmt: on

if __name__ == "__main__":
    bitblas.testing.main()
