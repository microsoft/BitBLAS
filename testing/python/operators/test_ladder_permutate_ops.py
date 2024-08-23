# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import bitblas
from bitblas.ops.ladder_permutate import LadderPermutate, LadderPermutateConfig
from bitblas import tvm

target = tvm.target.Target("llvm")


# fmt: off
def ladder_permutate_profile_latency(
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


def test_ladder_permutate_profile_latency():
    ladder_permutate_profile_latency(1024, 1024, "float16", -1, "float16", "B", True, 1,
                                     "nvidia-mma")
    ladder_permutate_profile_latency(1024, 1024, "float16", -1, "float16", "B", True, 2,
                                     "nvidia-mma")
    ladder_permutate_profile_latency(1024, 1024, "float16", -1, "float16", "B", True, 3,
                                     "nvidia-mma")
    ladder_permutate_profile_latency(1024, 1024, "float16", 4, "int8", "B", True, 2, "nvidia-mma")
    ladder_permutate_profile_latency(1024, 1024, "float16", 4, "int8", "B", True, 3, "nvidia-mma")


def ladder_permutate_profile_latency_cuda(
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
    latency = ladder_permutate.profile_latency()
    assert latency


def test_ladder_permutate_profile_latency_cuda():
    ladder_permutate_profile_latency_cuda(1024, 1024, "float16", -1, "float16", "A", True, 1,
                                          "nvidia-mma")
    ladder_permutate_profile_latency_cuda(1024, 1024, "float16", -1, "float16", "A", True, 2,
                                          "nvidia-mma")
    ladder_permutate_profile_latency_cuda(1024, 1024, "float16", 4, "int8", "A", True, 2,
                                          "nvidia-mma")


# fmt: on

if __name__ == "__main__":
    bitblas.testing.main()
