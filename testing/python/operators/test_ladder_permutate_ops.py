# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pytest
import tvm
import bitblas
from bitblas.ops.ladder_permutate import LadderPermutate, LadderPermutateConfig

target = tvm.target.Target("llvm")

# fmt: off
@pytest.mark.parametrize("M,N,datatype,dequantize_bits,propagate_kind,transpose_matrix,transform_kind,target_instruction", [
    (16384, 16384, "float16", -1, "B", False, 0, "nvidia-mma"),
])
def test_ladder_permutate_profile_latency(
    M,
    N,
    datatype,
    dequantize_bits,
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
    print(latency)
    assert latency

# fmt: on

if __name__ == "__main__":
    # bitblas.testing.main()
    test_ladder_permutate_profile_latency(
        16384, 16384, "float16", -1, "B", False, 0, "nvidia-mma"
    )
