# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pytest
import bitblas
from bitblas.ops.param_permutate import ParamPermutate, ParamPermutateConfig

from bitblas import tvm

target = tvm.target.Target("llvm")


# fmt: off
@pytest.mark.parametrize(
    "M,N,datatype,transpose_matrix,group_size,propagate_kind,target_instruction", [
        (1024, 1024, "float16", True, 1, True, "nvidia-mma"),
    ])
def test_param_permutate_profile_latency(
    M,
    N,
    datatype,
    transpose_matrix,
    group_size,
    propagate_kind,
    target_instruction,
):
    param_permutate_config = ParamPermutateConfig(
        M=M,
        N=N,
        datatype=datatype,
        propagate_kind=propagate_kind,
        group_size=group_size,
        transpose_matrix=transpose_matrix,
        target_instruction=target_instruction,
    )
    param_permutate = ParamPermutate(
        config=param_permutate_config,
        target=target,
    )
    latency = param_permutate.profile_latency()
    assert latency


# fmt: on

if __name__ == "__main__":
    bitblas.testing.main()
