# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pytest
import bitblas
import bitblas.testing
from bitblas.ops.lop3_permutate import LOP3Permutate, LOP3PermutateConfig

from bitblas import tvm

target = tvm.target.Target("llvm")


# fmt: off
@pytest.mark.parametrize("M,N,datatype,dequantize_bits,storage_dtype", [
    (1024, 1024, "float16", 4, "uint32"),
])
def test_lop3_permutate_profile_latency(
    M,
    N,
    datatype,
    dequantize_bits,
    storage_dtype
):

    lop3_permutate_config = LOP3PermutateConfig(
        M=M,
        N=N,
        datatype=datatype,
        dequantize_bits=dequantize_bits,
        storage_dtype=storage_dtype
    )
    lop3_permutate = LOP3Permutate(
        config=lop3_permutate_config,
        target=target,
    )
    latency = lop3_permutate.profile_latency()
    assert latency
# fmt: on

if __name__ == "__main__":
    bitblas.testing.main()
