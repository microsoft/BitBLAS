# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas import tvm as tvm
import bitblas.testing
from tvm.ir import structural_equal
from bitblas.ops.general_matmul.tilelang.dense.matmul_tensorcore import (
    MatmulScheduler,)


def assert_scheduler_simplify(M,
                              N,
                              K,
                              trans_A=False,
                              trans_B=True,
                              in_dtype="float16",
                              out_dtype="float16",
                              accum_dtype="float16"):
    matmul = MatmulScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
    ).deactivate_simplify().with_default_config()

    simplified = MatmulScheduler.Simplify(matmul)

    is_equal = structural_equal(matmul, simplified)
    assert is_equal is False, "Simplify should not return the same schedule"


def test_scheduler_simplify():
    assert_scheduler_simplify(128, 128, 128)


if __name__ == "__main__":
    bitblas.testing.main()
