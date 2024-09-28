# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas import tvm as tvm
import bitblas.testing
from tvm import tl
from tvm.ir import structural_equal
from bitblas.ops.general_matmul.tilelang.dense.matmul import (
    MatmulScheduler,
)

def test_scheduler_simplify(M,
                            N,
                            K,
                            trans_A=False,
                            trans_B=True,
                            dtypeAB="float16",
                            dtypeC="float16",
                            accum_dtype="float16"):
    matmul = MatmulScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        dtypeAB=dtypeAB,
        dtypeC=dtypeC,
        accum_dtype=accum_dtype,
    ).disable_simplify().with_default_config()
    
    simplified = MatmulScheduler.Simplify(matmul)

    is_equal = structural_equal(matmul, simplified)
    
    assert is_equal == False, "Simplify should not return the same schedule"

if __name__ == "__main__":
    bitblas.testing.main()
