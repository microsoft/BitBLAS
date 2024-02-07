import tvm
import bitblas
from bitblas.ops import Matmul
import numpy as np
import torch


def test_matmul_codegen_static_shape_optimize_s8():
    M = 16384
    N = 16384
    K = 16384

    target = tvm.target.Target("nvidia/nvidia-a100")

    matmul = Matmul(
        M=M,
        N=N,
        K=K,
        a_dtype="int8",
        b_dtype="int8",
        c_dtype="int32",
        propagate_a=False,
        propagate_b=False,
        layout="nt",
        target=target,
    )
    matmul.optimize()
    code = matmul.codegen(target=target)
    assert code


if __name__ == "__main__":
    test_matmul_codegen_static_shape_optimize_s8()