# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tvm
import bitblas
from bitblas.ops import MatmulConfig, Matmul
import numpy as np


def test_matmul_codegen_static_shape_optimize_s8():
    M = 16384
    N = 16384
    K = 16384

    target = tvm.target.Target("nvidia/nvidia-a100")

    matmul_config = MatmulConfig(
        M=M,
        N=N,
        K=K,
        in_dtype="int8",
        out_dtype="int8",
        accum_dtype="int32",
        propagate_a=False,
        propagate_b=False,
        layout="nt",
    )
    matmul = Matmul(matmul_config, target=target)

    matmul.hardware_aware_finetune()
    code = matmul.codegen(target=target)
    latency = matmul.profile_latency()
    print(latency)
    assert code


if __name__ == "__main__":
    test_matmul_codegen_static_shape_optimize_s8()
