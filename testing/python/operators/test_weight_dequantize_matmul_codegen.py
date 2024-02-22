import tvm
import bitblas
from bitblas.ops import Matmul, MatmulWeightOnlyDequantize
import numpy as np


def test_weight_only_matmul_codegen_static_shape_optimize():
    M = 16384
    N = 16384
    K = 16384

    target = tvm.target.Target("nvidia/nvidia-a100")

    matmul = MatmulWeightOnlyDequantize(
        M=M,
        N=N,
        K=K,
        in_dtype="float16",
        out_dtype="float16",
        accum_dtype="float16",
        propagate_b=True,
        bit=4,
        storage_dtype="int8",
        source_format="uint",
        with_scaling=False,
        group_size=-1,
        fast_decoding=True,
        with_bias=False,
        target=target,
    )
    matmul.optimize(topk=20)
    code = matmul.codegen(target=target)
    latency = matmul.profile_latency()
    print(latency)
    assert code


def test_weight_only_matmul_codegen_static_shape_optimize_s8():
    M = 16384
    N = 16384
    K = 16384

    target = tvm.target.Target("nvidia/nvidia-a100")

    matmul = MatmulWeightOnlyDequantize(
        M=M,
        N=N,
        K=K,
        in_dtype="int8",
        out_dtype="int8",
        accum_dtype="int32",
        propagate_b=True,
        bit=2,
        storage_dtype="int8",
        source_format="uint",
        with_scaling=False,
        group_size=-1,
        fast_decoding=True,
        with_bias=False,
        target=target,
    )
    matmul.optimize()
    code = matmul.codegen(target=target)
    latency = matmul.profile_latency()
    print(latency)
    assert code


if __name__ == "__main__":
    test_weight_only_matmul_codegen_static_shape_optimize()
    # test_weight_only_matmul_codegen_static_shape_optimize_s8()
