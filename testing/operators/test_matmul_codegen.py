import tvm
import bitblas
from bitblas.ops import Matmul
import numpy as np
import torch


def test_matmul_codegen_static_shape_default():
    M = 16384
    N = 16384
    K = 16384

    target = tvm.target.Target("nvidia/nvidia-a100")

    matmul = Matmul(
        M=M,
        N=N,
        K=K,
        a_dtype="float16",
        b_dtype="float16",
        c_dtype="float16",
        propagate_a=False,
        propagate_b=False,
        layout="nt",
        target=target,
    )
    code = matmul.codegen(target=target)
    assert code


def test_matmul_codegen_static_shape_optimize():
    M = 16384
    N = 16384
    K = 16384

    target = tvm.target.Target("nvidia/nvidia-a100")

    matmul = Matmul(
        M=M,
        N=N,
        K=K,
        a_dtype="float16",
        b_dtype="float16",
        c_dtype="float16",
        propagate_a=False,
        propagate_b=False,
        layout="nt",
        target=target,
    )
    matmul.optimize()
    code = matmul.codegen(target=target)
    assert code
    
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


def test_matmul_codegen_dynamic_range_optimize():
    M = [1024]
    N = 1024
    K = 1024

    target = tvm.target.Target("nvidia/nvidia-a100")

    matmul = Matmul(
        M=M,
        N=N,
        K=K,
        a_dtype="float16",
        b_dtype="float16",
        c_dtype="float16",
        propagate_a=False,
        propagate_b=False,
        layout="nt",
        target=target,
    )
    matmul.optimize()
    code = matmul.codegen(target=target)
    print(code)
    assert code


def test_matmul_profile_static_shape_default():
    M = 16384
    N = 16384
    K = 16384

    target = tvm.target.Target("nvidia/nvidia-a100")

    matmul = Matmul(
        M=M,
        N=N,
        K=K,
        a_dtype="float16",
        b_dtype="float16",
        c_dtype="float16",
        propagate_a=False,
        propagate_b=False,
        layout="nt",
        target=target,
    )
    code = matmul.codegen(target=target)
    latency = matmul.profile_latency()
    print(latency)


def test_matmul_profile_dynamic_shape_default():
    M = [16, 32, 64, 128]
    N = 16384
    K = 16384

    target = tvm.target.Target("nvidia/nvidia-a100")

    matmul = Matmul(
        M=M,
        N=N,
        K=K,
        a_dtype="float16",
        b_dtype="float16",
        c_dtype="float16",
        propagate_a=False,
        propagate_b=False,
        layout="nt",
        target=target,
    )
    code = matmul.codegen(target=target)
    latency = matmul.profile_latency()
    print(latency)


def test_matmul_invoke_static_shape_default():
    M = 16384
    N = 16384
    K = 16384

    target = tvm.target.Target("nvidia/nvidia-a100")

    matmul = Matmul(
        M=M,
        N=N,
        K=K,
        a_dtype="float16",
        b_dtype="float16",
        c_dtype="float16",
        propagate_a=False,
        propagate_b=False,
        layout="nt",
        target=target,
    )
    code = matmul.codegen(target=target)
    latency = matmul.profile_latency()
    a = torch.rand((M, K), dtype=torch.float16).cuda()
    b = torch.rand((N, K), dtype=torch.float16).cuda()
    c = torch.empty((M, N), dtype=torch.float16).cuda()
    matmul.forward(a, b, c)


if __name__ == "__main__":
    # test_matmul_codegen_static_shape_default() # passed
    # test_matmul_codegen_static_shape_optimize() # passed
    test_matmul_codegen_static_shape_optimize_s8()
    # test_matmul_codegen_dynamic_range_optimize() # passed
    # test_matmul_profile_static_shape_default() # passed
    # test_matmul_profile_dynamic_shape_default() # passed
    # test_matmul_invoke_static_shape_default()
    # test_matmul_codegen_dynamic_range_optimize()
