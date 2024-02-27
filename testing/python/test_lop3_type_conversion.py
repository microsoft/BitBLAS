# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tvm
from tvm.script import tir as T
import bitblas
from bitblas.base.roller.policy import TensorCorePolicy, DefaultPolicy
from bitblas.base.roller.arch import CUDA
from bitblas.gpu.matmul_analysis import get_tensorized_func_and_tags
from bitblas.base.utils import apply_and_build
from bitblas.ops.matmul_impl import matmul_nt, matmul_nt_dequantize_b
import numpy as np


def test_f16_f16_gemm():
    ir_module = matmul_nt(1, 16384, 16384, "float16", "float16")
    func = ir_module["main"]
    target = tvm.target.Target("nvidia/nvidia-a100")
    arch = CUDA(target)
    policy = DefaultPolicy(func=func, arch=arch)
    try:
        tensorized_func, tags = get_tensorized_func_and_tags(func, arch.target)
    except:
        tags = None
    if tags:
        policy = TensorCorePolicy(func=tensorized_func, arch=arch, tags=tags)

    configs = policy.emit_config(20)

    cpresults, best = apply_and_build(func, configs, arch, parallel_build=True)
    print(
        "[BitBLAS] The best latency of top 1 is {:.3f} ms".format(
            cpresults[0].latency * 1e3
        )
    )
    print(
        "[BitBLAS] The best latency of top 20 is {:.3f} ms".format(best.latency * 1e3)
    )


def test_f16_i4_gemm(M=1, N=16384, K=16384, bit=4, fast_decoding=True):
    ir_module = matmul_nt_dequantize_b(
        M,
        N,
        K,
        "float16",
        bit=bit,
        storage_dtype="uint32",
        with_scaling=True,
        group_size=-1,
        fast_decoding=fast_decoding,
    )
    func = ir_module["main"]
    target = tvm.target.Target("nvidia/nvidia-a100")
    arch = CUDA(target)
    policy = DefaultPolicy(func=func, arch=arch)
    try:
        tensorized_func, tags = get_tensorized_func_and_tags(func, arch.target)
    except:
        tags = None
    if tags:
        policy = TensorCorePolicy(func=tensorized_func, arch=arch, tags=tags)

    configs = policy.emit_config(20)
    # sch = bitblas.gpu.gemv.GEMVWithDequantizeInfo().apply_config(func, configs[0])
    cpresults, best = apply_and_build(func, configs, arch, parallel_build=True)
    print(
        "[BitBLAS] The best latency of top 1 is {:.3f} ms".format(
            cpresults[0].latency * 1e3
        )
    )
    print(
        "[BitBLAS] The best latency of top 20 is {:.3f} ms".format(best.latency * 1e3)
    )
    with open("debug/tmp.cu", "w") as f:
        f.write(str(best.code))


test_f16_i4_gemm()
