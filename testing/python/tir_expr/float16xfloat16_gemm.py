# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tvm
from tvm.script import tir as T
from bitblas.base.roller.policy import TensorCorePolicy, DefaultPolicy
from bitblas.base.roller.arch import CUDA
from bitblas.gpu.matmul_analysis import get_tensorized_func_and_tags
from bitblas.base.utils import apply_and_build
from bitblas.ops.matmul_impl import matmul_nt, matmul_nt_propagate_b_s8_s8_s32_mma
import numpy as np


def test_f16_f16_gemm():
    ir_module = matmul_nt(1024, 1024, 1024, "float16", "float16")
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

    configs = policy.emit_config(1)

    cpresults, best = apply_and_build(func, configs, arch, parallel_build=True)
    print(
        "[BitBLAS] The best latency of top 1 is {:.3f} ms".format(
            cpresults[0].latency * 1e3
        )
    )
    print(
        "[BitBLAS] The best latency of top 1 is {:.3f} ms".format(best.latency * 1e3)
    )

    numpy_a = np.random.randint(-4, 3, (1024, 1024)).astype("float16")
    numpy_b = np.random.randint(-4, 3, (1024, 1024)).astype("float16")
    numpy_c = np.matmul(numpy_a.astype("float16"), numpy_b.T.astype("float16"))
    ctx = tvm.cuda()
    tvm_a = tvm.nd.array(numpy_a, device=ctx)
    tvm_b = tvm.nd.array(numpy_b, device=ctx)
    tvm_c = tvm.nd.array(np.zeros((1024, 1024), dtype="float16"), device=ctx)
    print(best.code)
    best.mod(tvm_a, tvm_b, tvm_c)
    print(best.config)
    print("numpy_c ", numpy_c)
    print("tvm_c.asnumpy() ", tvm_c.asnumpy())


def test_i8_i8_gemm_propagate_b():
    ir_module = matmul_nt_propagate_b_s8_s8_s32_mma(
        16384, 16384, 16384, "int8", "int32"
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

    configs = policy.emit_config(1)

    cpresults, best = apply_and_build(func, configs, arch, parallel_build=True)
    print(
        "[BitBLAS] The best latency of top 1 is {:.3f} ms".format(
            cpresults[0].latency * 1e3
        )
    )
    print(
        "[BitBLAS] The best latency of top 1 is {:.3f} ms".format(best.latency * 1e3)
    )
    print(best.sch.mod)


test_f16_f16_gemm()
# test_i8_i8_gemm_propagate_b()
