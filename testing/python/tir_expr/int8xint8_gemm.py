# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tvm
import bitblas
import numpy as np
from bitblas.base.roller.policy import TensorCorePolicy, DefaultPolicy
from bitblas.base.roller.arch import CUDA
from bitblas.gpu.matmul_analysis import get_tensorized_func_and_tags
from bitblas.base.utils import apply_and_build
from bitblas.ops.matmul_impl import (
    matmul_nt,
    matmul_nt_dequantize_b,
    matmul_nt_dequantize_b_propagate_b,
    matmul_nt_dequantize_b_propagate_a_b,
    matmul_nt_propagate_b_s8_s8_s32_mma,
    matmul_nt_propagate_b_s8_s8_s32_cast_s8_mma,
    matmul_nt_propagate_a_propagate_b_s8_s8_s32_mma,
    matmul_nt_propagate_a_propagate_b_s8_s8_s32_mma_cast_s8,
)


def test_i8_i8_gemm():
    ir_module = matmul_nt(16384, 16384, 16384, "int8", "int32")
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
    print("[BitBLAS] The best latency of top 1 is {:.3f} ms".format(best.latency * 1e3))
    with open("debug/after_memory_rewrite.cu", "+w") as f:
        f.write(best.code)


def test_i8_i8_gemm_correctness():
    ir_module = matmul_nt(1024, 1024, 1024, "int8", "int32")
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
    print("[BitBLAS] The best latency of top 1 is {:.3f} ms".format(best.latency * 1e3))

    numpy_a = np.random.randint(-4, 3, (1024, 1024)).astype("int8")
    numpy_b = np.random.randint(-4, 3, (1024, 1024)).astype("int8")
    numpy_c = np.matmul(numpy_a.astype("int32"), numpy_b.T.astype("int32"))
    ctx = tvm.cuda()
    tvm_a = tvm.nd.array(numpy_a, device=ctx)
    tvm_b = tvm.nd.array(numpy_b, device=ctx)
    tvm_c = tvm.nd.array(np.zeros((1024, 1024), dtype="int32"), device=ctx)
    # print(best.sch.mod)
    # print(best.code)
    best.mod(tvm_a, tvm_b, tvm_c)
    print(best.config)
    print("numpy_c ", numpy_c)
    print("tvm_c.asnumpy() ", tvm_c.asnumpy())

    np.testing.assert_allclose(tvm_c.asnumpy(), numpy_c, atol=1e-5)
    # print(best.code)


def test_i8_i8_i32_gemm_propagate_b():
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

    configs = policy.emit_config(20)

    cpresults, best = apply_and_build(func, configs, arch, parallel_build=True)
    print(
        "[BitBLAS] The best latency of top 1 is {:.3f} ms".format(
            cpresults[0].latency * 1e3
        )
    )
    print("[BitBLAS] The best latency of top 1 is {:.3f} ms".format(best.latency * 1e3))


def test_i8_i8_i32_cast_i8_gemm_propagate_b():
    ir_module = matmul_nt_propagate_b_s8_s8_s32_cast_s8_mma(
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

    configs = policy.emit_config(20)

    cpresults, best = apply_and_build(func, configs, arch, parallel_build=True)
    print(
        "[BitBLAS] The best latency of top 1 is {:.3f} ms".format(
            cpresults[0].latency * 1e3
        )
    )
    print("[BitBLAS] The best latency of top 1 is {:.3f} ms".format(best.latency * 1e3))


def test_i8_i8_i32_gemm_propagate_a_propagate_b():
    ir_module = matmul_nt_propagate_a_propagate_b_s8_s8_s32_mma(
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

    configs = policy.emit_config(20)

    cpresults, best = apply_and_build(func, configs, arch, parallel_build=True)
    print(
        "[BitBLAS] The best latency of top 1 is {:.3f} ms".format(
            cpresults[0].latency * 1e3
        )
    )
    print("[BitBLAS] The best latency of top 1 is {:.3f} ms".format(best.latency * 1e3))


def test_i8_i8_i32_gemm_propagate_a_propagate_b_cast_s8():
    ir_module = matmul_nt_propagate_a_propagate_b_s8_s8_s32_mma_cast_s8(
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

    configs = policy.emit_config(20)

    cpresults, best = apply_and_build(func, configs, arch, parallel_build=True)
    print(
        "[BitBLAS] The best latency of top 1 is {:.3f} ms".format(
            cpresults[0].latency * 1e3
        )
    )
    print("[BitBLAS] The best latency of top 1 is {:.3f} ms".format(best.latency * 1e3))


def test_i8_i4_gemm():
    ir_module = matmul_nt_dequantize_b(16384, 16384, 16384, "int8", "int32")
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
    print("[BitBLAS] The best latency of top 1 is {:.3f} ms".format(best.latency * 1e3))


def test_i8_i4_propagate_b_gemm():
    ir_module = matmul_nt_dequantize_b_propagate_b(16384, 16384, 16384, "int8", "int32")
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
    print("[BitBLAS] The best latency of top 1 is {:.3f} ms".format(best.latency * 1e3))
    # print(best.sch.mod)
    print(best.code)


def test_i8_i4_propagate_a_propagate_b_gemm():
    ir_module = matmul_nt_dequantize_b_propagate_a_b(
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

    configs = policy.emit_config(20)

    cpresults, best = apply_and_build(func, configs, arch, parallel_build=True)
    print(
        "[BitBLAS] The best latency of top 1 is {:.3f} ms".format(
            cpresults[0].latency * 1e3
        )
    )
    print("[BitBLAS] The best latency of top 1 is {:.3f} ms".format(best.latency * 1e3))
    print(best.config)


def test_i8_i2_gemm():
    ir_module = matmul_nt_dequantize_b(1, 16384, 16384, "int8", "int32", bit=2)
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
    print(configs)
    cpresults, best = apply_and_build(func, configs, arch, parallel_build=True)
    print(
        "[BitBLAS] The best latency of top 1 is {:.3f} ms".format(
            cpresults[0].latency * 1e3
        )
    )
    print("[BitBLAS] The best latency of top 1 is {:.3f} ms".format(best.latency * 1e3))
    print(best.code)


def test_i8_i2_propagate_b_gemm():
    ir_module = matmul_nt_dequantize_b_propagate_b(
        16384,
        16384,
        16384,
        "int8",
        "int8",
        accum_dtype="int32",
        bit=2,
        fast_decoding=True,
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
    cpresults, best = apply_and_build(func, configs, arch, parallel_build=True)
    print(
        "[BitBLAS] The best latency of top 1 is {:.3f} ms".format(
            cpresults[0].latency * 1e3
        )
    )
    print("[BitBLAS] The best latency of top 1 is {:.3f} ms".format(best.latency * 1e3))
    with open("debug/after_memory_rewrite.cu", "+w") as f:
        f.write(best.code)


def test_i8_i2_propagate_a_propagate_b_gemm():
    ir_module = matmul_nt_dequantize_b_propagate_a_b(
        16384, 16384, 16384, "int8", "int32", "int8", bit=2, fast_decoding=False
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
    cpresults, best = apply_and_build(func, configs, arch, parallel_build=True)
    print(
        "[BitBLAS] The best latency of top 1 is {:.3f} ms".format(
            cpresults[0].latency * 1e3
        )
    )
    print("[BitBLAS] The best latency of top 1 is {:.3f} ms".format(best.latency * 1e3))
    with open("debug/after_memory_rewrite.cu", "+w") as f:
        f.write(best.code)


# test_i8_i8_gemm()
# test_i8_i8_gemm_correctness()
# test_i8_i8_i32_gemm_propagate_b()
# test_i8_i8_i32_cast_i8_gemm_propagate_b()
# test_i8_i8_i32_gemm_propagate_a_propagate_b()
# test_i8_i8_i32_gemm_propagate_a_propagate_b_cast_s8()
# test_i8_i4_gemm()
# test_i8_i4_propagate_b_gemm()
# test_i8_i4_propagate_a_propagate_b_gemm()

test_i8_i2_gemm()
# test_i8_i2_propagate_b_gemm()
# test_i8_i2_propagate_a_propagate_b_gemm()
