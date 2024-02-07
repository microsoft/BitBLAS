import tvm
from tvm.script import tir as T
from bitblas.base.roller.policy import TensorCorePolicy, DefaultPolicy
from bitblas.base.roller.arch import CUDA
from bitblas.gpu.matmul_analysis import get_tensorized_func_and_tags
from bitblas.base.utils import apply_and_build
from bitblas.ops.matmul_impl import (
    matmul_nt,
    matmul_nt_propagate_b_s8_s8_s32_mma
)


def test_f16_f16_gemm():
    ir_module = matmul_nt(16384, 16384, 16384, "float16", "float16")
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
        "[FastDlight] The best latency of top 1 is {:.3f} ms".format(
            cpresults[0].latency * 1e3
        )
    )
    print(
        "[FastDlight] The best latency of top 1 is {:.3f} ms".format(best.latency * 1e3)
    )

def test_i8_i8_gemm_propagate_b():
    ir_module = matmul_nt_propagate_b_s8_s8_s32_mma(16384, 16384, 16384, "int8", "int32")
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
        "[FastDlight] The best latency of top 1 is {:.3f} ms".format(
            cpresults[0].latency * 1e3
        )
    )
    print(
        "[FastDlight] The best latency of top 1 is {:.3f} ms".format(best.latency * 1e3)
    )
    print(best.sch.mod)

test_f16_f16_gemm()
# test_i8_i8_gemm_propagate_b()
