# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tvm
from bitblas.base.roller.policy import TensorCorePolicy, DefaultPolicy
from bitblas.base.roller.arch import CUDA
from bitblas.gpu.matmul_analysis import get_tensorized_func_and_tags
from bitblas.gpu import Matmul
from bitblas.utils import auto_detect_nvidia_target
from bitblas.base.utils import apply_and_build
from bitblas.ops.impl.matmul_impl import (
    matmul_nn,
    matmul_nt,
    matmul_nt_propagate_a_propagate_b,
)
import time

# fmt:off
test_shapes = [
    # (prim_func, input_args, default_bitblas_schedule),
    (matmul_nt, (1024, 1024, 1024, "float16", "float16"), Matmul),
    (matmul_nt, (16, 8192, 8192, "float16", "float16"), Matmul),
    (matmul_nt, (32, 8192, 8192, "float16", "float16"), Matmul),
    (matmul_nt, (16384, 16384, 16384, "float16", "float16"), Matmul),
    (matmul_nt, (16384, 16384, 16384, "int8", "int32"), Matmul),
    (matmul_nn, (1024, 1024, 1024, "float16", "float16"), Matmul),
    (matmul_nn, (8192, 8192, 8192, "float16", "float16"), Matmul),
    (matmul_nn, (16384, 16384, 16384, "float16", "float16"), Matmul),
    (matmul_nt, (1024, 1024, 1024, "float32", "float32"), Matmul),
    (matmul_nt_propagate_a_propagate_b, (16384, 16384, 16384, "float16", "float16", "float16"),
     Matmul),
]

llm_shapes = [
    # square test
    (matmul_nt_propagate_a_propagate_b, (16384, 16384, 16384, "float16", "float16"), Matmul),
    # BLOOM-176B
    (matmul_nt_propagate_a_propagate_b, (8192, 43008, 14336, "float16", "float16"), Matmul),
    (matmul_nt_propagate_a_propagate_b, (8192, 14336, 14336, "float16", "float16"), Matmul),
    (matmul_nt_propagate_a_propagate_b, (8192, 57344, 14336, "float16", "float16"), Matmul),
    (matmul_nt_propagate_a_propagate_b, (8192, 14336, 57344, "float16", "float16"), Matmul),
    # # OPT-65B
    (matmul_nt_propagate_a_propagate_b, (8192, 9216, 9216, "float16", "float16"), Matmul),
    (matmul_nt_propagate_a_propagate_b, (8192, 36864, 9216, "float16", "float16"), Matmul),
    (matmul_nt_propagate_a_propagate_b, (8192, 9216, 36864, "float16", "float16"), Matmul),
    (matmul_nt_propagate_a_propagate_b, (8192, 22016, 8192, "float16", "float16"), Matmul),
    # # LLAMA-70B/65B
    (matmul_nt_propagate_a_propagate_b, (8192, 8192, 22016, "float16", "float16"), Matmul),
    (matmul_nt_propagate_a_propagate_b, (8192, 8192, 8192, "float16", "float16"), Matmul),
    (matmul_nt_propagate_a_propagate_b, (8192, 28672, 8192, "float16", "float16"), Matmul),
    (matmul_nt_propagate_a_propagate_b, (8192, 8192, 28672, "float16", "float16"), Matmul),

    # square test
    (matmul_nt_propagate_a_propagate_b, (16384, 16384, 16384, "int8", "int8", "int32"), Matmul),
    # BLOOM-176B
    (matmul_nt_propagate_a_propagate_b, (8192, 43008, 14336, "int8", "int8", "int32"), Matmul),
    (matmul_nt_propagate_a_propagate_b, (8192, 14336, 14336, "int8", "int8", "int32"), Matmul),
    (matmul_nt_propagate_a_propagate_b, (8192, 57344, 14336, "int8", "int8", "int32"), Matmul),
    (matmul_nt_propagate_a_propagate_b, (8192, 14336, 57344, "int8", "int8", "int32"), Matmul),
    # OPT-65B
    (matmul_nt_propagate_a_propagate_b, (8192, 9216, 9216, "int8", "int8", "int32"), Matmul),
    (matmul_nt_propagate_a_propagate_b, (8192, 36864, 9216, "int8", "int8", "int32"), Matmul),
    (matmul_nt_propagate_a_propagate_b, (8192, 9216, 36864, "int8", "int8", "int32"), Matmul),
    (matmul_nt_propagate_a_propagate_b, (8192, 22016, 8192, "int8", "int8", "int32"), Matmul),
    # LLAMA-70B/65B
    (matmul_nt_propagate_a_propagate_b, (8192, 8192, 22016, "int8", "int8", "int32"), Matmul),
    (matmul_nt_propagate_a_propagate_b, (8192, 8192, 8192, "int8", "int8", "int32"), Matmul),
    (matmul_nt_propagate_a_propagate_b, (8192, 28672, 8192, "int8", "int8", "int32"), Matmul),
    (matmul_nt_propagate_a_propagate_b, (8192, 8192, 28672, "int8", "int8", "int32"), Matmul),
]

benchmark_sets = []
benchmark_sets.extend(llm_shapes)

# fmt:on

target = tvm.target.Target(auto_detect_nvidia_target())

benchmark_results = {}
for get_prim_func, input_args, d_schedule in benchmark_sets:
    ir_module = get_prim_func(*input_args)
    func = ir_module["main"]
    arch = CUDA(target)
    policy = DefaultPolicy(func=func, arch=arch)
    try:
        tensorized_func, tags = get_tensorized_func_and_tags(func, arch.target)
    except Exception:
        tags = None
    if tags:
        policy = TensorCorePolicy(func=tensorized_func, arch=arch, tags=tags)

    configs = policy.emit_config(20)

    tune_start = time.time()
    cpresults, best = apply_and_build(func, configs, arch, parallel_build=False)
    fast_tune_time = time.time() - tune_start
    print("[BitBLAS] The best latency of top 1 is {:.3f} ms".format(cpresults[0].latency * 1e3))
    print("[BitBLAS] The best latency of top 20 is {:.3f} ms".format(best.latency * 1e3))

    # evaluate the performance of the default schedule

    rule = d_schedule()
    default_tune_start = time.time()
    sch_default = rule.apply(func, target, False)
    with tvm.transform.PassContext(config={"tir.use_async_copy": True}):
        mod_default = tvm.build(sch_default.mod["main"], target="cuda")
    default_tune_time = time.time() - default_tune_start

    args = func.buffer_map.values()

    profile_tensors = best.profile_tensors

    timer_cuda_mod = mod_default.time_evaluator(mod_default.entry_name, arch.device, number=5)
    t = timer_cuda_mod(*profile_tensors).mean

    print("Time cost of BitBLAS default schedule: {:.3f} ms".format(t * 1e3))

    profile_config = {
        f"{get_prim_func.__name__}-{'-'.join([str(i) for i in input_args])}": {
            "fast_bitblas_top20_tune_time": fast_tune_time,
            "fast_bitblas_top1_latency": cpresults[0].latency * 1e3,
            "fast_bitblas_top20_latency": best.latency * 1e3,
            "default_bitblas_tune_time": default_tune_time,
            "default_bitblas_latency": t * 1e3,
        }
    }

    benchmark_results.update(profile_config)

headers = [
    "PrimFunc",
    "Input Arguments",
    "BitBLAS Top20 Tune Time",
    "BitBLAS Top1 Latency",
    "BitBLAS Top20 Latency",
    "DefaultDLight Tune Time",
    "DefaultDLight Latency",
]

col_width = (max(len(word) for row in [headers] + list(profile_config.values()) for word in row) + 2
            )  # padding

print("".join(word.ljust(col_width) for word in headers))

print("-" * col_width * len(headers))

for config, values in benchmark_results.items():
    args = config.split("-")
    func_name = args[0]
    input_args = "-".join(args[1:])
    row = [
        func_name,
        input_args,
        f" {str(values['fast_bitblas_top20_tune_time'])} s",
        f"{values['fast_bitblas_top1_latency']:.3f} ms",
        f"{values['fast_bitblas_top20_latency']:.3f} ms",
        str(values["default_bitblas_tune_time"]),
        f"{values['default_bitblas_latency']:.3f} ms",
    ]
    print("".join(word.ljust(col_width) for word in row))
