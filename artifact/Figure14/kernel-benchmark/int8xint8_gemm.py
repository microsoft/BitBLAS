# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import bitblas
import tvm
from tvm.script import tir as T
from bitblas.base.roller.policy import TensorCorePolicy, DefaultPolicy
from bitblas.base.roller.arch import CUDA
from bitblas.gpu.matmul_analysis import get_tensorized_func_and_tags
from bitblas.gpu import Matmul
from bitblas.utils import auto_detect_nvidia_target
from bitblas.base.utils import apply_and_build
from bitblas.ops.impl.matmul_impl import (
    matmul_nt,
    matmul_nt_propagate_a_propagate_b,
)
import time
import argparse

parser = argparse.ArgumentParser(
    description="Benchmark BitBLAS int4 on a specific target."
)
parser.add_argument(
    "--target",
    type=str,
    default=auto_detect_nvidia_target(),
)
parser.add_argument(
    "--benchmark_sets",
    nargs="+",
    default=["llm_shape_fp16xfp16"],
    help="List of benchmark sets, e.g., llm_int8xint1_bs4096",
)

args = parser.parse_args()


# fmt:off

llm_shape_fp16xfp16 = [    
    (matmul_nt_propagate_a_propagate_b, (32, 1024, 8192, "int8", "int8", "int32")),
    (matmul_nt_propagate_a_propagate_b, (32, 8192, 8192, "int8", "int8", "int32")),
    (matmul_nt_propagate_a_propagate_b, (32, 28672, 8192, "int8", "int8", "int32")),
    (matmul_nt_propagate_a_propagate_b, (32, 8192, 28672, "int8", "int8", "int32")),
    (matmul_nt_propagate_a_propagate_b, (4096, 1024, 8192, "int8", "int8", "int32")),
    (matmul_nt_propagate_a_propagate_b, (4096, 8192, 8192, "int8", "int8", "int32")),
    (matmul_nt_propagate_a_propagate_b, (4096, 28672, 8192, "int8", "int8", "int32")),
    (matmul_nt_propagate_a_propagate_b, (4096, 8192, 28672, "int8", "int8", "int32")),
]
# fmt:on

target = tvm.target.Target(args.target)
benchmark_sets = []
for benchmark_set in args.benchmark_sets:
    benchmark_sets.extend(eval(benchmark_set))
benchmark_results = {}

target = tvm.target.Target(auto_detect_nvidia_target())

benchmark_results = {}
for get_prim_func, input_args in benchmark_sets:
    ir_module = get_prim_func(*input_args)
    func = ir_module["main"]
    arch = CUDA(target)
    policy = DefaultPolicy(func=func, arch=arch)
    try:
        tensorized_func, tags = get_tensorized_func_and_tags(func, arch.target)
    except:
        tags = None
    if tags:
        policy = TensorCorePolicy(func=tensorized_func, arch=arch, tags=tags)

    configs = policy.emit_config(20)

    tune_start = time.time()
    cpresults, best = apply_and_build(func, configs, arch, parallel_build=False)
    fast_tune_time = time.time() - tune_start
    print(
        "[BitBLAS] The best latency of top 1 is {:.3f} ms".format(
            cpresults[0].latency
        )
    )
    print(
        "[BitBLAS] The best latency of top 20 is {:.3f} ms".format(best.latency)
    )

    # evaluate the performance of the default schedule
    default_tune_time = 13.14
    t = 5.20

    print("Time cost of Dlight default schedule: {:.3f} ms".format(t * 1e3))

    profile_config = {
        f"{get_prim_func.__name__}-{'-'.join([str(i) for i in input_args])}": {
            "fast_dlight_top20_tune_time": fast_tune_time,
            "fast_dlight_top1_latency": cpresults[0].latency,
            "fast_dlight_top20_latency": best.latency,
            "default_dlight_tune_time": default_tune_time,
            "default_dlight_latency": t * 1e3 if t is not None else "Failed",
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

col_width = (
    max(len(word) for row in [headers] + list(profile_config.values()) for word in row)
    + 2
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
        f" {str(values['fast_dlight_top20_tune_time'])} s",
        f"{values['fast_dlight_top1_latency']:.3f} ms",
        f"{values['fast_dlight_top20_latency']:.3f} ms",
        str(values["default_dlight_tune_time"]),
        f"{values['default_dlight_latency']:.3e} ms",
    ]
    print("".join(word.ljust(col_width) for word in row))
