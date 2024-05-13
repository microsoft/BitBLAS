# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import roller
from roller.graph import IRNode, OutputNode
from roller.policy import *
import os
from tvm import te

# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/" + fname

arch = "cuda"
arch = roller.arch.__getattribute__(arch)()
dtype="float16"

shapes = [
    [1, 8192, 8192],
    [1, 28672, 8192],
    [1, 8192, 28672],
    [1, 1024, 8192]
]

perf_map = {}
for M, N, K in shapes:

    def gemm(M, N, K):
        A = te.placeholder((M, K), name='A', dtype='float16')
        B = te.placeholder((N, K), name='B', dtype='float16')

        # Describe the matrix multiplication in TE
        k = te.reduce_axis((0, K), name='k')
        C = te.compute(
            (M, N),
            lambda i, j: te.sum(A[i, k].astype('float16') * B[j, k].astype('float16'), axis=[k]),
            name='C'
        )
        return A, B, C


    args = gemm(M, N, K)

    input_args = args[:2]
    output_args = [args[-1]]
    node = IRNode([None for _ in input_args], args, "roller_matmul")
    node.add_tag("tensorCoreConfig", [0, 1])
    # node.add_tag("roller_config", (True, True, 2))
    output_nodes = [OutputNode(node)]
    policy = DefaultPolicy(output_nodes, arch)
    configs = policy.emit_config(20)

    compile_results = []
    cgen = roller.CodeGenerator()
    for config in configs:
        cpresult = cgen.compile(output_nodes, config, "cuda", kernel_name="Fused")
        compile_results.append(cpresult)
    roller.utils.compile_and_load_parallel(compile_results, arch)
    best_latency = 10000
    best = None
    values = []
    for cpresult in compile_results:
        print(cpresult.config)
        code = cpresult.code
        if cpresult.lib is None:
            latency = 10000
        else:
            latency = cpresult.profile()
        values.append(latency)
        if latency < best_latency:
            best_latency = latency
            best = cpresult
        print(latency)
    print("best code", best.code)
    print("top1: {} \ttop10: {}".format(values[0], min(values)))
    print("-" * 80, flush=True)
    print("best config: {}".format(best.config))
    print("best latency: {}".format(best_latency))
    key = "{}_{}_{}".format(M, N, K)
    perf_map[key] = best_latency
    
for M, N, K in shapes:
    key = "{}_{}_{}".format(M, N, K)
    print("{}\t{}".format(key, perf_map[key]))