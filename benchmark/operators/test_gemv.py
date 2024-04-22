# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tvm
import ladder
from ladder.graph import IRNode, OutputNode
from ladder.policy import *
from tvm import relay
import os.path as osp
from tvm.contrib.target.onnx import to_onnx
from tvm.relay.testing import run_infer_type
from tvm.contrib import graph_executor
import os
from tvm.script import tir as T
from tvm import te

# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/" + fname

arch = "cuda"
arch = ladder.arch.__getattribute__(arch)()
dtype="float16"

shapes = [
    [1, 16384, 16384], 
]

for M, N, K in shapes:
    A = te.placeholder((M, K), name='A', dtype='float16')
    B = te.placeholder((N, K), name='B', dtype='float16')

    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name='k')
    C = te.compute(
        (N,),
        lambda j: te.sum(A[0, k] * B[j, k], axis=k),
        name='C'
    )

    input_args = [A, B]
    output_args = [C]
    node = IRNode([None for _ in input_args], input_args+output_args, "ladder_matmul")
    output_nodes = [OutputNode(node)]
    policy = DefaultPolicy(output_nodes, arch)
    configs = policy.emit_config(20)

    compile_results = []
    cgen = ladder.CodeGenerator()
    for config in configs:
        print(config)
        cpresult = cgen.compile(output_nodes, config, "cuda", kernel_name="Fused")
        compile_results.append(cpresult)
    ladder.utils.compile_and_load_parallel(compile_results, arch)
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
    
    print("top1: {} \ttop10: {}".format(values[0], min(values)))
    print("-" * 80, flush=True)
    print("best config: {}".format(best.config))
    print("best latency: {}".format(best_latency))
    # print("best code: {}".format(best.code))
    key = "{}_{}_{}".format(M, N, K)
    perf_map[key] = best_latency

for M, N, K in shapes:
    key = "{}_{}_{}".format(M, N, K)
    print("{}\t{}".format(key, perf_map[key]))