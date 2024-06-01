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
import torch
from tvm.script import tir as T
from tvm import te
from ladder.te_utils import connect_tensor_graph

# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/" + fname

arch = "cuda"
arch = ladder.arch.__getattribute__(arch)()
dtype="float16"

shapes = [
    # [16384, 16384, 16384],
    # [8192, 43008, 14336],
    # [8192, 14336, 14336],
    # [8192, 57344, 14336],
    # [8192, 14336, 57344],
    # [8192, 9216, 9216],
    # [8192, 36864, 9216],
    # [8192, 9216, 36864],
    # [8192, 22016, 8192],
    # [8192, 8192, 22016],
    # [8192, 8192, 8192],
    # [8192, 28672, 8192],
    # [8192, 8192, 28672],

    [32, 1024, 8192],
    [32, 1024, 8192],
    [32, 8192, 8192],
    [32, 8192, 8192],
    [32, 28672, 8192],
    [32, 28672, 8192],
    [32, 8192, 28672],
    [32, 1024, 8192],
    [4096, 1024, 8192],
    [4096, 8192, 8192],
    [4096, 8192, 8192],
    [4096, 28672, 8192],
    [4096, 28672, 8192],
    [4096, 8192, 28672],

]

shapes = [
    [32, 14336, 57344],
    [4096, 14336, 57344],
    [32, 8192, 28672],
    [4096, 8192, 28672],
]

perf_map = []
for M, N, K in shapes:

    def gemm(M, N, K):
        A = te.placeholder((M, K), name='A', dtype='float16')
        B = te.placeholder((N, K), name='B', dtype='float16')

        # Describe the matrix multiplication in TE
        k = te.reduce_axis((0, K), name='k')
        C = te.compute(
            (M, N),
            lambda i, j: te.sum(A[i, k] * B[j, k], axis=[k]),
            name='C'
        )
        return A, B, C


    args = gemm(M, N, K)

    input_args = args[:2]
    output_args = [args[-1]]
    node = IRNode([None for _ in input_args], args, "ladder_matmul")
    node.add_tag("tensorCoreConfig", [0, 1])
    # node.add_tag("ladder_config", (True, True))
    output_nodes = [OutputNode(node)]
    # policy = TCPolicy(output_nodes, arch)
    # policy = LadderPolicy(output_nodes, arch)
    policy = DefaultPolicy(output_nodes, arch)
    configs = policy.emit_config(20)

    compile_results = []
    cgen = ladder.CodeGenerator()
    for config in configs:
        cpresult = cgen.compile(output_nodes, config, "cuda", kernel_name="Fused")
        compile_results.append(cpresult)
    ladder.utils.compile_and_load_parallel(compile_results, arch, timeout=20)
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
    key = "{}_{}_{}".format(M, N, K)
    perf_map.append((key, best_latency))

for key, latency in perf_map:
    print("{}: {}".format(key, latency))