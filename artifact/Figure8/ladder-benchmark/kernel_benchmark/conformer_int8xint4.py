import tvm
import welder
from welder.graph import IRNode, OutputNode
from welder.policy import *
from tvm import relay
import os.path as osp
from tvm.contrib.target.onnx import to_onnx
from tvm.relay.testing import run_infer_type
from tvm.contrib import graph_executor
import os
from tvm.script import tir as T
from tvm import te
from welder.te_utils import connect_tensor_graph
from welder.reference import get_subgraph_reference_outputs
import numpy as np
import torch

# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/" + fname

arch = "cuda"
arch = welder.arch.__getattribute__(arch)()
dtype="float16"

shapes = [
    [65536, 512, 512],
    [512, 512, 512],
]

ft_shapes = [
    # [16, 12288, 3072],
    # [16, 9216, 12288],
    # [16, 12288, 12288],
    # [16, 12800, 12288],
    # [256, 9216, 12288],
    # [256, 12288, 3072],
    # [256, 12288, 12288],
    # [16384, 16384, 16384]
    [4096, 28672, 8192],
]

wmma_m = 16
wmma_n = 16
wmma_k = 32
# shapes = ft_shapes
perf_map = {}
for M, N, K in shapes:

    
    def ladder_gemm(M, N, K, wmma_m, wmma_n, wmma_k):
        def _tir_u8_to_int(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str):
            assert val.dtype == "int8"
            mask = tvm.tir.const((1 << nbit) - 1, "int8")
            return ((val >> (pos * nbit).astype("int8")) & mask).astype(dtype)

        def decode_func(n, k, nn, kk):
            return _tir_u8_to_int(4, B[n, k, nn, kk // 2], kk % 2, dtype="int8")

        A = te.placeholder((M // wmma_m, K // wmma_k, wmma_m, wmma_k), name='A', dtype='int8')
        B = te.placeholder((N // wmma_n, K // wmma_k, wmma_n,
                           wmma_k // 8 * 4), name='B', dtype='int8')
        
        B_decode = te.compute(
            (N // wmma_n, K // wmma_k, wmma_n, wmma_k),
            decode_func,
            name='B_decompress'
        )
        # Describe the matrix multiplication in TE
        k = te.reduce_axis((0, K // wmma_k), name='k')
        kk = te.reduce_axis((0, wmma_k), name='kk')
        C = te.compute(
            (M // wmma_m, N // wmma_n, wmma_m, wmma_n),
            lambda i, j, ii, jj: te.sum(A[i, k, ii, kk].astype(
                "int32") * B_decode[j, k, jj, kk].astype("int32"), axis=[k, kk]),
            name='C'
        )
        return A, B, C


    def reshape(M, N, wmma_m, wmma_n):
        C = te.placeholder((M // wmma_m, N // wmma_n, wmma_m,
                           wmma_n), name='C', dtype='int32')
        C_reshape = te.compute(
            (M, N),
            lambda i, j: C[i // wmma_m, j // wmma_n, i % wmma_m, j % wmma_n],
            name='C_reshape'
        )
        return C, C_reshape

    arg1 = ladder_gemm(M, N, K, wmma_m, wmma_n, wmma_k)
    arg2 = reshape(M, N, wmma_m, wmma_n)
    args = connect_tensor_graph(arg1, arg2, {arg2[0]:arg1[2]})

    input_args = args[:2]
    output_args = [args[-1]]
    node = IRNode([None for _ in input_args], args, "ladder_matmul")
    node.add_tag("tensorCoreConfig", [2, 3])
    node.add_tag("ladder_config", (True, True, 2))
    # node.add_tag("ladder_config", (False, False))
    output_nodes = [OutputNode(node)]
    policy = LadderPolicy(output_nodes, arch)
    configs = policy.emit_config(20)

    compile_results = []
    cgen = welder.CodeGenerator()
    for config in configs:
        cpresult = cgen.compile(output_nodes, config, "cuda", kernel_name="Fused")
        compile_results.append(cpresult)
    welder.utils.compile_and_load_parallel(compile_results, arch)
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
    
    with open("int8xint4_ladder_gemm.cu", "w+") as f:
        f.write(code)
    print("top1: {} \ttop10: {}".format(values[0], min(values)))
    print("-" * 80, flush=True)
    print("best config: {}".format(best.config))
    print("best latency: {}".format(best_latency))
    key = "{}_{}_{}".format(M, N, K)
    perf_map[key] = best_latency
    torch.cuda.cudart().cudaProfilerStart()
    best.get_example_outputs()
    torch.cuda.cudart().cudaProfilerStop()
    if False:
        out = best.get_example_outputs()
        ref_out = get_subgraph_reference_outputs(output_nodes)
        print(out[0])
        print(ref_out[0])
        for a, b in zip(out, ref_out):
            diff = np.max(np.abs(a-b))
            print("value diff:", diff)
for M, N, K in shapes:
    key = "{}_{}_{}".format(M, N, K)
    print("{}\t{}".format(key, perf_map[key]))