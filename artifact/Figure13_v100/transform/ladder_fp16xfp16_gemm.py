import tvm
import ladder
from ladder.graph import IRNode, OutputNode
from ladder.policy import *
from ladder.te_utils import connect_tensor_graph
from tvm import relay
import os.path as osp
from tvm.contrib.target.onnx import to_onnx
from tvm.relay.testing import run_infer_type
from tvm.contrib import graph_executor
import os
from tvm.script import tir as T
from tvm import te
import torch

# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/" + fname

arch = "cuda"
arch = ladder.arch.__getattribute__(arch)()
dtype="float16"

@tvm.register_func
def tvm_callback_cuda_postproc(code):
    # print(code)
    code = code.replace("""decode_i4s_to_f16(input1_shared_local, mediate0_local);""", """decode_i4s_to_f16(input1_shared_local, (half *)mediate0_local);;""")
    code = code.replace("""decode_i4s_to_f16(input1_shared_local_1, mediate0_local_1);""", """decode_i4s_to_f16(input1_shared_local_1, (half *)mediate0_local_1);""")
    return code

wmma_m = 16
wmma_n = 16
wmma_k = 16

bit = 8
n_float_per_i8 = 8 // bit
mask = (1 << bit) - 1
llama2_shapes = [
    [4096, 1024, 8192],
    [4096, 8192, 8192],
    [4096, 28672, 8192],
    [4096, 8192, 28672],
]

shapes = llama2_shapes
out_dtype="float16"
perf_map = []
for M, N, K in shapes:
    def ladder_gemm(M, N, K, wmma_m, wmma_n, wmma_k):
        A = te.placeholder((M // wmma_m, K // wmma_k, wmma_m ,wmma_k), name='A', dtype='float16')
        B = te.placeholder((N // wmma_n, K // wmma_k, wmma_n, wmma_k), name='B', dtype='float16')

        # Describe the matrix multiplication in TE
        k = te.reduce_axis((0, K // wmma_k), name='k')
        kk = te.reduce_axis((0, wmma_k), name='kk')

        C = te.compute(
            (M // wmma_m, N // wmma_n, wmma_m, wmma_n),
            lambda i, j, ii, jj: te.sum(A[i, k, ii, kk].astype(out_dtype) * B[j, k, jj, kk].astype(out_dtype), axis=[k, kk]),
            name='C'
        )

        return A, B, C


    def reshape(M, N, wmma_m, wmma_n):
        C = te.placeholder((M // wmma_m, N // wmma_n, wmma_m, wmma_n), name='C', dtype='float16')
        C_reshape = te.compute(
            (M, N),
            lambda i, j: C[i // wmma_m, j // wmma_n, i % wmma_m, j % wmma_n],
            name='C_reshape'
        )
        return C, C_reshape

    def layout_transform(M, N):
        C = te.placeholder((M, N), name='C', dtype='float16')
        D = te.compute(
            (1, M, N),
            lambda b, i, j: C[i, j],
            name='C_reshape'
        )
        return C, D

    def bias(B, M, N):
        C = te.placeholder((B, M, N), name='C', dtype='float16')
        D = te.placeholder((B, M, N), name='D', dtype='float16')
        E = te.compute(
            (B, M, N),
            lambda b, i, j: C[b, i, j] + D[b, i, j],
            name='C_reshape'
        )
        return C, D, E
    
    arg1 = ladder_gemm(M, N, K, wmma_m, wmma_n, wmma_k)
    arg2 = reshape(M, N, wmma_m, wmma_n)
    arg3 = layout_transform(M, N)
    arg4 = bias(1, M, N)
    args = arg1
    # args = tuple(connect_tensor_graph(args, arg2, {arg2[0]:arg1[-1]}))
    # args = tuple(connect_tensor_graph(args, arg3, {arg3[0]:args[-1]}))
    # args = tuple(connect_tensor_graph(args, arg4, {arg4[0]:args[-1]}))

    input_args = args[:-1]
    output_args = [args[-1]]

    node = IRNode([None for _ in input_args], args, "ladder_matmul")
    node.add_tag("tensorCoreConfig", [2, 3])
    node.add_tag("ladder_config", (False, False, 1))
    output_nodes = [OutputNode(node)]
    policy = LadderPolicy(output_nodes, arch)
    configs = policy.emit_config(20)

    compile_results = []
    cgen = ladder.CodeGenerator()
    for config in configs:
        try:
            cpresult = cgen.compile(output_nodes, config, "cuda", kernel_name="Fused")
        except:
            continue
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
    with open("best_code.cu", "w+") as f:
        f.write(code)
    torch.cuda.cudart().cudaProfilerStart()
    best.get_example_outputs()
    torch.cuda.cudart().cudaProfilerStop()
    # print(best.code)
    print("top1: {} \ttop10: {}".format(values[0], min(values)))
    print("-" * 80, flush=True)
    print("best config: {}".format(best.config))
    print("best latency: {}".format(best_latency))
    key = "{}_{}_{}".format(M, N, K)
    perf_map.append((key, best_latency))

for key, latency in perf_map:
    print("{}\t{}".format(key, latency))
