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
from ladder.te_utils import connect_tensor_graph
import torch
import time
# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/" + fname

arch = "cuda"
arch = ladder.arch.__getattribute__(arch)()
dtype="float16"

shapes = [
    [16384, 16384, 16384],

]

ft_shapes = [
    [16, 16384, 16384]
    # [16, 12288, 3072],
    # [16, 9216, 12288],
    # [16, 12288, 12288],
    # [16, 12800, 12288],
    # [256, 9216, 12288],
    # [256, 12288, 3072],
    # [256, 12288, 12288],
]

llm_shapes = [
    [4096, 1024, 8192],
    [4096, 8192, 8192],
    [4096, 28672, 8192],
    [4096, 8192, 28672],
]

wmma_m = 16
wmma_n = 16
wmma_k = 16
# shapes = ft_shapes
shapes = llm_shapes
perf_map = {}
cost_map = {}
bit = 8
n_float_per_i8 = 8 // bit
mask = (1 << bit) - 1
shapes = ft_shapes
shapes = llm_shapes
for M, N, K in shapes:
    group_size = 32
    
    def _tir_u8_to_f8_to_float(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str, scale: tvm.tir.PrimExpr = None):
        assert nbit == 8
        # e_f4 == 0 -> e_f32 = 0
        mask = tvm.tir.const((1 << nbit) - 1, "uint32")
        f8 = (val >> (pos.astype("uint32") * tvm.tir.const(nbit, "uint32"))) & mask
        s = f8 >> tvm.tir.const(7, "uint32")
        # e5m2
        e_f8 = (f8 >> tvm.tir.const(2, "uint32")) & tvm.tir.const(31, "uint32")
        e_f16 = T.max(e_f8 + scale, tvm.tir.const(63, "uint32"))
        m_f8 = e_f8 & tvm.tir.const(2, "uint32")
        return tvm.tir.reinterpret(dtype, (e_f16 | (s << tvm.tir.const(8, "uint32"))) << tvm.tir.const(7, "uint32") | m_f8)
  
        
    def ladder_gemm(M, N, K, wmma_m, wmma_n, wmma_k):
        A = te.placeholder((M // wmma_m, K // wmma_k, wmma_m, wmma_k // 8 * bit), name='A', dtype='int8')
        B = te.placeholder((N // wmma_n, K // wmma_k, wmma_n, wmma_k // 8 * bit), name='B', dtype='int8')
        AScales = te.placeholder((K // group_size, N), name='AScales', dtype='uint8')
        BScales = te.placeholder((K // group_size, N), name='BScales', dtype='uint8')

        def A_decode_func(n, k, nn, kk):
            w = _tir_u8_to_f8_to_float(bit, A[n, k, nn, kk // n_float_per_i8], kk % n_float_per_i8, "float16", AScales[(k * wmma_k + kk) // group_size, n * wmma_n + nn])
            return w
        
        A_decode = te.compute(
            (M // wmma_m, K // wmma_k, wmma_m, wmma_k),
            A_decode_func,
            name='A_decode'
        )
        
        def B_decode_func(n, k, nn, kk):
            w = _tir_u8_to_f8_to_float(bit, B[n, k, nn, kk // n_float_per_i8], kk % n_float_per_i8, "float16", BScales[(k * wmma_k + kk) // group_size, n * wmma_n + nn])
            return w
        
        B_decode = te.compute(
            (N // wmma_n, K // wmma_k, wmma_n, wmma_k),
            B_decode_func,
            name='B_decode'
        )
        
        # Describe the matrix multiplication in TE
        k = te.reduce_axis((0, K // wmma_k), name='k')
        kk = te.reduce_axis((0, wmma_k), name='kk')
        C = te.compute(
            (M // wmma_m, N // wmma_n, wmma_m, wmma_n),
            lambda i, j, ii, jj: te.sum(A_decode[i, k, ii, kk].astype('float32') * B_decode[j, k, jj, kk].astype('float32'), axis=[k, kk]),
            name='C'
        )
        return A, B, AScales, BScales, C


    def reshape(M, N, wmma_m, wmma_n):
        C = te.placeholder((M // wmma_m, N // wmma_n, wmma_m, wmma_n), name='C', dtype='float32')
        C_reshape = te.compute(
            (M, N),
            lambda i, j: C[i // wmma_m, j // wmma_n, i % wmma_m, j % wmma_n],
            name='C_reshape'
        )
        return C, C_reshape

    arg1 = ladder_gemm(M, N, K, wmma_m, wmma_n, wmma_k)
    # arg2 = reshape(M, N, wmma_m, wmma_n)
    args = arg1
    # args = connect_tensor_graph(arg1, arg2, {arg2[0]:arg1[2]})

    input_args = args[:-1]
    output_args = [args[-1]]
    node = IRNode([None for _ in input_args], args, "ladder_matmul")
    node.add_tag("tensorCoreConfig", [2, 3])
    node.add_tag("consistent_config", (False, False))
    pipeline_stage = 2
    if arch.platform == 'CUDA':
        if arch.compute_capability == '80':
            pipeline_stage = 2
    node.add_tag("ladder_config", (True, True, pipeline_stage))
    output_nodes = [OutputNode(node)]
    policy = LadderPolicy(output_nodes, arch)
    start = time.time()
    configs = policy.emit_config(20)

    compile_results = []
    cgen = ladder.CodeGenerator()
    for config in configs:
        cpresult = cgen.compile(output_nodes, config, "cuda", kernel_name="Fused")
        compile_results.append(cpresult)
    ladder.utils.compile_and_load_parallel(compile_results, arch)
    cost = time.time() - start
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
    # print(best.code)
    print("top1: {} \ttop10: {}".format(values[0], min(values)))
    print("-" * 80, flush=True)
    print("best config: {}".format(best.config))
    print("best latency: {}".format(best_latency))
    key = "{}_{}_{}".format(M, N, K)
    perf_map[key] = best_latency
    cost_map[key] = cost
    torch.cuda.cudart().cudaProfilerStart()
    best.get_example_outputs()
    torch.cuda.cudart().cudaProfilerStop()
    
for M, N, K in shapes:
    key = "{}_{}_{}".format(M, N, K)
    print("{}\t{}\t{}".format(key, perf_map[key], cost_map[key]))