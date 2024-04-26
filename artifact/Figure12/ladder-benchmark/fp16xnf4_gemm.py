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

shapes = [
    [16384, 16384, 16384],
    # [16, 1024, 8192],
    # [16, 8192, 8192],
    # [16, 8192, 28672],
    # [16, 28672, 8192],
    # [32, 1024, 8192],
    # [32, 8192, 8192],
    # [32, 8192, 28672],
    # [32, 28672, 8192],
    # [64, 1024, 8192],
    # [64, 8192, 8192],
    # [64, 8192, 28672],
    # [64, 28672, 8192],
    # [128, 1024, 8192],
    # [128, 8192, 8192],
    # [128, 8192, 28672],
    # [128, 28672, 8192],
    # [1024, 1024, 8192],
    # [1024, 8192, 8192],
    # [1024, 8192, 28672],
    # [1024, 28672, 8192],
    # [4096, 1024, 8192],
    # [4096, 8192, 8192],
    # [4096, 8192, 28672],
    # [4096, 28672, 8192],
    # [8192, 1024, 8192],
    # [8192, 8192, 8192],
    # [8192, 8192, 28672],
    # [8192, 28672, 8192],
    # [16384, 1024, 8192],
    # [16384, 8192, 8192],
    # [16384, 8192, 28672],
    # [16384, 28672, 8192],
]

ft_shapes = [
    # [16, 12288, 3072],
    # [16, 9216, 12288],
    # [16, 12288, 12288],
    # [16, 12800, 12288],
    # [256, 9216, 12288],
    # [256, 12288, 3072],
    # [256, 12288, 12288],
]
llama2_shapes = [
    # 70b
    [32, 1024, 8192],
    [32, 1024, 8192],
    [32, 8192, 8192],
    [32, 8192, 8192],
    [32, 28672, 8192],
    [32, 28672, 8192],
    [32, 8192, 28672],
    [4096, 1024, 8192],
    [4096, 1024, 8192],
    [4096, 8192, 8192],
    [4096, 8192, 8192],
    [4096, 28672, 8192],
    [4096, 28672, 8192],
    [4096, 8192, 28672],
]
bloom_shapes = [
    # 176b
    [32, 43008, 14336],
    [32, 14336, 14336],
    [32, 57344, 14336],
    [32, 14336, 57344],
    [4096, 43008, 14336],
    [4096, 14336, 14336],
    [4096, 57344, 14336],
    [4096, 14336, 57344], 
]
text_encoder_shapes = [
    [512, 768, 768],
    [512, 3072, 3072],
]

vae_encoder_shapes = [
    [32768, 512, 512],
]

vae_decoder_shapes = [
    [32768, 512, 512],
]

Conformer_shapes = [
    [65536, 512, 512],
]

vit_shapes = [
    [6400, 384, 384],
    [6400, 1536, 1536],
]

shapes = llama2_shapes + bloom_shapes + text_encoder_shapes + vae_encoder_shapes + vae_decoder_shapes + Conformer_shapes + vit_shapes
perf_map = []
for M, N, K in shapes:
    wmma_m = 16
    wmma_n = 16
    wmma_k = 16

    bit = 4
    n_float_per_i8 = 8 // bit
    mask = (1 << bit) - 1


    def _tir_u8_to_int(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return (val >> (pos * nbit).astype("int8")) & mask
        

    def ladder_gemm(M, N, K, wmma_m, wmma_n, wmma_k):
        A = te.placeholder((M // wmma_m, K // wmma_k, wmma_m ,wmma_k), name='A', dtype='float16')
        B = te.placeholder((N // wmma_n, K // wmma_k, wmma_n, wmma_k // 8 * bit), name='B', dtype='int8')
        LUT = te.placeholder((1 << bit, ), name='LUT', dtype='float16')

        def decode_func(n, k, nn, kk):
            w = _tir_u8_to_int(bit, B[n, k, nn, kk // n_float_per_i8], kk % n_float_per_i8)
            return LUT[w]

        B_decode = te.compute(
            (N // wmma_n, K // wmma_k, wmma_n, wmma_k),
            decode_func,
            name='B_decode'
        )

        # Describe the matrix multiplication in TE
        k = te.reduce_axis((0, K // wmma_k), name='k')
        kk = te.reduce_axis((0, wmma_k), name='kk')

        C = te.compute(
            (M // wmma_m, N // wmma_n, wmma_m, wmma_n),
            lambda i, j, ii, jj: te.sum(A[i, k, ii, kk] * B_decode[j, k, jj, kk], axis=[k, kk]),
            name='C'
        )

        return A, B, LUT, C


    def reshape(M, N, wmma_m, wmma_n):
        C = te.placeholder((M // wmma_m, N // wmma_n, wmma_m, wmma_n), name='C', dtype='float16')
        C_reshape = te.compute(
            (M, N),
            lambda i, j: C[i // wmma_m, j // wmma_n, i % wmma_m, j % wmma_n],
            name='C_reshape'
        )
        return C, C_reshape

    arg1 = ladder_gemm(M, N, K, wmma_m, wmma_n, wmma_k)
    # arg2 = reshape(M, N, wmma_m, wmma_n)
    # args = connect_tensor_graph(arg1, arg2, {arg2[0]:arg1[2]})

    input_args = arg1[:3]
    output_args = [arg1[-1]]

    node = IRNode([None for _ in input_args], arg1, "ladder_matmul")
    node.add_tag("tensorCoreConfig", [2, 3])
    node.add_tag("ladder_config", (True, True))
    output_nodes = [OutputNode(node)]
    policy = LadderPolicy(output_nodes, arch)
    configs = policy.emit_config(40)

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
    # print(best.code)
    print("top1: {} \ttop10: {}".format(values[0], min(values)))
    print("-" * 80, flush=True)
    print("best config: {}".format(best.config))
    print("best latency: {}".format(best_latency))
    with open("best_code.cu", "w") as f:
        f.write(best.code)
    key = "{}_{}_{}".format(M, N, K)
    perf_map.append((key, best_latency))

for key, latency in perf_map:
    print("{}\t{}".format(key, latency))
