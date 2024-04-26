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

# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/" + fname

arch = "cuda"
arch = ladder.arch.__getattribute__(arch)()
dtype="float16"

shapes = [
    # [1, 1024, 8192], 
    # [1, 8192, 8192], 
    # [1, 8192, 28672], 
    # [1, 28672, 8192],
    [1, 16384, 16384] 
]
ft_shapes = [
    [1, 9216, 12288],
    [1, 12288, 12288],
    [1, 12800, 12288],
    [1, 12288, 3072],
]
llama2_shapes = [
    # 70b
    [1, 1024, 8192],
    [1, 1024, 8192],
    [1, 8192, 8192],
    [1, 8192, 8192],
    [1, 28672, 8192],
    [1, 28672, 8192],
    [1, 8192, 28672],
]
bloom_shapes = [
    [1, 43008, 14336],
    [1, 14336, 14336],
    [1, 57344, 14336],
    [1, 14336, 57344],
]
shapes = llama2_shapes + bloom_shapes
perf_map = []
for M, N, K in shapes:
    wmma_m = 16
    wmma_n = 16
    wmma_k = 16
    
    bit = 4
    n_float_per_i8 = 8 // bit
    mask = (1 << bit) - 1
    group_size = K

    def _tir_u8_to_int(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return (val >> (pos * nbit).astype("int8")) & mask

        
    def gemm(M, N, K):
        A = te.placeholder((M, K), name='A', dtype='float16')
        B = te.placeholder((N, K // 8 * bit), name='B', dtype='int8')
        LUT = te.placeholder((1 << bit, ), name='LUT', dtype='float16')
        def decode_func(n, k):
            w = _tir_u8_to_int(bit, B[n, k // n_float_per_i8], k % n_float_per_i8)
            return LUT[w]

        B_decode = te.compute(
            (N, K),
            decode_func,
            name='B_decode'
        )

        # Describe the matrix multiplication in TE
        k = te.reduce_axis((0, K), name='k')
        C = te.compute(
            (M, N),
            lambda i, j: te.sum(A[i, k] * B_decode[j, k], axis=k),
            name='C'
        )

        return A, B, LUT, C

    arg1 = gemm(M, N, K)
    args = arg1

    input_args = args[:-1]
    output_args = [args[-1]]

    node = IRNode([None for _ in input_args], arg1, "ladder_matmul")
    output_nodes = [OutputNode(node)]
    policy = DefaultPolicy(output_nodes, arch)
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
        code = cpresult.code
        if cpresult.lib is None:
            latency = 10000
        else:
            latency = cpresult.profile()
        values.append(latency)
        if latency < best_latency:
            best_latency = latency
            best = cpresult
        # print(latency)

    print(best.code)
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