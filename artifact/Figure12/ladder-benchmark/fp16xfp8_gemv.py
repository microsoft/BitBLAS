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


ft_shapes = [
    [1, 16384, 16384],
]
llm_shapes = [
    [1, 16384, 16384],
    [1, 43008, 14336],
    [1, 14336, 14336],
    [1, 57344, 14336],
    [1, 14336, 57344],
    [1, 9216, 9216],
    [1, 36864, 9216],
    [1, 9216, 36864],
    [1, 22016, 8192],
    [1, 8192, 22016],
    [1, 8192, 8192],
    [1, 28672, 8192],
    [1, 8192, 28672],
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
shapes = [
    [1, 14336, 57344],
    [1, 8192, 28672],
]

perf_map = []

wmma_m = 16
wmma_n = 16
wmma_k = 16

bit = 8
n_float_per_i8 = 8 // bit
mask = (1 << bit) - 1
# shapes = ft_shapes
perf_map = []
for M, N, K in shapes:
    group_size = K
    def _tir_u8_to_f8_to_float(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str):
        assert val.dtype == "int8"
        assert nbit == 8
        return T.reinterpret('float16', T.Cast('int16', val) << 8)

    def ladder_gemm(M, N, K):
        A = te.placeholder((M, K // 8 * bit), name='A', dtype='int8')
        B = te.placeholder((N, K // 8 * bit), name='B', dtype='int8')

        def B_decode_func(n, k):
            w = _tir_u8_to_f8_to_float(bit, B[n, k // n_float_per_i8], k % n_float_per_i8, "float16")
            return w
        
        B_decode = te.compute(
            (N, K),
            B_decode_func,
            name='B_decode'
        )

        # Describe the matrix multiplication in TE
        k = te.reduce_axis((0, K), name='k')

        C = te.compute(
            (M, N),
            lambda i, j: te.sum(A[i, k] * B_decode[j, k], axis=[k]),
            name='C'
        )

        return A, B, C

    arg1 = ladder_gemm(M, N, K)
    args = arg1

    input_args = args[:-1]
    output_args = [args[-1]]

    node = IRNode([None for _ in input_args], args, "ladder_matmul")
    node.add_tag("consistent_config", (False, False))
    output_nodes = [OutputNode(node)]
    policy = DefaultPolicy(output_nodes, arch)
    configs = policy.emit_config(20)

    compile_results = []
    cgen = ladder.CodeGenerator()
    for config in configs:
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

