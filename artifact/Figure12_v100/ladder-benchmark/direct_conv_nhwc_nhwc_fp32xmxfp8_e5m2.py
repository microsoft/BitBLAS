import numpy as np
import ladder
import tvm
from tvm import relay
import os.path as osp
from tvm.contrib.target.onnx import to_onnx
from tvm.relay.testing import run_infer_type
from tvm.contrib import graph_executor
from ladder.te_utils import connect_tensor_graph
from tvm import te, tir
from ladder.graph import IRNode, OutputNode
from ladder.policy import *
from ladder.reference import get_subgraph_reference_outputs
import os
import torch
from tvm.script import tir as T
# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/" + fname

arch = 'cuda'
arch = ladder.arch.__getattribute__(arch)()
dtype="float32"
bit = 8
n_float_per_i8 = 8 // bit
mask = (1 << bit) - 1

def ladder_conv_nhwc_hwnc(n, f, h, w, c, kh, kw, s, d, p):
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
        return tvm.tir.reinterpret(dtype, (((e_f16 | (s << tvm.tir.const(8, "uint32"))) << 2) | m_f8) << tvm.tir.const(25, "uint32"))
    
    A = te.placeholder((n, h, w, c), name='input', dtype='float32')
    B = te.placeholder((kh*kw*c, f // 8 * bit), name='weight', dtype='int8')
    Scales = te.placeholder(((kh*kw*c) // group_size, f), name='Scales', dtype='uint8')
    
    def B_decode_func(n, k):
        w = _tir_u8_to_f8_to_float(bit, B[n, k // n_float_per_i8], k % n_float_per_i8, "float32", Scales[k // group_size, n])
        return w

    B_decode = te.compute(
        (kh*kw*c, f),
        B_decode_func,
        name='B_decode'
    )
    pad_shape = (n, h + 2 * p, w + 2 * p, c)
    pad_value = tir.const(0.0, A.dtype)
    pad = te.compute(
                    pad_shape,
                    lambda n, h, w, c: te.if_then_else(
                        tir.all(
                            h >= p,
                            w >= p,
                            h < pad_shape[1] - p,
                            w < pad_shape[2] - p,
                        ),
                        A[n, h - p, w - p, c],
                        pad_value,
                    ),
                    name="pad",
                )
    kernel_h, kernel_w = kh, kw
    stride_h, stride_w = s, s
    dilation_h, dilation_w = d, d
    k_size = kernel_h * kernel_w * c
    k_axis = te.reduce_axis((0, k_size), name="k")
    out_h = (
        h + p + p - 1 - (kernel_h - 1) * dilation_h
    ) // stride_h + 1
    out_w = (
        w + p + p - 1 - (kernel_w - 1) * dilation_w
    ) // stride_w + 1
    n_size = out_h * out_w * n
    # Describe the matrix multiplication in TE
    data = te.compute(
                [n_size, k_size],
                lambda n, k: pad[
                    n // (out_h * out_w),
                    (n % (out_h * out_w) // out_w) * stride_h
                    + (k // (kernel_w * (c))) * dilation_h,
                    (n % out_w) * stride_w + (k // (c) % kernel_w) * dilation_w,
                    k % c,
                ],
                name="data",
            )
    C = te.compute(
            [n_size, f],
            lambda i, j: te.sum(data[i, k_axis] * B_decode[k_axis, j], axis=[k_axis]),
            "T_conv",
        )
    return A, B, Scales, C


def reshape(_N, _H, _W, _C, wmma_m, wmma_n):
    M = _N * _H * _W
    N = _C
    C = te.placeholder((M // wmma_m, N // wmma_n, wmma_m, wmma_n), name='C', dtype='float16')
    C_reshpae = te.compute(
        (_N // wmma_m, _H, _W, _C // wmma_n, wmma_m, wmma_n),
        lambda n, h, w, c, nn, cc: C[w + _W *h + _W * _H * n, c, nn, cc],
        name='C_reshape'
    )
    return C, C_reshpae

def reshape_nhwc(_N, _H, _W, _C, wmma_m, wmma_n):
    C = te.placeholder((_N // wmma_m, _H, _W, _C // wmma_n, wmma_m, wmma_n), name='C', dtype='float16')
    C_reshpae = te.compute(
        (_N, _H, _W, _C),
        lambda n, h, w, c: C[n // wmma_m, h, w, c // wmma_n, n % wmma_m, c % wmma_n],
        name='C_reshape'
    )
    return C, C_reshpae

def bias(_N, _H, _W, _C):
    A = te.placeholder((_N, _H, _W, _C), name='A', dtype='float16')
    B = te.placeholder((_C,), name='B', dtype='float16')
    C = te.compute(
        (_N, _H, _W, _C),
        lambda n, h, w, c: A[n, h, w, c] + B[c],
        name='C'
    )
    return A, B, C

def relu(_N, _H, _W, _C):
    A = te.placeholder((_N, _H, _W, _C), name='A', dtype='float16')
    B = te.compute(
        (_N, _H, _W, _C),
        lambda n, h, w, c: te.max(A[n, h, w, c], tir.const(0.0, A.dtype)),
        name='B'
    )
    return A, B

def layout_transform(_N, _H, _W, _C, wmma_m = 16, wmma_n = 16):
    A = te.placeholder((_N, _H, _W, _C), name='A', dtype='float16')
    B = te.compute(
        (_N // wmma_m, _H, _W, _C // wmma_n, wmma_m, wmma_n),
        lambda n, h, w, c, nn, cc: A[n * wmma_m + nn, h, w, c * wmma_n + cc],
        name='B'
    )
    return A, B

def add_conv(_N, _H, _W, _C):
    A = te.placeholder((_N, _H, _W, _C), name='A', dtype='float16')
    B = te.placeholder((_N, _H, _W, _C), name='B', dtype='float16')
    C = te.compute(
        (_N, _H, _W, _C),
        lambda n, h, w, c: A[n, h, w, c] + B[n, h, w, c],
        name='C'
    )
    return A, B, C

def layout_transform_nhwc2nchw(_N, _H, _W, _C):
    A = te.placeholder((_N, _H, _W, _C), name='A', dtype='float16')
    B = te.compute(
        (_N, _C, _H, _W),
        lambda n, c, h, w: A[n, h, w, c],
        name='B'
    )
    return A, B

# the topi layout_transform compute does not simplify well when lowering, so we implement a new one here
def A_global_16x16_to_shared_load_16x16_layout(i, j):
    thread_id = i * 2 + j // 8
    row = thread_id % 16
    col = (j % 8) + (thread_id // 16) * 8
    return row, col

def B_global_16x16_to_shared_load_16x16_layout(i, j):
    thread_id = i * 2 + j // 8
    row = (i // 8) * 8 + (thread_id % 8)
    col = (j % 8) + 8 * ((thread_id // 8) % 2)
    return row, col


def layout_transform_with_func(_N, _H, _W, _C, wmma_m = 16, wmma_n = 16, func=None):
    def fcompute(*args):
        warp_i, warp_j = args[-2:]
        spatial_args = args[:-2]
        permutate_i, permutate_j = func(warp_i, warp_j)
        new_index = (*spatial_args, permutate_i, permutate_j)
        return A[new_index]
    A = te.placeholder((_N // wmma_m, _H, _W, _C // wmma_n, wmma_m, wmma_n), name='A', dtype='float16')
    B = te.compute(
        (_N // wmma_m, _H, _W, _C // wmma_n, wmma_m, wmma_n),
        fcompute,
        name='B'
    )
    return A, B

shufflenet_shapes = [
    [128, 232, 14, 14, 232, 3, 3, 2, 1, 1, 232],
    [128, 116, 14, 14, 116, 1, 1, 1, 1, 0, 1],
    [128, 116, 28, 28, 116, 3, 3, 2, 1, 1, 116],
    [128, 24, 28, 28, 58, 1, 1, 1, 1, 0, 1],
    [128, 24, 56, 56, 24, 3, 3, 2, 1, 1, 24],
    [128, 3, 224, 224, 24, 3, 3, 2, 1, 1, 1],
    [128, 58, 28, 28, 58, 1, 1, 1, 1, 0, 1],
    [128, 58, 56, 56, 58, 3, 3, 2, 1, 1, 58],
    [128, 24, 56, 56, 58, 1, 1, 1, 1, 0, 1],
    [128, 58, 28, 28, 58, 3, 3, 1, 1, 1, 58],
    [128, 116, 28, 28, 116, 1, 1, 1, 1, 0, 1],
    [128, 116, 14, 14, 116, 3, 3, 1, 1, 1, 116],
    [128, 232, 14, 14, 232, 1, 1, 1, 1, 0, 1],
    [128, 232, 7, 7, 232, 3, 3, 1, 1, 1, 232],
]


shufflenet_shapes_b1 = [
    [1, 464, 7, 7, 1024, 1, 1, 1, 1, 0, 1],
    [1, 116, 14, 14, 116, 1, 1, 1, 1, 0, 1],
    # [1, 24, 28, 28, 58, 1, 1, 1, 1, 0, 1],
    # [1, 3, 224, 224, 24, 3, 3, 2, 1, 1, 1],
    # [1, 58, 28, 28, 58, 1, 1, 1, 1, 0, 1],
    # [1, 24, 56, 56, 58, 1, 1, 1, 1, 0, 1],
    # [1, 116, 28, 28, 116, 1, 1, 1, 1, 0, 1],
    # [1, 232, 14, 14, 232, 1, 1, 1, 1, 0, 1],
]


resnet_shapes_b1 = [
    [1, 512, 7, 7, 2048, 1, 1, 1, 1, 0, 1],
    [1, 512, 14, 14, 512, 3, 3, 2, 1, 1, 1],
    [1, 1024, 14, 14, 512, 1, 1, 1, 1, 0, 1],
    [1, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 1],
    [1, 256, 28, 28, 256, 3, 3, 2, 1, 1, 1],
    [1, 512, 28, 28, 256, 1, 1, 1, 1, 0, 1],
    [1, 128, 28, 28, 512, 1, 1, 1, 1, 0, 1],
    [1, 128, 56, 56, 128, 3, 3, 2, 1, 1, 1],
    [1, 256, 56, 56, 128, 1, 1, 1, 1, 0, 1],
    [1, 64, 56, 56, 256, 1, 1, 1, 1, 0, 1],
    [1, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1],
    [1, 64, 56, 56, 64, 1, 1, 1, 1, 0, 1],
    [1, 256, 56, 56, 64, 1, 1, 1, 1, 0, 1],
    [1, 256, 56, 56, 512, 1, 1, 2, 1, 0, 1],
    [1, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1],
    [1, 512, 28, 28, 128, 1, 1, 1, 1, 0, 1],
    [1, 512, 28, 28, 1024, 1, 1, 2, 1, 0, 1],
    [1, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1],
    [1, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 1],
    [1, 1024, 14, 14, 2048, 1, 1, 2, 1, 0, 1],
    [1, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1],
    [1, 2048, 7, 7, 512, 1, 1, 1, 1, 0, 1],
]



shapes = [
    [1, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1],
    [1, 64, 56, 56, 64, 1, 1, 1, 1, 0, 1],
    [1, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1],
    [1, 512, 28, 28, 128, 1, 1, 1, 1, 0, 1],
    [128, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1],
    [128, 64, 56, 56, 64, 1, 1, 1, 1, 0, 1],
    [128, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1],
    [128, 512, 28, 28, 128, 1, 1, 1, 1, 0, 1],
]
perf_map = []
for n, c, h, w, f, kh, kw, s, d, p, g in shapes:
    key = f'{n}_{c}_{h}_{w}_{f}_{kh}_{kw}_{s}_{d}_{p}_{g}'
    oh = (h + 2 * p - kh) // s + 1
    ow = (w + 2 * p - kw) // s + 1
    print("n: {}, f: {}, h: {}, w: {}, c: {}, kh: {}, kw: {}, s: {}, d: {}, p: {}, oh: {}, ow: {}".format(n, f, h, w, c, kh, kw, s, d, p, oh, ow))
    compute_flops = 2 * n * f * oh * ow * c * kh * kw
    arg1 = ladder_conv_nhwc_hwnc(n, f, h, w, c, kh, kw, s, d, p)
    arg2 = reshape(n, oh, ow, f, 16, 16)
    arg3 = reshape_nhwc(n, oh, ow, f, 16, 16)
    arg4 = bias(n, oh, ow, f)
    arg5 = add_conv(n, oh, ow, f)
    arg6 = relu(n, oh, ow, f)
    arg7 = layout_transform(n, oh, ow, f)
    arg8 = layout_transform_with_func(n, oh, ow, f, func=A_global_16x16_to_shared_load_16x16_layout)
    arg9 = layout_transform_nhwc2nchw(n, oh, ow, f)
    
    args = arg1
    # args = tuple(connect_tensor_graph(args, arg2, {arg2[0]:args[-1]}))
    # args = tuple(connect_tensor_graph(args, arg3, {arg3[0]:args[-1]}))
    # args = tuple(connect_tensor_graph(args, arg4, {arg4[0]:args[-1]}))
    # args = tuple(connect_tensor_graph(args, arg5, {arg5[0]:args[-1]}))
    # args = tuple(connect_tensor_graph(args, arg6, {arg6[0]:args[-1]}))
    # args = tuple(connect_tensor_graph(args, arg7, {arg7[0]:args[-1]}))
    # args = tuple(connect_tensor_graph(args, arg8, {arg8[0]:args[-1]}))

    input_args = args[:-1]
    output_args = [args[-1]]
    node = IRNode([None for _ in input_args], args, "ladder_conv2d_reshape_bias")
    # node.add_tag("consistent_config", (True, True))
    output_nodes = [OutputNode(node)]
    policy = DefaultPolicy(output_nodes, arch)
    configs = policy.emit_config(40)

    compile_results = []
    cgen = ladder.CodeGenerator()
    for config in configs:
        try:
            cpresult = cgen.compile(output_nodes, config, "cuda", kernel_name="Fused")
        except:
            continue
        compile_results.append(cpresult)
    ladder.utils.compile_and_load_parallel(compile_results, arch, timeout=100)
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
    print('code: ', code)
    print("top1: {} \ttop10: {}".format(values[0], min(values)))
    print("-" * 80, flush=True)
    print("best config: {}".format(best.config))
    print("best latency: {}".format(best_latency))
    print(f"{(compute_flops/(best_latency * 1e-3))/ pow((1024), 4)} tflops, {(compute_flops/(best_latency * 1e-3))/ pow((1024), 4) / 145 * 100} %")
    
    perf_map.append((key, best_latency))

for key, latency in perf_map:
    print("{}\t{}".format(key, latency))

