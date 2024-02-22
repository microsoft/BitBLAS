# pre-transformed tir expression of matmul
import tvm
from tvm.script import tir as T
from tvm import te, tir
from .quantization import (
    _tir_packed_int_to_int_to_float,
    _tir_packed_uint_to_uint_to_float,
)


def matmul_nt_dyn_m(
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
):
    @tvm.script.ir_module
    class MatmulNT:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(a, [m, K], dtype=in_dtype)
            B = T.match_buffer(b, [N, K], dtype=in_dtype)
            C = T.match_buffer(c, [m, N], dtype=out_dtype)

            for i, j, k in T.grid(m, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = tvm.tir.const(0, out_dtype)
                    C[vi, vj] = C[vi, vj] + A[vi, vk].astype(out_dtype) * B[
                        vj, vk
                    ].astype(out_dtype)

    @tvm.script.ir_module
    class MatmulNTWithAccum:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(a, [m, K], dtype=in_dtype)
            B = T.match_buffer(b, [N, K], dtype=in_dtype)
            C = T.match_buffer(c, [m, N], dtype=out_dtype)
            accum = T.alloc_buffer([m, N], dtype=accum_dtype)
            for i, j, k in T.grid(m, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        accum[vi, vj] = tvm.tir.const(0, accum_dtype)
                    accum[vi, vj] = accum[vi, vj] + A[vi, vk].astype(accum_dtype) * B[
                        vj, vk
                    ].astype(accum_dtype)

            for i, j in T.grid(m, N):
                with T.block("C"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    C[vi, vj] = accum[vi, vj].astype(out_dtype)

    @tvm.script.ir_module
    class MatmulNTWithAccumBias:
        @T.prim_func
        def main(a: T.handle, b: T.handle, bias: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(a, [m, K], dtype=in_dtype)
            B = T.match_buffer(b, [N, K], dtype=in_dtype)
            Bias = T.match_buffer(bias, [N], dtype=out_dtype)
            C = T.match_buffer(c, [m, N], dtype=out_dtype)
            accum = T.alloc_buffer([m, N], dtype=accum_dtype)
            accum_bias = T.alloc_buffer([m, N], dtype=out_dtype)
            for i, j, k in T.grid(m, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        accum[vi, vj] = tvm.tir.const(0, accum_dtype)
                    accum[vi, vj] = accum[vi, vj] + A[vi, vk].astype(accum_dtype) * B[
                        vj, vk
                    ].astype(accum_dtype)

            for i, j in T.grid(m, N):
                with T.block("C"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    accum_bias[vi, vj] = accum[vi, vj].astype(out_dtype)

            for i, j in T.grid(m, N):
                with T.block("Bias"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    C[vi, vj] = accum_bias[vi, vj] + Bias[vj]

    final_module = MatmulNT
    if with_bias:
        final_module = MatmulNTWithAccumBias
    elif accum_dtype != out_dtype:
        final_module = MatmulNTWithAccum

    return final_module


def matmul_nn_dyn_m(N, K, in_dtype="float16", out_dtype="float16"):
    @tvm.script.ir_module
    class MatmulNN:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(a, [m, K], dtype=in_dtype)
            B = T.match_buffer(b, [K, N], dtype=in_dtype)
            C = T.match_buffer(c, [m, N], dtype=out_dtype)

            for i, j, k in T.grid(m, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = tvm.tir.const(0, out_dtype)
                    C[vi, vj] = C[vi, vj] + A[vi, vk].astype(out_dtype) * B[
                        vk, vj
                    ].astype(out_dtype)

    return MatmulNN


def matmul_nn(
    M,
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
):
    A = te.placeholder((M, K), name="A", dtype=in_dtype)
    B = te.placeholder((K, N), name="B", dtype=in_dtype)
    Bias = te.placeholder((N,), name="Bias", dtype=in_dtype)

    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(
            A[i, k].astype(accum_dtype) * B[k, j].astype(accum_dtype), axis=k
        ),
        name="C",
    )
    last_output = C
    if accum_dtype != out_dtype:
        D = te.compute((M, N), lambda i, j: C[i, j].astype(out_dtype), name="D")
        last_output = D

    if with_bias:
        E = te.compute((M, N), lambda i, j: last_output[i, j] + Bias[j], name="E")
        last_output = E

    if with_bias:
        args = [A, B, Bias, last_output]
    else:
        args = [A, B, last_output]

    func = te.create_prim_func(args)

    return tvm.IRModule.from_expr(func)


def matmul_nt(
    M,
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
):
    A = te.placeholder((M, K), name="A", dtype=in_dtype)
    B = te.placeholder((N, K), name="B", dtype=in_dtype)
    Bias = te.placeholder((N,), name="Bias", dtype=in_dtype)

    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(
            A[i, k].astype(accum_dtype) * B[j, k].astype(accum_dtype), axis=k
        ),
        name="C",
    )
    last_output = C
    if accum_dtype != out_dtype:
        D = te.compute((M, N), lambda i, j: C[i, j].astype(out_dtype), name="D")
        last_output = D

    if with_bias:
        E = te.compute((M, N), lambda i, j: last_output[i, j] + Bias[j], name="E")
        last_output = E

    if with_bias:
        args = [A, B, Bias, last_output]
    else:
        args = [A, B, last_output]

    func = te.create_prim_func(args)

    return tvm.IRModule.from_expr(func)


def matmul_nt_propagate_a_propagate_b_s8_s8_s32_mma(
    M, N, K, in_dtype="int8", out_dtype="int32"
):
    wm, wn, wk = 16, 16, 16
    if in_dtype == "int8":
        wm, wn, wk = 16, 16, 32

    @tvm.script.ir_module
    class MyModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "tir.noalias": True,
                    "smooth_a": True,
                    "smooth_b": True,
                }
            )
            A = T.match_buffer(a, [M // wm, K // wk, wm, wk], dtype=in_dtype)
            B = T.match_buffer(b, [N // wn, K // wk, wn, wk], dtype=in_dtype)
            C = T.match_buffer(c, [M, N], dtype=out_dtype)
            A_reindex = T.alloc_buffer([M, K], dtype=in_dtype)
            B_reindex = T.alloc_buffer([N, K], dtype=in_dtype)

            for i, k in T.grid(M, K):
                with T.block("A_reindex"):
                    vi, vk = T.axis.remap("SS", [i, k])
                    A_reindex[vi, vk] = A[
                        vi // wn,
                        vk // wk,
                        vi % wn % 8 * 2 + vk % wk // 16,
                        vi % wn // 8 * 16 + vk % 16,
                    ]

            for j, k in T.grid(N, K):
                with T.block("B_reindex"):
                    vj, vk = T.axis.remap("SS", [j, k])
                    B_reindex[vj, vk] = B[
                        vj // wn,
                        vk // wk,
                        vj % wn // 8 * 8 + vj % 4 * 2 + vk % wk // 16,
                        vj % 8 // 4 * 16 + vk % 16,
                    ]

            for i, j, k in T.grid(M, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = tvm.tir.const(0, out_dtype)
                    C[vi, vj] = C[vi, vj] + A_reindex[vi, vk].astype(
                        out_dtype
                    ) * B_reindex[vj, vk].astype(out_dtype)

    return MyModule


def matmul_nt_propagate_a_propagate_b_s8_s8_s32_mma_cast_s8(
    M, N, K, in_dtype="int8", out_dtype="int32"
):
    wm, wn, wk = 16, 16, 16
    if in_dtype == "int8":
        wm, wn, wk = 16, 16, 32

    @tvm.script.ir_module
    class MyModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "tir.noalias": True,
                    "smooth_a": True,
                    "smooth_b": True,
                }
            )
            A = T.match_buffer(a, [M // wm, K // wk, wm, wk], dtype=in_dtype)
            B = T.match_buffer(b, [N // wn, K // wk, wn, wk], dtype=in_dtype)
            C = T.alloc_buffer([M, N], dtype=out_dtype)
            A_reindex = T.alloc_buffer([M, K], dtype=in_dtype)
            B_reindex = T.alloc_buffer([N, K], dtype=in_dtype)
            D = T.match_buffer(c, [M, N], dtype="int8")

            for i, k in T.grid(M, K):
                with T.block("A_reindex"):
                    vi, vk = T.axis.remap("SS", [i, k])
                    A_reindex[vi, vk] = A[
                        vi // wn,
                        vk // wk,
                        vi % wn % 8 * 2 + vk % wk // 16,
                        vi % wn // 8 * 16 + vk % 16,
                    ]

            for j, k in T.grid(N, K):
                with T.block("B_reindex"):
                    vj, vk = T.axis.remap("SS", [j, k])
                    B_reindex[vj, vk] = B[
                        vj // wn,
                        vk // wk,
                        vj % wn // 8 * 8 + vj % 4 * 2 + vk % wk // 16,
                        vj % 8 // 4 * 16 + vk % 16,
                    ]

            for i, j, k in T.grid(M, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = tvm.tir.const(0, out_dtype)
                    C[vi, vj] = C[vi, vj] + A_reindex[vi, vk].astype(
                        out_dtype
                    ) * B_reindex[vj, vk].astype(out_dtype)

            for i, j in T.grid(M, N):
                with T.block(""):
                    vi, vj = T.axis.remap("SS", [i, j])
                    D[vi, vj] = C[vi, vj].astype("int8")

    return MyModule


def matmul_nt_propagate_b_s8_s8_s32_mma(M, N, K, in_dtype="int8", out_dtype="int32"):
    wm, wn, wk = 16, 16, 16
    if in_dtype == "int8":
        wm, wn, wk = 16, 16, 32

    @tvm.script.ir_module
    class MyModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {"global_symbol": "main", "tir.noalias": True, "smooth_b": True}
            )
            A = T.match_buffer(a, [M, K], dtype=in_dtype)
            B = T.match_buffer(b, [N // wn, K // wk, wn, wk], dtype=in_dtype)
            C = T.match_buffer(c, [M, N], dtype=out_dtype)
            B_reindex = T.alloc_buffer([N, K], dtype=in_dtype)

            for j, k in T.grid(N, K):
                with T.block("B_reindex"):
                    vj, vk = T.axis.remap("SS", [j, k])
                    B_reindex[vj, vk] = B[
                        vj // wn,
                        vk // wk,
                        vj % wn // 8 * 8 + vj % 4 * 2 + vk % wk // 16,
                        vj % 8 // 4 * 16 + vk % 16,
                    ]

            for i, j, k in T.grid(M, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = tvm.tir.const(0, out_dtype)
                    C[vi, vj] = C[vi, vj] + A[vi, vk].astype(out_dtype) * B_reindex[
                        vj, vk
                    ].astype(out_dtype)

    return MyModule


def matmul_nt_propagate_b_s8_s8_s32_cast_s8_mma(
    M, N, K, in_dtype="int8", out_dtype="int32"
):
    wm, wn, wk = 16, 16, 16
    if in_dtype == "int8":
        wm, wn, wk = 16, 16, 32

    @tvm.script.ir_module
    class MyModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {"global_symbol": "main", "tir.noalias": True, "smooth_b": True}
            )
            A = T.match_buffer(a, [M, K], dtype=in_dtype)
            B = T.match_buffer(b, [N // wn, K // wk, wn, wk], dtype=in_dtype)
            C = T.alloc_buffer([M, N], dtype=out_dtype)
            B_reindex = T.alloc_buffer([N, K], dtype=in_dtype)
            D = T.match_buffer(c, [M, N], dtype="int8")

            for j, k in T.grid(N, K):
                with T.block("B_reindex"):
                    vj, vk = T.axis.remap("SS", [j, k])
                    B_reindex[vj, vk] = B[
                        vj // wn,
                        vk // wk,
                        vj % wn // 8 * 8 + vj % 4 * 2 + vk % wk // 16,
                        vj % 8 // 4 * 16 + vk % 16,
                    ]

            for i, j, k in T.grid(M, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = tvm.tir.const(0, out_dtype)
                    C[vi, vj] = C[vi, vj] + A[vi, vk].astype(out_dtype) * B_reindex[
                        vj, vk
                    ].astype(out_dtype)

            for i, j in T.grid(M, N):
                with T.block("C"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    D[vi, vj] = C[vi, vj].astype("int8")

    return MyModule


def matmul_nt_propagate_b_f16_f16_f16_mma(
    M, N, K, in_dtype="float16", out_dtype="float16"
):
    wm, wn, wk = 16, 16, 16
    if in_dtype == "int8":
        wm, wn, wk = 16, 16, 32

    @tvm.script.ir_module
    class MyModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {"global_symbol": "main", "tir.noalias": True, "smooth_b": True}
            )
            A = T.match_buffer(a, [M, K], dtype=in_dtype)
            B = T.match_buffer(b, [N // wn, K // wk, wn, wk], dtype=in_dtype)
            C = T.match_buffer(c, [M, N], dtype=out_dtype)
            B_reindex = T.alloc_buffer([N, K], dtype=in_dtype)

            for j, k in T.grid(N, K):
                with T.block("B_reindex"):
                    vj, vk = T.axis.remap("SS", [j, k])
                    B_reindex[vj, vk] = B[
                        vj // wn,
                        vk // wk,
                        vj % wn // 8 * 8 + vj % 4 * 2 + vk % wn // 8,
                        vj % 8 // 4 * 8 + vk % 8,
                    ]

            for i, j, k in T.grid(M, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = tvm.tir.const(0, out_dtype)
                    C[vi, vj] = C[vi, vj] + A[vi, vk].astype(out_dtype) * B_reindex[
                        vj, vk
                    ].astype(out_dtype)

    return MyModule


def matmul_nt_propagate_a_b(M, N, K, in_dtype="float16", out_dtype="float16"):
    wm, wn, wk = 16, 16, 16
    if in_dtype == "int8":
        wm, wn, wk = 16, 16, 32

    @tvm.script.ir_module
    class MyModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "tir.noalias": True,
                    "smooth_a": True,
                    "smooth_b": True,
                }
            )
            A = T.match_buffer(a, [M // wm, K // wk, wm, wk], dtype=in_dtype)
            B = T.match_buffer(b, [N // wn, K // wk, wn, wk], dtype=in_dtype)
            C = T.match_buffer(c, [M, N], dtype=out_dtype)
            A_reindex = T.alloc_buffer([M, K], dtype=in_dtype)
            B_reindex = T.alloc_buffer([N, K], dtype=in_dtype)

            for i, k in T.grid(M, K):
                with T.block("A_reindex"):
                    vj, vk = T.axis.remap("SS", [i, k])
                    A_reindex[vj, vk] = A[vj // wm, vk // wk, vj % wm, vk % wk]

            for j, k in T.grid(N, K):
                with T.block("B_reindex"):
                    vj, vk = T.axis.remap("SS", [j, k])
                    B_reindex[vj, vk] = B[vj // wn, vk // wk, vj % wn, vk % wk]

            for i, j, k in T.grid(M, N, K):
                with T.block("C"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = tvm.tir.const(0, out_dtype)
                    C[vi, vj] = C[vi, vj] + A_reindex[vi, vk].astype(
                        out_dtype
                    ) * B_reindex[vj, vk].astype(out_dtype)

    return MyModule


def matmul_nt_dequantize_b(
    M,
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    bit=4,
    storage_dtype="int8",
    source_format="uint",
    with_scaling=False,
    group_size=-1,
    fast_decoding=False,
    with_bias=False,
):
    storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))

    n_float_per_elem = storage_nbit // bit
    if group_size == -1:
        group_size = K
    A = te.placeholder((M, K), name="A", dtype=in_dtype)
    B = te.placeholder((N, K // storage_nbit * bit), name="B", dtype=storage_dtype)
    Scale = te.placeholder((N, K // group_size), name="Scale", dtype=in_dtype)
    Bias = te.placeholder((N,), name="Bias", dtype=in_dtype)

    def decode_func(n, k):
        if storage_dtype[:3] == "int":
            w = _tir_packed_int_to_int_to_float(storage_nbit)(
                bit, B[n, k // n_float_per_elem], k % n_float_per_elem, dtype=in_dtype
            )
        elif storage_dtype[:4] == "uint":
            w = _tir_packed_uint_to_uint_to_float(storage_nbit)(
                bit, B[n, k // n_float_per_elem], k % n_float_per_elem, dtype=in_dtype
            )
        else:
            raise ValueError("Unsupported storage dtype: {}".format(storage_dtype))

        if with_scaling:
            w = w * Scale[n, k // group_size]
        return w

    B_decode = te.compute((N, K), decode_func, name="B_decode")

    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(
            A[i, k].astype(accum_dtype) * B_decode[j, k].astype(accum_dtype), axis=k
        ),
        name="C",
    )
    D = te.compute((M, N), lambda i, j: C[i, j].astype(out_dtype), name="D")
    args = [A, B]
    last_output = D
    if with_scaling:
        args.append(Scale)
    if with_bias:
        E = te.compute((M, N), lambda i, j: D[i, j] + Bias[j], name="E")
        last_output = E
        args.append(Bias)
    args.append(last_output)

    func = te.create_prim_func(args).with_attr(
        "dequantize_info",
        {
            "B_decode": {
                "decode_block": "B_decode",
                "fast_decoding": fast_decoding,
                "source_format": {
                    "bits": bit,
                    "format": source_format,
                },
                "storage_dtype": storage_dtype,
                "target_format": in_dtype,
                "with_scaling": with_scaling,
                "group_size": group_size,
            }
        },
    )
    return tvm.IRModule.from_expr(func)


def matmul_nt_dequantize_b_propagate_b(
    M,
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    bit=4,
    storage_dtype="int8",
    source_format="uint",
    with_scaling=False,
    group_size=-1,
    fast_decoding=False,
    with_bias=False,
):
    wm, wn, wk = 16, 16, 16
    if in_dtype == "int8":
        wm, wn, wk = 16, 16, 32

    storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))

    n_float_per_elem = storage_nbit // bit
    if group_size == -1:
        group_size = K

    A = te.placeholder((M, K), name="A", dtype=in_dtype)
    B = te.placeholder((N // wn, K // wk, wn, wk // 8 * bit), name="B", dtype="int8")
    Scale = te.placeholder((N, K // group_size), name="Scale", dtype=in_dtype)
    Bias = te.placeholder((N,), name="Bias", dtype=in_dtype)

    def decode_func(n, k, nn, kk):
        if storage_dtype[:3] == "int":
            w = _tir_packed_int_to_int_to_float(storage_nbit)(
                bit,
                B[n, k, nn, kk // n_float_per_elem],
                kk % n_float_per_elem,
                dtype=in_dtype,
            )
        elif storage_dtype[:4] == "uint":
            w = _tir_packed_uint_to_uint_to_float(storage_nbit)(
                bit,
                B[n, k, nn, kk // n_float_per_elem],
                kk % n_float_per_elem,
                dtype=in_dtype,
            )
        else:
            raise ValueError("Unsupported storage dtype: {}".format(storage_dtype))

        if with_scaling:
            w = w * Scale[n * wn + nn, (k * wk + kk) // group_size]
        return w

    B_decode = te.compute((N // wn, K // wk, wn, wk), decode_func, name="B_decode")

    B_reindex = te.compute(
        (N, K),
        lambda i, j: B_decode[i // wn, j // wk, i % wn, j % wk],
        name="B_reindex",
    )

    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(
            A[i, k].astype(accum_dtype) * B_reindex[j, k].astype(accum_dtype), axis=k
        ),
        name="C",
    )
    D = te.compute((M, N), lambda i, j: C[i, j].astype(out_dtype), name="D")
    args = [A, B]
    last_output = D
    if with_scaling:
        args.append(Scale)
    if with_bias:
        E = te.compute((M, N), lambda i, j: D[i, j] + Bias[j], name="E")
        last_output = E
        args.append(Bias)
    args.append(last_output)

    func = te.create_prim_func([A, B, D]).with_attr(
        "dequantize_info",
        {
            "B_decode": {
                "decode_block": "B_decode",
                "fast_decoding": fast_decoding,
                "source_format": {
                    "bits": bit,
                    "format": source_format,
                },
                "storage_dtype": storage_dtype,
                "target_format": in_dtype,
                "with_scaling": with_scaling,
                "group_size": group_size,
            }
        },
    )
    func = func.with_attr("smooth_b", True)

    return tvm.IRModule.from_expr(func)


def matmul_nt_dequantize_b_propagate_a_b(
    M,
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    bit=4,
    storage_dtype="int8",
    source_format="uint",
    with_scaling=False,
    group_size=-1,
    fast_decoding=False,
):
    if in_dtype == "int8":
        wm, wn, wk = 16, 16, 32

    storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))

    n_float_per_elem = storage_nbit // bit
    if group_size == -1:
        group_size = K

    A = te.placeholder((M // wm, K // wk, wm, wk), name="A", dtype=in_dtype)
    B = te.placeholder((N // wn, K // wk, wn, wk // 8 * bit), name="B", dtype="int8")
    Scale = te.placeholder((N, K // group_size), name="Scale", dtype=in_dtype)

    def decode_func(n, k, nn, kk):
        if storage_dtype[:3] == "int":
            w = _tir_packed_int_to_int_to_float(storage_nbit)(
                bit,
                B[n, k, nn, kk // n_float_per_elem],
                kk % n_float_per_elem,
                dtype=in_dtype,
            )
        elif storage_dtype[:4] == "uint":
            w = _tir_packed_uint_to_uint_to_float(storage_nbit)(
                bit,
                B[n, k, nn, kk // n_float_per_elem],
                kk % n_float_per_elem,
                dtype=in_dtype,
            )
        else:
            raise ValueError("Unsupported storage dtype: {}".format(storage_dtype))

        if with_scaling:
            w = w * Scale[n * wn + nn, (k * wk + kk) // group_size]
        return w

    B_decode = te.compute((N // wn, K // wk, wn, wk), decode_func, name="B_decode")

    B_reindex = te.compute(
        (N, K),
        lambda i, j: B_decode[i // wn, j // wk, i % wn, j % wk],
        name="B_reindex",
    )

    A_reindex = te.compute(
        (M, K), lambda i, j: A[i // wm, j // wk, i % wm, j % wk], name="A_reindex"
    )
    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(
            A_reindex[i, k].astype(out_dtype) * B_reindex[j, k].astype(out_dtype),
            axis=k,
        ),
        name="C",
    )
    D = te.compute((M, N), lambda i, j: C[i, j].astype(cast_dtype), name="D")
    func = te.create_prim_func([A, B, D]).with_attr(
        "dequantize_info",
        {
            "B_decode": {
                "decode_block": "B_decode",
                "fast_decoding": fast_decoding,
                "source_format": {
                    "bits": bit,
                    "format": source_format,
                },
                "storage_dtype": storage_dtype,
                "target_format": in_dtype,
                "with_scaling": with_scaling,
                "group_size": group_size,
            }
        },
    )
    func = func.with_attr("smooth_a", True)
    func = func.with_attr("smooth_b", True)

    return tvm.IRModule.from_expr(func)


def matmul_nt_af4(M, N, K, in_dtype="float16", out_dtype="float16"):
    bit = 4
    n_float_per_i8 = 8 // bit

    def _tir_u8_to_int(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return (val >> (pos * nbit).astype("int8")) & mask

    A = te.placeholder((M, K), name="A", dtype=in_dtype)
    B = te.placeholder((N, K // 8 * bit), name="B", dtype="int8")
    LUT = te.placeholder((1 << bit,), name="LUT", dtype="float16")

    def decode_func(n, k):
        w = _tir_u8_to_int(bit, B[n, k // n_float_per_i8], k % n_float_per_i8)
        return LUT[w]

    B_decode = te.compute((N, K), decode_func, name="B_decode")

    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N), lambda i, j: te.sum(A[i, k] * B_decode[j, k], axis=k), name="C"
    )
    func = te.create_prim_func([A, B, LUT, C]).with_attr(
        "dequantize_info",
        {
            "B": {
                "decode_block": "B_decode",
                "source_format": {
                    "bits": 4,
                    "format": "af",
                },
                "target_format": "float16",
            }
        },
    )
    return tvm.IRModule.from_expr(func)


def matmul_nt_af4_propagate_a_b(M, N, K, in_dtype="float16", out_dtype="float16"):
    bit = 4
    n_float_per_i8 = 8 // bit

    def _tir_u8_to_int(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return (val >> (pos * nbit).astype("int8")) & mask

    A = te.placeholder((M // 16, K // 16, 16, 16), name="A", dtype=in_dtype)
    B = te.placeholder((N // 16, K // 16, 16, 16 // 8 * bit), name="B", dtype="int8")
    LUT = te.placeholder((1 << bit,), name="LUT", dtype="float16")

    def decode_func(n, k, nn, kk):
        w = _tir_u8_to_int(bit, B[n, k, nn, kk // n_float_per_i8], kk % n_float_per_i8)
        return LUT[w]

    B_decode = te.compute((N // 16, K // 16, 16, 16), decode_func, name="B_decode")

    B_reindex = te.compute(
        (N, K),
        lambda i, j: B_decode[i // 16, j // 16, i % 16, j % 16],
        name="B_reindex",
    )

    A_reindex = te.compute(
        (M, K), lambda i, j: A[i // 16, j // 16, i % 16, j % 16], name="A_reindex"
    )
    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N), lambda i, j: te.sum(A_reindex[i, k] * B_reindex[j, k], axis=k), name="C"
    )
    func = te.create_prim_func([A, B, LUT, C]).with_attr(
        "dequantize_info",
        {
            "B": {
                "decode_block": "B_decode",
                "source_format": {
                    "bits": 4,
                    "format": "af",
                },
                "target_format": "float16",
            }
        },
    )
    func = func.with_attr("smooth_a", True)
    func = func.with_attr("smooth_b", True)

    return tvm.IRModule.from_expr(func)


# register the func
matmul_impl_factory = {
    "matmul_nt": matmul_nt,
    "matmul_nt_dyn_m": matmul_nt_dyn_m,
    "matmul_nn": matmul_nn,
    "matmul_nn_dyn_m": matmul_nn_dyn_m,
    "matmul_nt_propagate_b_f16_f16_mma": matmul_nt_propagate_b_f16_f16_f16_mma,
    "matmul_nt_pa_pb": matmul_nt_propagate_a_b,
    "matmul_nt_pa_pb_f16_f16_mma": matmul_nt_propagate_a_b,
    "matmul_nt_dequantize_b": matmul_nt_dequantize_b,
    "matmul_nt_dequantize_b_pb": matmul_nt_dequantize_b_propagate_b,
}
