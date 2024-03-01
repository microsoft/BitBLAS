# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# pre-transformed tir expression of matmul
import tvm
from tvm.script import tir as T
from tvm import te, tir
from bitblas.gpu.matmul_analysis import get_propagate_map


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


def matmul_nn_dyn_m(
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
):
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

    @tvm.script.ir_module
    class MatmulNNWithAccum:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(a, [m, K], dtype=in_dtype)
            B = T.match_buffer(b, [K, N], dtype=in_dtype)
            C = T.match_buffer(c, [m, N], dtype=out_dtype)
            accum = T.alloc_buffer([m, N], dtype=accum_dtype)
            for i, j, k in T.grid(m, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        accum[vi, vj] = tvm.tir.const(0, accum_dtype)
                    accum[vi, vj] = accum[vi, vj] + A[vi, vk].astype(accum_dtype) * B[
                        vk, vj
                    ].astype(accum_dtype)

            for i, j in T.grid(m, N):
                with T.block("C"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    C[vi, vj] = accum[vi, vj].astype(out_dtype)

    @tvm.script.ir_module
    class MatmulNNWithAccumBias:
        @T.prim_func
        def main(a: T.handle, b: T.handle, bias: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(a, [m, K], dtype=in_dtype)
            B = T.match_buffer(b, [K, N], dtype=in_dtype)
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
                        vk, vj
                    ].astype(accum_dtype)

            for i, j in T.grid(m, N):
                with T.block("C"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    accum_bias[vi, vj] = accum[vi, vj].astype(out_dtype)

            for i, j in T.grid(m, N):
                with T.block("Bias"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    C[vi, vj] = accum_bias[vi, vj] + Bias[vj]

    final_module = MatmulNN
    if with_bias:
        final_module = MatmulNNWithAccumBias
    elif accum_dtype != out_dtype:
        final_module = MatmulNNWithAccum

    return final_module


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


def matmul_dyn_m(
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
    layout="nt",
):
    if layout == "nn":
        return matmul_nn_dyn_m(N, K, in_dtype, out_dtype, accum_dtype, with_bias)
    return matmul_nt_dyn_m(N, K, in_dtype, out_dtype, accum_dtype, with_bias)


def matmul(
    M,
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
    layout="nt",
):
    if layout == "nn":
        return matmul_nn(M, N, K, in_dtype, out_dtype, accum_dtype, with_bias)
    return matmul_nt(M, N, K, in_dtype, out_dtype, accum_dtype, with_bias)


# always assume propagate both intra and inter layout in BitBLAS
# as we do not have to do runtime layout transform
def matmul_nt_propagate_a_dyn_m(
    M,
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
): ...


def matmul_nt_propagate_b_dyn_m(
    M,
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
): ...


def matmul_nt_propagate_a_propagate_b_dyn_m(
    M,
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
): ...


def matmul_nt_propagate_a(
    M,
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
):
    l = r = 16
    if in_dtype == "int8":
        l, r = 16, 32

    intra_index_map, _ = get_propagate_map(trans=False, dtype=in_dtype, matrix_name="A")

    A = te.placeholder((M // l, K // r, l, r), name="A", dtype=in_dtype)
    B = te.placeholder((N, K), name="B", dtype=in_dtype)
    Bias = te.placeholder((N,), name="Bias", dtype=in_dtype)

    def fcompute(i, j):
        warp_i, warp_j = i % l, j % r
        spatial_args = i // l, j // r
        permutate_i, permutate_j = intra_index_map.map_indices([warp_i, warp_j])
        new_index = (*spatial_args, permutate_i, permutate_j)
        return A[new_index]

    A_reindex = te.compute(
        (M, K),
        fcompute,
        name="A_reindex",
    )
    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(
            A_reindex[i, k].astype(accum_dtype) * B[j, k].astype(accum_dtype), axis=k
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
    func = func.with_attr("smooth_a", True)

    return tvm.IRModule.from_expr(func)


def matmul_nt_propagate_b(
    M,
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
):
    l = r = 16
    if in_dtype == "int8":
        l, r = 16, 32

    intra_index_map, _ = get_propagate_map(trans=True, dtype=in_dtype, matrix_name="B")

    A = te.placeholder((M, K), name="A", dtype=in_dtype)
    B = te.placeholder((N // l, K // r, l, r), name="B", dtype=in_dtype)
    Bias = te.placeholder((N,), name="Bias", dtype=in_dtype)

    def fcompute(i, j):
        warp_i, warp_j = i % l, j % r
        spatial_args = i // l, j // r
        permutate_i, permutate_j = intra_index_map.map_indices([warp_i, warp_j])
        new_index = (*spatial_args, permutate_i, permutate_j)
        return B[new_index]

    B_reindex = te.compute(
        (N, K),
        fcompute,
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
    func = func.with_attr("smooth_b", True)

    return tvm.IRModule.from_expr(func)


def matmul_nt_propagate_a_propagate_b(
    M,
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
):
    l = r = 16
    if in_dtype == "int8":
        l, r = 16, 32

    A = te.placeholder((M // l, K // r, l, r), name="A", dtype=in_dtype)
    B = te.placeholder((N // l, K // r, l, r), name="B", dtype=in_dtype)
    Bias = te.placeholder((N,), name="Bias", dtype=in_dtype)

    intra_index_map, _ = get_propagate_map(trans=False, dtype=in_dtype, matrix_name="A")

    def fcompute(i, j):
        warp_i, warp_j = i % l, j % r
        spatial_args = i // l, j // r
        permutate_i, permutate_j = intra_index_map.map_indices([warp_i, warp_j])
        new_index = (*spatial_args, permutate_i, permutate_j)
        return A[new_index]

    A_reindex = te.compute(
        (M, K),
        fcompute,
        name="A_reindex",
    )

    intra_index_map, _ = get_propagate_map(trans=True, dtype=in_dtype, matrix_name="B")

    def fcompute(i, j):
        warp_i, warp_j = i % l, j % r
        spatial_args = i // l, j // r
        permutate_i, permutate_j = intra_index_map.map_indices([warp_i, warp_j])
        new_index = (*spatial_args, permutate_i, permutate_j)
        return B[new_index]

    B_reindex = te.compute(
        (N, K),
        fcompute,
        name="B_reindex",
    )
    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(
            A_reindex[i, k].astype(accum_dtype) * B_reindex[j, k].astype(accum_dtype),
            axis=k,
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
    func = func.with_attr("smooth_a", True)
    func = func.with_attr("smooth_b", True)

    return tvm.IRModule.from_expr(func)


def _select_implementation_dyn_m(
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
    layout="nt",
    propagate_a=False,
    propagate_b=False,
):
    if layout == "nn":
        if propagate_a or propagate_b:
            raise ValueError(
                "Currently only support propagate_a=False and propagate_b=False for layout=nn"
            )
        return matmul_dyn_m(N, K, in_dtype, out_dtype, accum_dtype, with_bias, layout)
    elif layout == "nt":
        if propagate_a and propagate_b:
            return matmul_nt_propagate_a_propagate_b_dyn_m(
                N, K, in_dtype, out_dtype, accum_dtype, with_bias
            )
        elif propagate_a:
            return matmul_nt_propagate_a_dyn_m(
                N, K, in_dtype, out_dtype, accum_dtype, with_bias
            )
        elif propagate_b:
            return matmul_nt_propagate_b_dyn_m(
                N, K, in_dtype, out_dtype, accum_dtype, with_bias
            )
        else:
            return matmul_dyn_m(
                N, K, in_dtype, out_dtype, accum_dtype, with_bias, layout
            )


def select_implementation(
    M=None,
    N=16384,
    K=16384,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
    layout="nt",
    propagate_a=False,
    propagate_b=False,
):
    if not isinstance(M, int):
        return _select_implementation_dyn_m(
            N,
            K,
            in_dtype,
            out_dtype,
            accum_dtype,
            with_bias,
            layout,
            propagate_a,
            propagate_b,
        )
    if layout == "nn":
        if propagate_a or propagate_b:
            raise ValueError(
                "Currently only support propagate_a=False and propagate_b=False for layout=nn"
            )
        return matmul(M, N, K, in_dtype, out_dtype, accum_dtype, with_bias, layout)
    elif layout == "nt":
        if propagate_a and propagate_b:
            return matmul_nt_propagate_a_propagate_b(
                M, N, K, in_dtype, out_dtype, accum_dtype, with_bias
            )
        elif propagate_a:
            return matmul_nt_propagate_a(
                M, N, K, in_dtype, out_dtype, accum_dtype, with_bias
            )
        elif propagate_b:
            return matmul_nt_propagate_b(
                M, N, K, in_dtype, out_dtype, accum_dtype, with_bias
            )
        else:
            return matmul(M, N, K, in_dtype, out_dtype, accum_dtype, with_bias, layout)
    else:
        raise ValueError(f"Unsupported layout: {layout}")
