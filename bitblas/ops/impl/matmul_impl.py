# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# pre-transformed tir expression of matmul
import tvm
from tvm import te
from bitblas.gpu.matmul_analysis import get_propagate_map
from bitblas.ops.operator import TransformKind


def matmul_nn(
    M,
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
):
    if not isinstance(M, int):
        M = tvm.te.var("m")
    A = te.placeholder((M, K), name="A", dtype=in_dtype)
    B = te.placeholder((K, N), name="B", dtype=in_dtype)
    Bias = te.placeholder((N,), name="Bias", dtype=in_dtype)

    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k].astype(accum_dtype) * B[k, j].astype(accum_dtype), axis=k),
        name="C",
    )
    last_output = C
    if accum_dtype != out_dtype:
        D = te.compute((M, N), lambda i, j: C[i, j].astype(out_dtype), name="D")
        last_output = D

    if with_bias:
        E = te.compute((M, N), lambda i, j: last_output[i, j] + Bias[j], name="E")
        last_output = E

    args = [A, B, Bias, last_output] if with_bias else [A, B, last_output]

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
    if not isinstance(M, int):
        M = tvm.te.var("m")
    A = te.placeholder((M, K), name="A", dtype=in_dtype)
    B = te.placeholder((N, K), name="B", dtype=in_dtype)
    Bias = te.placeholder((N,), name="Bias", dtype=in_dtype)

    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k].astype(accum_dtype) * B[j, k].astype(accum_dtype), axis=k),
        name="C",
    )
    last_output = C
    if accum_dtype != out_dtype:
        D = te.compute((M, N), lambda i, j: C[i, j].astype(out_dtype), name="D")
        last_output = D

    if with_bias:
        E = te.compute((M, N), lambda i, j: last_output[i, j] + Bias[j], name="E")
        last_output = E

    args = [A, B, Bias, last_output] if with_bias else [A, B, last_output]

    func = te.create_prim_func(args)

    return tvm.IRModule.from_expr(func)


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


def matmul_nt_propagate_a(
    M,
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
    transform_kind: TransformKind = TransformKind.IntraWarpTransform,
):
    if not isinstance(M, int):
        M = tvm.te.var("m")
    l = r = 16  # noqa: E741
    if in_dtype in ["int8", "e4m3_float8", "e5m2_float8"]:
        l, r = 16, 32  # noqa: E741

    _, inversed_index_map = get_propagate_map(trans=False, dtype=in_dtype, matrix_name="A")

    A = te.placeholder((M // l, K // r, l, r), name="A", dtype=in_dtype)
    B = te.placeholder((N, K), name="B", dtype=in_dtype)
    Bias = te.placeholder((N,), name="Bias", dtype=in_dtype)

    def fcompute(i, j):
        warp_i, warp_j = i % l, j % r
        spatial_args = i // l, j // r
        if transform_kind >= TransformKind.IntraWarpTransform:
            warp_i, warp_j = inversed_index_map.map_indices([warp_i, warp_j])
        new_index = (*spatial_args, warp_i, warp_j)
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
            A_reindex[i, k].astype(accum_dtype) * B[j, k].astype(accum_dtype), axis=k),
        name="C",
    )
    last_output = C
    if accum_dtype != out_dtype:
        D = te.compute((M, N), lambda i, j: C[i, j].astype(out_dtype), name="D")
        last_output = D

    if with_bias:
        E = te.compute((M, N), lambda i, j: last_output[i, j] + Bias[j], name="E")
        last_output = E

    args = [A, B, Bias, last_output] if with_bias else [A, B, last_output]

    func = te.create_prim_func(args)
    func = func.with_attr("input_transform_kind", transform_kind.value)

    return tvm.IRModule.from_expr(func)


def matmul_nt_propagate_b(
    M,
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
    transform_kind: TransformKind = TransformKind.IntraWarpTransform,
):
    if not isinstance(M, int):
        M = tvm.te.var("m")
    l = r = 16  # noqa: E741
    if in_dtype in ["int8", "e4m3_float8", "e5m2_float8"]:
        l, r = 16, 32  # noqa: E741

    _, inversed_index_map = get_propagate_map(trans=True, dtype=in_dtype, matrix_name="B")

    A = te.placeholder((M, K), name="A", dtype=in_dtype)
    B = te.placeholder((N // l, K // r, l, r), name="B", dtype=in_dtype)
    Bias = te.placeholder((N,), name="Bias", dtype=in_dtype)

    def fcompute(i, j):
        warp_i, warp_j = i % l, j % r
        spatial_args = i // l, j // r
        if transform_kind >= TransformKind.IntraWarpTransform:
            warp_i, warp_j = inversed_index_map.map_indices([warp_i, warp_j])
        new_index = (*spatial_args, warp_i, warp_j)
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
            A[i, k].astype(accum_dtype) * B_reindex[j, k].astype(accum_dtype), axis=k),
        name="C",
    )
    last_output = C
    if accum_dtype != out_dtype:
        D = te.compute((M, N), lambda i, j: C[i, j].astype(out_dtype), name="D")
        last_output = D

    if with_bias:
        E = te.compute((M, N), lambda i, j: last_output[i, j] + Bias[j], name="E")
        last_output = E

    args = [A, B, Bias, last_output] if with_bias else [A, B, last_output]

    func = te.create_prim_func(args)
    func = func.with_attr("weight_transform_kind", transform_kind.value)

    return tvm.IRModule.from_expr(func)


def matmul_nt_propagate_a_propagate_b(
    M,
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
    transform_kind_input: TransformKind = TransformKind.IntraWarpTransform,
    transform_kind_weight: TransformKind = TransformKind.IntraWarpTransform,
):
    if not isinstance(M, int):
        M = tvm.te.var("m")
    l = r = 16  # noqa: E741
    if in_dtype in ["int8", "e4m3_float8", "e5m2_float8"]:
        l, r = 16, 32  # noqa: E741

    A = te.placeholder((M // l, K // r, l, r), name="A", dtype=in_dtype)
    B = te.placeholder((N // l, K // r, l, r), name="B", dtype=in_dtype)
    Bias = te.placeholder((N,), name="Bias", dtype=in_dtype)

    _, inversed_index_map = get_propagate_map(trans=False, dtype=in_dtype, matrix_name="A")

    def fcompute(i, j):
        warp_i, warp_j = i % l, j % r
        spatial_args = i // l, j // r
        if transform_kind_input >= TransformKind.IntraWarpTransform:
            warp_i, warp_j = inversed_index_map.map_indices([warp_i, warp_j])
        new_index = (*spatial_args, warp_i, warp_j)
        return A[new_index]

    A_reindex = te.compute(
        (M, K),
        fcompute,
        name="A_reindex",
    )

    _, inversed_index_map = get_propagate_map(trans=True, dtype=in_dtype, matrix_name="B")

    def fcompute(i, j):
        warp_i, warp_j = i % l, j % r
        spatial_args = i // l, j // r
        if transform_kind_weight >= TransformKind.IntraWarpTransform:
            warp_i, warp_j = inversed_index_map.map_indices([warp_i, warp_j])
        new_index = (*spatial_args, warp_i, warp_j)
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

    args = [A, B, Bias, last_output] if with_bias else [A, B, last_output]

    func = te.create_prim_func(args)
    func = func.with_attr("input_transform_kind", transform_kind_input.value)
    func = func.with_attr("weight_transform_kind", transform_kind_weight.value)

    return tvm.IRModule.from_expr(func)


def select_implementation(
    M=None,
    N=16384,
    K=16384,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
    layout="nt",
    propagate_a: TransformKind = TransformKind.NonTransform,
    propagate_b: TransformKind = TransformKind.NonTransform,
):
    if layout == "nn":
        if propagate_a or propagate_b:
            raise ValueError(
                "Currently only support propagate_a=False and propagate_b=False for layout=nn")
        return matmul(M, N, K, in_dtype, out_dtype, accum_dtype, with_bias, layout)
    elif layout == "nt":
        if propagate_a and propagate_b:
            return matmul_nt_propagate_a_propagate_b(
                M,
                N,
                K,
                in_dtype,
                out_dtype,
                accum_dtype,
                with_bias,
                transform_kind_input=propagate_a,
                transform_kind_weight=propagate_b,
            )
        elif propagate_a:
            return matmul_nt_propagate_a(
                M,
                N,
                K,
                in_dtype,
                out_dtype,
                accum_dtype,
                with_bias,
                transform_kind=propagate_a,
            )
        elif propagate_b:
            return matmul_nt_propagate_b(
                M,
                N,
                K,
                in_dtype,
                out_dtype,
                accum_dtype,
                with_bias,
                transform_kind=propagate_b,
            )
        else:
            return matmul(M, N, K, in_dtype, out_dtype, accum_dtype, with_bias, layout)
    else:
        raise ValueError(f"Unsupported layout: {layout}")
