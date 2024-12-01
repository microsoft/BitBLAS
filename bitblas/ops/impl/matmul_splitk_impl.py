# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# pre-transformed tir expression of matmul
from bitblas import tvm
from tvm import te
from bitblas.base.operator_common import TransformKind


def matmul_nt(
    SplitK,
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
    RK = K // SplitK
    k = te.reduce_axis((0, RK), name="k")
    C = te.compute(
        (SplitK, M, N),
        lambda sk, i, j: te.sum(
            A[i, sk * RK + k].astype(accum_dtype) * B[j, sk * RK + k].astype(accum_dtype), axis=k),
        name="C",
    )
    last_output = C
    if accum_dtype != out_dtype:
        D = te.compute((SplitK, M, N), lambda b, i, j: C[b, i, j].astype(out_dtype), name="D")
        last_output = D

    if with_bias:
        E = te.compute((SplitK, M, N), lambda b, i, j: last_output[b, i, j] + Bias[j], name="E")
        last_output = E

    args = [A, B, Bias, last_output] if with_bias else [A, B, last_output]

    func = te.create_prim_func(args)

    return tvm.IRModule.from_expr(func)


def matmul(
    SplitK,
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
        raise ValueError("Currently only support layout=nt")
    return matmul_nt(SplitK, M, N, K, in_dtype, out_dtype, accum_dtype, with_bias)


def select_implementation(
    SplitK=1,
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
            raise ValueError("Currently only support propagate_a or propagate_b for layout=nt")
        elif propagate_a:
            raise ValueError("Currently only support propagate_a=False for layout=nt")
        elif propagate_b:
            raise ValueError("Currently only support propagate_b=False for layout=nt")
        else:
            return matmul(SplitK, M, N, K, in_dtype, out_dtype, accum_dtype, with_bias, layout)
    else:
        raise ValueError(f"Unsupported layout: {layout}")
