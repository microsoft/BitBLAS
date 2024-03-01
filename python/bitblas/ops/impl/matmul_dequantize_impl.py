# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# pre-transformed tir expression of matmul
import tvm
from tvm.script import tir as T
from tvm import te, DataType
from tvm.tir import IndexMap
from bitblas.gpu.matmul_analysis import get_propagate_map
from bitblas.quantization import (
    _tir_packed_to_signed_convert,
    _tir_packed_to_unsigned_convert,
)


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
    storage_type = str("".join(c for c in storage_dtype if not c.isdigit()))
    n_float_per_elem = storage_nbit // bit
    if group_size == -1:
        group_size = K
    A = te.placeholder((M, K), name="A", dtype=in_dtype)
    B = te.placeholder((N, K // storage_nbit * bit), name="B", dtype=storage_dtype)
    LUT = te.placeholder((1 << bit,), name="LUT", dtype=in_dtype)
    Scale = te.placeholder((N, K // group_size), name="Scale", dtype=in_dtype)
    Bias = te.placeholder((N,), name="Bias", dtype=in_dtype)

    def decode_func(n, k):
        if source_format == "uint":
            w = _tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
                bit, B[n, k // n_float_per_elem], k % n_float_per_elem, dtype=in_dtype
            )
        elif source_format == "int":
            w = _tir_packed_to_signed_convert(storage_type, storage_nbit)(
                bit, B[n, k // n_float_per_elem], k % n_float_per_elem, dtype=in_dtype
            )
        elif source_format == "af":
            w = LUT[
                _tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
                    bit,
                    B[n, k // n_float_per_elem],
                    k % n_float_per_elem,
                    dtype="int32",  # assume the index data type is int32
                )
            ]
        else:
            raise ValueError("Unsupported source_format: {}".format(source_format))

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
    if source_format == "af":
        args.append(LUT)
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
    l = r = 16
    if in_dtype == "int8":
        l, r = 16, 32

    intra_index_map, _ = get_propagate_map(trans=True, dtype=in_dtype, matrix_name="B")
    target_dtype = DataType(in_dtype)
    scaling_factor = 1
    if bit > 0 and bit < target_dtype.bits:
        scaling_factor = (
            (target_dtype.bits // bit)
            * DataType(storage_dtype).bits
            // target_dtype.bits
        )
        initial_indices = intra_index_map.initial_indices
        scaling_final_indices = intra_index_map.map_indices(
            initial_indices[:-1] + [initial_indices[-1] * scaling_factor]
        )
        scaling_final_indices = scaling_final_indices[:-1] + [
            scaling_final_indices[-1] // scaling_factor
        ]
        intra_index_map = IndexMap(
            initial_indices,
            scaling_final_indices,
            None,
        )

    storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
    storage_type = str("".join(c for c in storage_dtype if not c.isdigit()))
    n_float_per_elem = storage_nbit // bit
    if group_size == -1:
        group_size = K
    qr = r * bit // storage_nbit
    A = te.placeholder((M, K), name="A", dtype=in_dtype)
    B = te.placeholder(
        (N // l, (K // scaling_factor) // qr, l, qr), name="B", dtype=storage_dtype
    )
    LUT = te.placeholder((1 << bit,), name="LUT", dtype=in_dtype)
    Scale = te.placeholder((N, K // group_size), name="Scale", dtype=in_dtype)
    Bias = te.placeholder((N,), name="Bias", dtype=in_dtype)

    def fcompute(i, j):
        warp_i, warp_j = i % l, j % qr
        spatial_args = i // l, j // qr
        permutate_i, permutate_j = intra_index_map.map_indices([warp_i, warp_j])
        new_index = (*spatial_args, permutate_i, permutate_j)
        return B[new_index]

    B_reindex = te.compute(
        (N, K // storage_nbit * bit),
        fcompute,
        name="B_reindex",
    )

    def decode_func(n, k):
        if source_format == "uint":
            w = _tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
                bit,
                B_reindex[n, k // n_float_per_elem],
                k % n_float_per_elem,
                dtype=in_dtype,
            )
        elif source_format == "int":
            w = _tir_packed_to_signed_convert(storage_type, storage_nbit)(
                bit,
                B_reindex[n, k // n_float_per_elem],
                k % n_float_per_elem,
                dtype=in_dtype,
            )
        elif source_format == "af":
            w = LUT[
                _tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
                    bit,
                    B_reindex[n, k // n_float_per_elem],
                    k % n_float_per_elem,
                    dtype="int32",  # assume the index data type is int32
                )
            ]
        else:
            raise ValueError("Unsupported source_format: {}".format(source_format))

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
    if source_format == "af":
        args.append(LUT)
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
    func = func.with_attr("smooth_b", True)
    return tvm.IRModule.from_expr(func)


def select_implementation(
    M=None,
    N=1024,
    K=1024,
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
    layout="nt",
    propagate_a=False,
    propagate_b=False,
):
    if not isinstance(M, int):
        raise ValueError("Currently do not implement with dynamic symbolic")
    if layout == "nn":
        raise ValueError(
            "Currently only support propagate_a=False and propagate_b=False for layout=nn in Dequantize Implementation"
        )
    elif layout == "nt":
        if propagate_a and propagate_b:
            raise NotImplementedError
        elif propagate_a:
            raise NotImplementedError
        elif propagate_b:
            return matmul_nt_dequantize_b_propagate_b(
                M,
                N,
                K,
                in_dtype,
                out_dtype,
                accum_dtype,
                bit,
                storage_dtype,
                source_format,
                with_scaling,
                group_size,
                fast_decoding,
                with_bias,
            )
        else:
            return matmul_nt_dequantize_b(
                M,
                N,
                K,
                in_dtype,
                out_dtype,
                accum_dtype,
                bit,
                storage_dtype,
                source_format,
                with_scaling,
                group_size,
                fast_decoding,
                with_bias,
            )
    else:
        raise ValueError(f"Unsupported layout: {layout}")
