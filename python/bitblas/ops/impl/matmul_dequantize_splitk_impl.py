# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# pre-transformed tir expression of matmul
import tvm
from tvm import te
from bitblas.quantization import (_tir_packed_int_to_int_convert, _tir_packed_to_signed_convert,
                                  _tir_packed_to_unsigned_convert, _tir_u32_to_f4_to_f16,
                                  _tir_u8_to_f8_e4m3_to_f16)


def matmul_nt_dequantize_b(
    SplitK,
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
    with_zeros=False,
    group_size=-1,
    fast_decoding=False,
    with_bias=False,
    zeros_mode="original",
):
    assert bit in [1, 2, 4, 8], "Unsupported bit: {}".format(bit)
    if not isinstance(M, int):
        M = tvm.te.var("m")

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
            if bit == 8:
                # 8 bit does not need to be compressed
                w = B[n, k].astype(in_dtype)
            else:
                w = _tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
                    bit, B[n, k // n_float_per_elem], k % n_float_per_elem, dtype=in_dtype)
        elif source_format == "int":
            if bit == 1:
                # Dequantize int1 to -1 and 1. Without this step, the values would be 0 and 1, identical to uint1.
                w = _tir_packed_int_to_int_convert(storage_type, storage_nbit)(
                    bit, B[n, k // n_float_per_elem], k % n_float_per_elem, dtype=in_dtype)
            elif bit == 8:
                # 8 bit does not need to be compressed
                w = B[n, k].astype(in_dtype)
            else:
                w = _tir_packed_to_signed_convert(storage_type, storage_nbit)(
                    bit, B[n, k // n_float_per_elem], k % n_float_per_elem, dtype=in_dtype)
        elif source_format == "fp":
            w = _tir_u32_to_f4_to_f16(
                bit, B[n, k // n_float_per_elem], k % n_float_per_elem, dtype=in_dtype)
        elif source_format == "fp_e4m3":
            w = _tir_u8_to_f8_e4m3_to_f16(bit, B[n, k], dtype=in_dtype)
        elif source_format == "nf":
            w = LUT[_tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
                bit,
                B[n, k // n_float_per_elem],
                k % n_float_per_elem,
                dtype="int32",  # assume the index data type is int32
            )]
        else:
            raise ValueError("Unsupported source_format: {}".format(source_format))

        if not with_scaling:
            return w

        if not with_zeros:
            return w * Scale[n, k // group_size]

        return w

    B_decode = te.compute((N, K), decode_func, name="B_decode")
    # Describe the matrix multiplication in TE
    RK = K // SplitK
    k = te.reduce_axis((0, RK), name="k")
    C = te.compute(
        (SplitK, M, N),
        lambda sk, i, j: te.sum(
            A[i, sk * RK + k].astype(accum_dtype) * B_decode[j, sk * RK + k].astype(accum_dtype),
            axis=k),
        name="C",
    )
    D = te.compute((SplitK, M, N), lambda b, i, j: C[b, i, j].astype(out_dtype), name="D")
    args = [A, B]
    last_output = D
    if source_format == "nf":
        args.append(LUT)
    if with_scaling:
        args.append(Scale)
    if with_bias:
        E = te.compute((SplitK, M, N), lambda b, i, j: D[b, i, j] + Bias[j], name="E")
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
                "with_zeros": with_zeros,
                "zeros_mode": zeros_mode,
                "group_size": group_size,
            }
        },
    )
    return tvm.IRModule.from_expr(func)


def select_implementation(
    SplitK=1,
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
    with_zeros=False,
    group_size=-1,
    fast_decoding=False,
    with_bias=False,
    layout="nt",
    zeros_mode="original",
    propagate_a=False,
    propagate_b=False,
):
    if layout == "nn":
        raise ValueError(
            "Currently only support propagate_a=False and propagate_b=False for layout=nn in Dequantize Implementation"
        )
    elif layout == "nt":
        if propagate_a and propagate_b:
            raise ValueError("Currently only support propagate_a or propagate_b for layout=nt")
        elif propagate_a:
            raise ValueError("Currently only support propagate_a=False for layout=nt")
        elif propagate_b:
            raise ValueError("Currently only support propagate_b=False for layout=nt")
        else:
            return matmul_nt_dequantize_b(
                SplitK,
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
                with_zeros,
                group_size,
                fast_decoding,
                with_bias,
                zeros_mode,
            )
    else:
        raise ValueError(f"Unsupported layout: {layout}")
