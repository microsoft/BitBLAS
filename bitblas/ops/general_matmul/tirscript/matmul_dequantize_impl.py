# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# pre-transformed tir expression of matmul
from bitblas import tvm
from tvm import te, DataType
from tvm.tir import IndexMap
from bitblas.ops.operator import TransformKind
from bitblas.gpu.matmul_analysis import get_propagate_map
from bitblas.quantization import (
    _tir_packed_int_to_int_convert,
    _tir_packed_to_signed_convert,
    _tir_packed_to_unsigned_convert,
    _tir_u32_to_f4_to_f16,
    _tir_u8_to_f8_e4m3_to_f16,
    _tir_packed_to_unsigned_convert_with_zeros,
)
from typing import Union


class MatMulNTDequantizeEmitter:

    def __init__(
        self,
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
        propagate_a: TransformKind = TransformKind.NonTransform,
        propagate_b: TransformKind = TransformKind.NonTransform,
    ):
        self.M = self._validate_dimension(M, "M")
        self.N = N
        self.K = K
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.accum_dtype = accum_dtype
        self.bit = bit
        self.storage_dtype = storage_dtype
        self.source_format = source_format
        self.with_scaling = with_scaling
        self.with_zeros = with_zeros
        self.group_size = group_size if group_size != -1 else K
        self.fast_decoding = fast_decoding
        self.with_bias = with_bias
        self.zeros_mode = zeros_mode
        self.propagate_a = self._legalize_transform_kind(propagate_a)
        self.propagate_b = self._legalize_transform_kind(propagate_b)

        self._validate_bit()
        self._validate_layout()

    @staticmethod
    def _validate_dimension(dim, name):
        if not isinstance(dim, int):
            return tvm.te.var(name.lower())
        return dim

    def _validate_bit(self):
        if self.bit not in [1, 2, 4, 8]:
            raise ValueError(f"Unsupported bit: {self.bit}")

    def _validate_layout(self):
        # TODO: extend the dequantize operators into General Layout
        pass

    def _legalize_group_size(self):
        if self.group_size == -1:
            self.group_size = self.K

    def _legalize_transform_kind(self, propagate):
        if propagate is None:
            return TransformKind.NonTransform
        if isinstance(propagate, bool):
            return (TransformKind.IntraWarpTransform if propagate else TransformKind.NonTransform)
        elif isinstance(propagate, int):
            return TransformKind(propagate)

    def _create_placeholders(self):
        storage_dtype = self.storage_dtype
        storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
        in_dtype = self.in_dtype
        bit = self.bit
        l = r = 16  # noqa: E741
        if in_dtype in ["int8", "e4m3_float8", "e5m2_float8"]:
            l, r = 16, 32  # noqa: E741

        A = te.placeholder((self.M, self.K), name="A", dtype=in_dtype)
        B = te.placeholder((self.N, self.K // storage_nbit * bit), name="B", dtype=storage_dtype)
        if self.propagate_a:
            A = te.placeholder((self.M // l, self.K // r, l, r), name="A", dtype=in_dtype)
        if self.propagate_b:
            target_dtype = DataType(in_dtype)
            scaling_factor = 1
            if bit > 0 and bit < target_dtype.bits:
                scaling_factor = ((target_dtype.bits // bit) * DataType(storage_dtype).bits //
                                  target_dtype.bits)
            qr = r * bit // storage_nbit
            B = te.placeholder((self.N // l, (self.K // scaling_factor) // qr, l, qr),
                               name="B",
                               dtype=storage_dtype)

        LUT = te.placeholder((1 << bit,), name="LUT", dtype=in_dtype)
        Scale = te.placeholder((self.N, self.K // self.group_size), name="Scale", dtype=in_dtype)
        Zeros = te.placeholder((self.N, self.K // self.group_size), name="Zeros", dtype=in_dtype)
        QZeros = te.placeholder(((self.K // self.group_size), self.N // storage_nbit * bit),
                                name="QZeros",
                                dtype=self.storage_dtype)
        Bias = te.placeholder((self.N,), name="Bias", dtype=in_dtype)
        return A, B, LUT, Scale, Zeros, QZeros, Bias

    def _propagate_input(self, tensor, transform_kind=TransformKind.NonTransform, matrix_name="A"):
        if transform_kind == TransformKind.NonTransform:
            return tensor
        in_dtype = self.in_dtype
        l = r = 16  # noqa: E741
        if in_dtype in ["int8", "e4m3_float8", "e5m2_float8"]:
            l, r = 16, 32  # noqa: E741
        _, inversed_index_map = get_propagate_map(
            trans=False, dtype=in_dtype, matrix_name=matrix_name)

        def fcompute(i, j):
            warp_i, warp_j = i % l, j % r
            spatial_args = i // l, j // r
            if transform_kind >= TransformKind.IntraWarpTransform:
                warp_i, warp_j = inversed_index_map.map_indices([warp_i, warp_j])
            new_index = (*spatial_args, warp_i, warp_j)
            return tensor[new_index]

        return te.compute(
            (self.M, self.K),
            fcompute,
            name=f"{matrix_name}_reindex",
        )

    def _propagage_weight(self, tensor, transform_kind=TransformKind.NonTransform, matrix_name="B"):
        if transform_kind == TransformKind.NonTransform:
            return tensor
        in_dtype = self.in_dtype
        bit = self.bit
        storage_dtype = self.storage_dtype
        storage_nbit = int("".join(c for c in self.storage_dtype if c.isdigit()))

        l = r = 16  # noqa: E741
        if in_dtype in ["int8", "e4m3_float8", "e5m2_float8"]:
            l, r = 16, 32  # noqa: E741
        _, inversed_index_map = get_propagate_map(
            trans=True, dtype=in_dtype, matrix_name=matrix_name)
        target_dtype = DataType(in_dtype)
        scaling_factor = 1
        if bit > 0 and bit < target_dtype.bits:
            scaling_factor = ((target_dtype.bits // bit) * DataType(storage_dtype).bits //
                              target_dtype.bits)
            initial_indices = inversed_index_map.initial_indices
            scaling_final_indices = inversed_index_map.map_indices(
                initial_indices[:-1] + [initial_indices[-1] * scaling_factor])
            scaling_final_indices = scaling_final_indices[:-1] + [
                scaling_final_indices[-1] // scaling_factor
            ]
            inversed_index_map = IndexMap(
                initial_indices,
                scaling_final_indices,
                None,
            )

        qr = r * bit // storage_nbit

        def fcompute(i, j):
            warp_i, warp_j = i % l, j % qr
            spatial_args = i // l, j // qr
            if transform_kind >= TransformKind.IntraWarpTransform:
                warp_i, warp_j = inversed_index_map.map_indices([warp_i, warp_j])
            new_index = (*spatial_args, warp_i, warp_j)
            return tensor[new_index]

        return te.compute(
            (self.N, self.K // storage_nbit * bit),
            fcompute,
            name=f"{matrix_name}_reindex",
        )

    def _decode_func(self, B, LUT, Scale, Zeros, QZeros):
        bit = self.bit
        in_dtype = self.in_dtype
        storage_dtype = self.storage_dtype
        storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
        storage_type = str("".join(c for c in storage_dtype if not c.isdigit()))
        n_float_per_elem = storage_nbit // bit

        # TODO: Move the decode function into a more general place
        def decode(n, k):
            w = None
            if self.with_zeros and self.zeros_mode == "quantized":
                qzeros_dequantize = _tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
                    bit,
                    QZeros[k, n // n_float_per_elem],
                    n % n_float_per_elem,
                    dtype=self.storage_dtype,
                )
                w = _tir_packed_to_unsigned_convert_with_zeros(storage_type, storage_nbit)(
                    bit,
                    B[n, k // n_float_per_elem],
                    k % n_float_per_elem,
                    qzeros_dequantize,
                    dtype=in_dtype,
                )
            elif self.source_format == "uint":
                if bit == 8:
                    w = B[n, k].astype(in_dtype)
                w = _tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
                    bit, B[n, k // n_float_per_elem], k % n_float_per_elem, dtype=in_dtype)
            elif self.source_format == "int":
                if bit == 1:
                    w = _tir_packed_int_to_int_convert(storage_type, storage_nbit)(
                        bit, B[n, k // n_float_per_elem], k % n_float_per_elem, dtype=in_dtype)
                if bit == 8:
                    w = B[n, k].astype(in_dtype)
                w = _tir_packed_to_signed_convert(storage_type, storage_nbit)(
                    bit, B[n, k // n_float_per_elem], k % n_float_per_elem, dtype=in_dtype)
            elif self.source_format == "fp":
                w = _tir_u32_to_f4_to_f16(
                    bit, B[n, k // n_float_per_elem], k % n_float_per_elem, dtype=in_dtype)
            elif self.source_format == "fp_e4m3":
                w = _tir_u8_to_f8_e4m3_to_f16(bit, B[n, k], dtype=in_dtype)
            elif self.source_format == "nf":
                index = _tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
                    bit,
                    B[n, k // n_float_per_elem],
                    k % n_float_per_elem,
                    dtype="int32",
                )
                w = LUT[index]
            else:
                raise ValueError(f"Unsupported source_format: {self.source_format}")

            assert w is not None, "w is None"

            group_size = self.group_size
            zeros_mode = self.zeros_mode

            if not self.with_scaling:
                return w

            if not self.with_zeros:
                return w * Scale[n, k // group_size]

            if zeros_mode == "original":
                w = (w - Zeros[n, k // group_size]) * Scale[n, k // group_size]
            elif zeros_mode == "rescale":
                w = w * Scale[n, k // group_size] - Zeros[n, k // group_size]
            elif zeros_mode == "quantized":
                w = w * Scale[n, k // group_size]
            else:
                raise ValueError("Unsupported zeros_mode: {}".format(zeros_mode))

            return w

        return te.compute((self.N, self.K), decode, name="B_decode")

    def _compute_matmul(self, A, B_decode):
        k = te.reduce_axis((0, self.K), name="k")
        C = te.compute(
            (self.M, self.N),
            lambda i, j: te.sum(
                A[i, k].astype(self.accum_dtype) * B_decode[j, k].astype(self.accum_dtype), axis=k),
            name="C",
        )
        return C

    def _convert_dtype(self, tensor):
        if self.accum_dtype != self.out_dtype:
            return te.compute((self.M, self.N),
                              lambda i, j: tensor[i, j].astype(self.out_dtype),
                              name="D")
        return tensor

    def _apply_bias(self, tensor, Bias):
        if self.with_bias:
            return te.compute((self.M, self.N), lambda i, j: tensor[i, j] + Bias[j], name="E")
        return tensor

    def emit(self):
        A, B, LUT, Scale, Zeros, QZeros, Bias = self._create_placeholders()
        A_reindex = self._propagate_input(A, self.propagate_a, "A")
        B_reindex = self._propagage_weight(B, self.propagate_b, "B")

        B_decode = self._decode_func(B_reindex, LUT, Scale, Zeros, QZeros)
        C = self._compute_matmul(A_reindex, B_decode)
        D = self._convert_dtype(C)
        last_output = self._apply_bias(D, Bias)

        args = [A, B]
        if self.source_format == "nf":
            args.append(LUT)
        if self.with_scaling:
            args.append(Scale)
        if self.with_zeros:
            args.append(QZeros if self.zeros_mode == "quantized" else Zeros)
        if self.with_bias:
            args.append(Bias)
        args.append(last_output)

        func = te.create_prim_func(args).with_attr(
            "dequantize_info",
            {
                "B_decode": {
                    "decode_block": "B_decode",
                    "fast_decoding": self.fast_decoding,
                    "source_format": {
                        "bits": self.bit,
                        "format": self.source_format,
                    },
                    "storage_dtype": self.storage_dtype,
                    "target_format": self.in_dtype,
                    "with_zeros": self.with_zeros,
                    "zeros_mode": self.zeros_mode,
                    "with_scaling": self.with_scaling,
                    "group_size": self.group_size,
                }
            },
        )
        if self.propagate_a:
            func = func.with_attr("input_transform_kind", self.propagate_a.value)
        if self.propagate_b:
            func = func.with_attr("weight_transform_kind", self.propagate_b.value)
        return tvm.IRModule.from_expr(func)


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
    Zeros = te.placeholder((N, K // group_size), name="Zeros", dtype=in_dtype)
    QZeros = te.placeholder(((K // group_size), N // storage_nbit * bit),
                            name="QZeros",
                            dtype=storage_dtype)
    Bias = te.placeholder((N,), name="Bias", dtype=in_dtype)

    def qzeros_dequantize(k, n):
        return _tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
            bit,
            QZeros[k, n // n_float_per_elem],
            n % n_float_per_elem,
            dtype=storage_dtype,
        )

    Dequantize_qzeros = None
    if with_zeros and zeros_mode == "quantized":
        Dequantize_qzeros = te.compute(
            (K // group_size, N),
            qzeros_dequantize,
            name="Dequantize_zeros",
        )

    def decode_func(n, k):
        if with_zeros and zeros_mode == "quantized":
            assert Dequantize_qzeros is not None, "Dequantize_zeros is None"
            w = _tir_packed_to_unsigned_convert_with_zeros(storage_type, storage_nbit)(
                bit,
                B[n, k // n_float_per_elem],
                k % n_float_per_elem,
                Dequantize_qzeros[k // group_size, n],
                dtype=in_dtype,
            )
        elif source_format == "uint":
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

        if zeros_mode == "original":
            w = (w - Zeros[n, k // group_size]) * Scale[n, k // group_size]
        elif zeros_mode == "rescale":
            w = w * Scale[n, k // group_size] - Zeros[n, k // group_size]
        elif zeros_mode == "quantized":
            w = w * Scale[n, k // group_size]
        else:
            raise ValueError("Unsupported zeros_mode: {}".format(zeros_mode))

        return w

    B_decode = te.compute((N, K), decode_func, name="B_decode")
    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(
            A[i, k].astype(accum_dtype) * B_decode[j, k].astype(accum_dtype), axis=k),
        name="C",
    )

    last_output = C
    if accum_dtype != out_dtype:
        D = te.compute((M, N), lambda i, j: C[i, j].astype(out_dtype), name="D")
        last_output = D
    args = [A, B]
    if source_format == "nf":
        args.append(LUT)
    if with_scaling:
        args.append(Scale)
    if with_zeros:
        if zeros_mode == "quantized":
            args.append(QZeros)
        else:
            args.append(Zeros)
    if with_bias:
        last_output = te.compute((M, N), lambda i, j: last_output[i, j] + Bias[j], name="E")
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
    with_zeros=False,
    group_size=-1,
    fast_decoding=False,
    with_bias=False,
    zeros_mode="original",
    transform_kind: Union[int, TransformKind] = TransformKind.IntraWarpTransform,
):
    if isinstance(transform_kind, int):
        transform_kind = TransformKind(transform_kind)

    assert bit in [1, 2, 4, 8], "Unsupported bit: {}".format(bit)
    if not isinstance(M, int):
        M = tvm.te.var("m")

    l = r = 16  # noqa: E741
    if in_dtype in ["int8", "e4m3_float8", "e5m2_float8"]:
        l, r = 16, 32  # noqa: E741

    _, inverse_indexmap = get_propagate_map(trans=True, dtype=in_dtype, matrix_name="B")
    target_dtype = DataType(in_dtype)
    scaling_factor = 1
    if bit > 0 and bit < target_dtype.bits:
        scaling_factor = ((target_dtype.bits // bit) * DataType(storage_dtype).bits //
                          target_dtype.bits)
        initial_indices = inverse_indexmap.initial_indices
        scaling_final_indices = inverse_indexmap.map_indices(initial_indices[:-1] +
                                                             [initial_indices[-1] * scaling_factor])
        scaling_final_indices = scaling_final_indices[:-1] + [
            scaling_final_indices[-1] // scaling_factor
        ]
        inverse_indexmap = IndexMap(
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
    B = te.placeholder((N // l, (K // scaling_factor) // qr, l, qr), name="B", dtype=storage_dtype)
    LUT = te.placeholder((1 << bit,), name="LUT", dtype=in_dtype)
    Scale = te.placeholder((N, K // group_size), name="Scale", dtype=in_dtype)
    Zeros = te.placeholder((N, K // group_size), name="Zeros", dtype=in_dtype)
    Bias = te.placeholder((N,), name="Bias", dtype=in_dtype)

    def fcompute(i, j):
        warp_i, warp_j = i % l, j % qr
        spatial_args = i // l, j // qr
        if transform_kind >= TransformKind.IntraWarpTransform:
            warp_i, warp_j = inverse_indexmap.map_indices([warp_i, warp_j])
        new_index = (*spatial_args, warp_i, warp_j)
        return B[new_index]

    B_reindex = te.compute(
        (N, K // storage_nbit * bit),
        fcompute,
        name="B_reindex",
    )

    def decode_func(n, k):
        if source_format == "uint":
            if bit == 8:
                # 8 bit does not need to be compressed
                w = B_reindex[n, k].astype(in_dtype)
            else:
                w = _tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
                    bit,
                    B_reindex[n, k // n_float_per_elem],
                    k % n_float_per_elem,
                    dtype=in_dtype,
                )
        elif source_format == "int":
            if bit == 1:
                # Dequantize int1 to -1 and 1. Without this step, the values would be 0 and 1, identical to uint1.
                w = _tir_packed_int_to_int_convert(storage_type, storage_nbit)(
                    bit, B_reindex[n, k // n_float_per_elem], k % n_float_per_elem, dtype=in_dtype)
            elif bit == 8:
                # 8 bit does not need to be compressed
                w = B_reindex[n, k].astype(in_dtype)
            else:
                w = _tir_packed_to_signed_convert(storage_type, storage_nbit)(
                    bit,
                    B_reindex[n, k // n_float_per_elem],
                    k % n_float_per_elem,
                    dtype=in_dtype,
                )
        elif source_format == "fp":
            w = _tir_u32_to_f4_to_f16(
                bit,
                B_reindex[n, k // n_float_per_elem],
                k % n_float_per_elem,
                dtype=in_dtype,
            )
        elif source_format == "fp_e4m3":
            w = _tir_u8_to_f8_e4m3_to_f16(bit, B_reindex[n, k], dtype=in_dtype)
        elif source_format == "nf":
            w = LUT[_tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
                bit,
                B_reindex[n, k // n_float_per_elem],
                k % n_float_per_elem,
                dtype="int32",  # assume the index data type is int32
            )]
        else:
            raise ValueError("Unsupported source_format: {}".format(source_format))

        if not with_scaling:
            return w

        if not with_zeros:
            return w * Scale[n, k // group_size]

        if zeros_mode == "original":
            w = (w - Zeros[n, k // group_size]) * Scale[n, k // group_size]
        elif zeros_mode == "rescale":
            w = w * Scale[n, k // group_size] - Zeros[n, k // group_size]
        else:
            raise ValueError("Unsupported zeros_mode: {}".format(zeros_mode))

        return w

    B_decode = te.compute((N, K), decode_func, name="B_decode")

    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(
            A[i, k].astype(accum_dtype) * B_decode[j, k].astype(accum_dtype), axis=k),
        name="C",
    )
    last_output = C
    if accum_dtype != out_dtype:
        D = te.compute((M, N), lambda i, j: C[i, j].astype(out_dtype), name="D")
        last_output = D
    args = [A, B]
    if source_format == "nf":
        args.append(LUT)
    if with_scaling:
        args.append(Scale)
    if with_zeros:
        args.append(Zeros)
    if with_bias:
        last_output = te.compute((M, N), lambda i, j: last_output[i, j] + Bias[j], name="E")
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
                "with_zeros": with_zeros,
                "zeros_mode": zeros_mode,
                "with_scaling": with_scaling,
                "group_size": group_size,
            }
        },
    )
    func = func.with_attr("weight_transform_kind", transform_kind.value)
    return tvm.IRModule.from_expr(func)


def matmul_nt_dequantize_b_propagate_a_propagate_b(
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
    transform_kind_input: Union[int, TransformKind] = TransformKind.IntraWarpTransform,
    transform_kind_weight: Union[int, TransformKind] = TransformKind.IntraWarpTransform,
):
    if isinstance(transform_kind_input, int):
        transform_kind_input = TransformKind(transform_kind_input)
    if isinstance(transform_kind_weight, int):
        transform_kind_weight = TransformKind(transform_kind_weight)

    assert bit in [1, 2, 4, 8], "Unsupported bit: {}".format(bit)
    if not isinstance(M, int):
        M = tvm.te.var("m")

    l = r = 16  # noqa: E741
    if in_dtype in ["int8", "e4m3_float8", "e5m2_float8"]:
        l, r = 16, 32  # noqa: E741
    _, inversed_index_map = get_propagate_map(trans=False, dtype=in_dtype, matrix_name="A")
    A = te.placeholder((M // l, K // r, l, r), name="A", dtype=in_dtype)

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
    target_dtype = DataType(in_dtype)
    scaling_factor = 1
    if bit > 0 and bit < target_dtype.bits:
        scaling_factor = ((target_dtype.bits // bit) * DataType(storage_dtype).bits //
                          target_dtype.bits)
        initial_indices = inversed_index_map.initial_indices
        scaling_final_indices = inversed_index_map.map_indices(
            initial_indices[:-1] + [initial_indices[-1] * scaling_factor])
        scaling_final_indices = scaling_final_indices[:-1] + [
            scaling_final_indices[-1] // scaling_factor
        ]
        inversed_index_map = IndexMap(
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
    B = te.placeholder((N // l, (K // scaling_factor) // qr, l, qr), name="B", dtype=storage_dtype)
    LUT = te.placeholder((1 << bit,), name="LUT", dtype=in_dtype)
    Scale = te.placeholder((N, K // group_size), name="Scale", dtype=in_dtype)
    Zeros = te.placeholder((N, K // group_size), name="Zeros", dtype=in_dtype)
    Bias = te.placeholder((N,), name="Bias", dtype=in_dtype)

    def fcompute(i, j):
        warp_i, warp_j = i % l, j % qr
        spatial_args = i // l, j // qr
        if transform_kind_weight >= TransformKind.IntraWarpTransform:
            warp_i, warp_j = inversed_index_map.map_indices([warp_i, warp_j])
        new_index = (*spatial_args, warp_i, warp_j)
        return B[new_index]

    B_reindex = te.compute(
        (N, K // storage_nbit * bit),
        fcompute,
        name="B_reindex",
    )

    def decode_func(n, k):
        if source_format == "uint":
            if bit == 8:
                # 8 bit does not need to be compressed
                w = B_reindex[n, k].astype(in_dtype)
            else:
                w = _tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
                    bit,
                    B_reindex[n, k // n_float_per_elem],
                    k % n_float_per_elem,
                    dtype=in_dtype,
                )
        elif source_format == "int":
            # Dequantize int1 to -1 and 1. Without this step, the values would be 0 and 1, identical to uint1.
            if bit == 1:
                w = _tir_packed_int_to_int_convert(storage_type, storage_nbit)(
                    bit, B_reindex[n, k // n_float_per_elem], k % n_float_per_elem, dtype=in_dtype)
            elif bit == 8:
                # 8 bit does not need to be compressed
                w = B_reindex[n, k].astype(in_dtype)
            else:
                w = _tir_packed_to_signed_convert(storage_type, storage_nbit)(
                    bit,
                    B_reindex[n, k // n_float_per_elem],
                    k % n_float_per_elem,
                    dtype=in_dtype,
                )
        elif source_format == "fp":
            w = _tir_u32_to_f4_to_f16(
                bit,
                B_reindex[n, k // n_float_per_elem],
                k % n_float_per_elem,
                dtype=in_dtype,
            )
        elif source_format == "fp_e4m3":
            w = _tir_u8_to_f8_e4m3_to_f16(bit, B_reindex[n, k], dtype=in_dtype)
        elif source_format == "nf":
            w = LUT[_tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
                bit,
                B_reindex[n, k // n_float_per_elem],
                k % n_float_per_elem,
                dtype="int32",  # assume the index data type is int32
            )]
        else:
            raise ValueError("Unsupported source_format: {}".format(source_format))

        if not with_scaling:
            return w

        if not with_zeros:
            return w * Scale[n, k // group_size]

        if zeros_mode == "original":
            w = (w - Zeros[n, k // group_size]) * Scale[n, k // group_size]
        elif zeros_mode == "rescale":
            w = w * Scale[n, k // group_size] - Zeros[n, k // group_size]
        else:
            raise ValueError("Unsupported zeros_mode: {}".format(zeros_mode))

        return w

    B_decode = te.compute((N, K), decode_func, name="B_decode")

    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(
            A_reindex[i, k].astype(accum_dtype) * B_decode[j, k].astype(accum_dtype),
            axis=k,
        ),
        name="C",
    )
    last_output = C
    if accum_dtype != out_dtype:
        D = te.compute((M, N), lambda i, j: C[i, j].astype(out_dtype), name="D")
        last_output = D
    args = [A, B]
    if source_format == "nf":
        args.append(LUT)
    if with_scaling:
        args.append(Scale)
    if with_zeros:
        args.append(Zeros)
    if with_bias:
        last_output = te.compute((M, N), lambda i, j: last_output[i, j] + Bias[j], name="E")
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
                "with_zeros": with_zeros,
                "zeros_mode": zeros_mode,
                "with_scaling": with_scaling,
                "group_size": group_size,
            }
        },
    )
    func = func.with_attr("input_transform_kind", transform_kind_input.value)
    func = func.with_attr("weight_transform_kind", transform_kind_weight.value)
    return tvm.IRModule.from_expr(func)


# Should be refactored with Emitter
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
            return matmul_nt_dequantize_b_propagate_a_propagate_b(
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
                transform_kind_input=propagate_a,
                transform_kind_weight=propagate_b,
            )
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
                with_zeros,
                group_size,
                fast_decoding,
                with_bias,
                zeros_mode,
                transform_kind=propagate_b,
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
                with_zeros,
                group_size,
                fast_decoding,
                with_bias,
                zeros_mode,
            )
    else:
        raise ValueError(f"Unsupported layout: {layout}")
