# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm as tvm
from bitblas import tilelang as tilelang
from functools import reduce
from typing import Optional, List
import tilelang.language as T
from tvm import DataType
from tvm.tir import PrimFunc

from dataclasses import dataclass
from bitblas.tl.base_hint import BaseTLHint
from bitblas.base.roller.hint import Hint
from .matmul_dequantize_simt import MatmulDequantizeSIMTBaseScheduler
from bitblas.quantization import (
    _tir_packed_to_unsigned_convert,)


@dataclass
class GemvDequantizeSIMTScheduler(MatmulDequantizeSIMTBaseScheduler):
    # Fine-grained matrix multiplication scheduler
    # Allows for more detailed configuration.

    # Default Hint Configuration
    n_partition: int = 8
    reduce_thread: int = 32

    class TLHint(BaseTLHint):

        hint_type: str = "GemvDequantizeSIMTScheduler"

        def __init__(self):
            super().__init__()

        @classmethod
        def from_roller_hint(cls, hint: Hint):
            tl_hint = cls()
            for key, value in hint.__dict__.items():
                setattr(tl_hint, key, value)

            def prod(iterable):
                return reduce(lambda x, y: x * y, iterable, 1)

            n_partition = int(prod(hint.thread))
            reduce_thread = int(prod(hint.reduce_thread))

            tl_hint.n_partition = n_partition
            tl_hint.reduce_thread = reduce_thread

            return tl_hint

        def get_config_params(self):
            return {
                "n_partition": self.n_partition,
                "reduce_thread": self.reduce_thread,
            }

        def __repr__(self):
            return ("{"
                    f"n_partition: {self.n_partition}, "
                    f"reduce_thread: {self.reduce_thread}, "
                    "}")

    def get_hint_type(self):
        return self.TLHint.hint_type

    def serialize_hints_to_configs(self, hints: List[Hint]):
        configs = []
        for hint in hints:
            config = self.TLHint.from_roller_hint(hint)
            configs.append(config)
        return configs

    def with_default_config(self) -> PrimFunc:
        n_partition = getattr(self, "n_partition", 8)
        reduce_thread = getattr(self, "reduce_thread", 16)

        return self.apply_config(
            n_partition=n_partition,
            reduce_thread=reduce_thread,
        )

    def apply_config(
        self,
        n_partition: Optional[int] = None,
        reduce_thread: Optional[int] = None,
    ):
        assert n_partition is not None, "n_partition must be provided"
        assert reduce_thread is not None, (
            "reduce_thread must be provided currently, as related bitblas.gpu.gemv.GEMV"
            "sch_outer_reduction_with_config is not implemented")

        M = self.maybe_dynamic(self.M, "m")
        N, K = self.N, self.K
        assert isinstance(N, int) and isinstance(K, int), "Do not support dynamic N and K Currently"

        trans_A, trans_B = self.trans_A, self.trans_B

        assert trans_A is False, "Dequantize only implement for trans_A=False currently"
        assert trans_B is True, "Dequantize only implement for trans_B=TRue currently"

        in_dtype, out_dtype, accum_dtype = (
            self.in_dtype,
            self.out_dtype,
            self.accum_dtype,
        )
        fast_decoding = self.fast_decoding
        with_bias = self.with_bias

        num_bits = self.num_bits
        storage_dtype = self.storage_dtype
        source_format = self.source_format
        storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
        num_elems_per_byte = self.num_elems_per_byte

        MAX_TRANSACTION_SIZE_IN_BITS = 128
        micro_size_k = MAX_TRANSACTION_SIZE_IN_BITS // DataType(in_dtype).bits
        micro_size_k_compressed = micro_size_k // num_elems_per_byte
        block_N = n_partition
        block_K = reduce_thread * micro_size_k

        group_size = self.group_size
        if group_size == -1:
            group_size = K

        A_shape = (M, K)
        B_shape = (N, K // storage_nbit * num_bits)
        LUT_shape = (1 << num_bits,)
        Scale_shape = (N, K // group_size)
        Zeros_shape = (N, K // group_size)
        Qzeros_shape = ((K // group_size), N // storage_nbit * num_bits)
        C_shape = (M, N)
        Bias_shape = (N,)

        dp4a_size = 4
        use_dp4a = in_dtype == "int8" and accum_dtype == "int32"

        local_scale_size = max(1, micro_size_k // group_size)
        local_zeros_size = max(1, micro_size_k // group_size)
        local_qzeros_size = max(1, micro_size_k // group_size)

        import_source: Optional[str] = None
        func_name: str = ""
        if fast_decoding is True:
            # Lazy import to decrease the startup time
            # as intrin registry may take a while to load
            from bitblas.gpu.intrin.lop3 import get_lop3_intrin_group

            lop3_intrin_info = get_lop3_intrin_group(
                out_dtype=in_dtype,
                source_format=source_format,
                source_bit=num_bits,
                storage_dtype=storage_dtype,
                with_scaling=self.with_scaling,
                with_zeros=self.with_zeros,
                zeros_mode=self.zeros_mode,
            )
            import_source = lop3_intrin_info["c_source"]
            func_name = lop3_intrin_info["func_name"]
            assert import_source is not None, "lop3_intrin_info is not found"
            assert func_name is not None, "lop3_intrin_info is not found"
            import_source = self.common_header + import_source

        @T.prim_func
        def main(
                A: T.Buffer(A_shape, in_dtype),
                B: T.Buffer(B_shape, storage_dtype),
                LUT: T.Buffer(LUT_shape, in_dtype),
                Scale: T.Buffer(Scale_shape, in_dtype),
                Qzeros: T.Buffer(Qzeros_shape, storage_dtype),
                Zeros: T.Buffer(Zeros_shape, in_dtype),
                Bias: T.Buffer(Bias_shape, in_dtype),
                C: T.Buffer(C_shape, out_dtype),
        ):
            with T.Kernel(
                    T.ceildiv(N, n_partition),
                    M,
                    threads=(reduce_thread, n_partition),
            ) as (
                    bx,
                    by,
            ):
                A_local = T.alloc_local((micro_size_k,), in_dtype)
                B_quant_local = T.alloc_local([micro_size_k_compressed], storage_dtype)
                scale_local = T.alloc_local([local_scale_size], in_dtype)
                zeros_local = T.alloc_local([local_zeros_size], in_dtype)
                dequant_qzeros_local = T.alloc_local([local_qzeros_size], storage_dtype)
                B_dequantize_local = T.alloc_local([micro_size_k], in_dtype)
                accum_res = T.alloc_local((1,), accum_dtype)
                reduced_accum_res = T.alloc_local((1,), accum_dtype)

                kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
                ni = T.thread_binding(0, n_partition, thread="threadIdx.y")

                T.import_source(import_source)

                T.clear(accum_res)
                for ko in T.serial(T.ceildiv(K, block_K)):
                    for v in T.vectorized(micro_size_k):
                        A_local[v] = A[by, ko * block_K + kr * micro_size_k + v]

                    for v in T.vectorized(micro_size_k_compressed):
                        B_quant_local[v] = B[
                            bx * n_partition + ni,
                            ko * (reduce_thread * micro_size_k_compressed) +
                            kr * micro_size_k_compressed + v,
                        ]

                    self.dequantize(
                        B_quant_local,
                        scale_local,
                        zeros_local,
                        dequant_qzeros_local,
                        B_dequantize_local,
                        LUT,
                        Scale,
                        Zeros,
                        Qzeros,
                        micro_size_k,
                        bx,
                        ni,
                        kr,
                        ko,
                        block_N,
                        block_K,
                        fast_decoding,
                        func_name,
                    )

                    if use_dp4a:
                        for ki in T.serial(micro_size_k // dp4a_size):
                            T.dp4a(
                                A_local[ki * dp4a_size],
                                B_dequantize_local[ki * dp4a_size],
                                accum_res[0],
                            )
                    else:
                        for ki in T.serial(micro_size_k):
                            accum_res[0] += A_local[ki] * B_dequantize_local[ki]

                with T.attr(
                        T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
                        "reduce_scope",
                        T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1),
                            accum_res[0],
                            True,
                            reduced_accum_res[0],
                            kr,
                            dtype="handle",
                        ))
                if kr == 0:
                    if with_bias:
                        C[by, bx * n_partition + ni] = (
                            reduced_accum_res[0] + Bias[bx * n_partition + ni])
                    else:
                        C[by, bx * n_partition + ni] = reduced_accum_res[0]

        return self.post_process(main)

    # GEMV Normal Dequant
    def _normal_dequant(
        self,
        compressed_weight_local: T.Buffer,
        scale_local: T.Buffer,
        zeros_local: T.Buffer,
        dequant_qzeros_local: T.Buffer,
        dequant_weight_local: T.Buffer,
        lut_buffer: T.Buffer,
        scale_buffer: T.Buffer,
        zeros_buffer: T.Buffer,
        qzeros_buffer: T.Buffer,
        local_size: int,
        pid_n: T.Var,
        ni: T.Var,
        kr: T.Var,
        k: T.Var,
        stride_n: int,
        stride_k: int,
    ):
        num_elems_per_byte = self.num_elems_per_byte
        with_scaling = self.with_scaling
        with_zeros = self.with_zeros
        zeros_mode = self.zeros_mode
        num_bits = self.num_bits
        in_dtype = self.in_dtype
        group_size = self.group_size
        storage_dtype = self.storage_dtype
        source_format = self.source_format
        is_lut = source_format == "nf"
        storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
        storage_type = str("".join(c for c in storage_dtype if not c.isdigit()))
        (local_scale_size,) = scale_local.shape
        (local_zeros_size,) = zeros_local.shape
        (local_qzeros_size,) = dequant_qzeros_local.shape

        @T.macro
        def _normal_dequant_impl(
            compressed_weight_local: T.Buffer,
            dequant_weight_local: T.Buffer,
            lut_buffer: T.Buffer,
            scale_buffer: T.Buffer,
            zeros_buffer: T.Buffer,
            qzeros_buffer: T.Buffer,
        ):
            if is_lut:
                for v in T.serial(0, local_size):
                    index = _tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
                        num_bits,
                        compressed_weight_local[v // num_elems_per_byte],
                        v % num_elems_per_byte,
                        "int32"  # default index dtype
                    )
                    dequant_weight_local[v] = lut_buffer[index]
            else:
                if with_scaling:
                    for v in T.vectorized(0, local_scale_size):
                        vi = ni
                        vj = kr * local_size + v
                        scale_local[v] = scale_buffer[
                            pid_n * stride_n + vi,
                            (k * stride_k + vj) // group_size,
                        ]

                if with_scaling and with_zeros:
                    if zeros_mode in ["original", "rescale"]:
                        for v in T.vectorized(0, local_zeros_size):
                            vi = ni
                            vj = kr * local_size + v
                            zeros_local[v] = zeros_buffer[
                                pid_n * stride_n + vi,
                                (k * stride_k + vj) // group_size,
                            ]
                    elif zeros_mode == "quantized":
                        for v in T.vectorized(0, local_qzeros_size):
                            vi = ni
                            vj = kr * local_size + v
                            dequant_qzeros_local[v] = _tir_packed_to_unsigned_convert(
                                storage_type, storage_nbit)(
                                    num_bits,
                                    qzeros_buffer[
                                        (k * stride_k + vj) // group_size,
                                        (pid_n * stride_n + vi) // num_elems_per_byte,
                                    ],
                                    (pid_n * stride_n + vi) % num_elems_per_byte,
                                    dtype=storage_dtype,
                                )
                    else:
                        raise ValueError(f"Unsupported zeros_mode: {zeros_mode}")

                for v in T.serial(0, local_size):
                    if not with_scaling:
                        dequant_weight_local[v] = self._decode_func(
                            num_bits,
                            compressed_weight_local[v // num_elems_per_byte],
                            v % num_elems_per_byte,
                            dtype=in_dtype,
                        )
                    elif not with_zeros:
                        dequant_weight_local[v] = (
                            self._decode_func(
                                num_bits,
                                compressed_weight_local[v // num_elems_per_byte],
                                v % num_elems_per_byte,
                                dtype=in_dtype,
                            ) * scale_local[v // group_size])
                    elif zeros_mode == "original":
                        dequant_weight_local[v] = (self._decode_func(
                            num_bits,
                            compressed_weight_local[v // num_elems_per_byte],
                            v % num_elems_per_byte,
                            dtype=in_dtype,
                        ) - zeros_local[v // group_size]) * scale_local[v // group_size]
                    elif zeros_mode == "rescale":
                        dequant_weight_local[v] = (
                            self._decode_func(
                                num_bits,
                                compressed_weight_local[v // num_elems_per_byte],
                                v % num_elems_per_byte,
                                dtype=in_dtype,
                            ) * scale_local[v // group_size] - zeros_local[v // group_size])
                    elif zeros_mode == "quantized":
                        dequant_weight_local[v] = (self._decode_func(
                            num_bits,
                            compressed_weight_local[v // num_elems_per_byte],
                            v % num_elems_per_byte,
                            zero=dequant_qzeros_local[v // group_size],
                            dtype=in_dtype,
                        )) * scale_local[v // group_size]
                    else:
                        raise ValueError(f"Unsupported zeros_mode: {zeros_mode}")

        return _normal_dequant_impl(
            compressed_weight_local,
            dequant_weight_local,
            lut_buffer,
            scale_buffer,
            zeros_buffer,
            qzeros_buffer,
        )

    # GEMV Fast Dequant
    def _normal_fast_dequant(
        self,
        compressed_weight_local: T.Buffer,
        scale_local: T.Buffer,
        zeros_local: T.Buffer,
        dequant_qzeros_local: T.Buffer,
        dequant_weight_local: T.Buffer,
        scale_buffer: T.Buffer,
        zeros_buffer: T.Buffer,
        qzeros_buffer: T.Buffer,
        func_name: str,
        local_size: int,
        pid_n: T.Var,
        ni: T.Var,
        kr: T.Var,
        k: T.Var,
        stride_n: int,
        stride_k: int,
    ):
        num_elems_per_byte = self.num_elems_per_byte
        with_scaling = self.with_scaling
        with_zeros = self.with_zeros
        zeros_mode = self.zeros_mode
        num_bits = self.num_bits
        in_dtype = self.in_dtype
        group_size = self.group_size
        storage_dtype = self.storage_dtype
        storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
        storage_type = str("".join(c for c in storage_dtype if not c.isdigit()))
        (local_scale_size,) = scale_local.shape
        (local_zeros_size,) = zeros_local.shape
        (local_qzeros_size,) = dequant_qzeros_local.shape

        @T.macro
        def _normal_fast_dequant_impl(
            compressed_weight_local: T.Buffer,
            dequant_weight_local: T.Buffer,
            scale_buffer: T.Buffer,
            zeros_buffer: T.Buffer,
            qzeros_buffer: T.Buffer,
        ):
            if with_scaling:
                for v in T.vectorized(0, local_scale_size):
                    vi = ni
                    vj = kr * local_size + v
                    scale_local[v] = scale_buffer[
                        pid_n * stride_n + vi,
                        (k * stride_k + vj) // group_size,
                    ]

            if with_scaling and with_zeros:
                if zeros_mode in ["original", "rescale"]:
                    for v in T.vectorized(0, local_zeros_size):
                        vi = ni
                        vj = kr * local_size + v
                        zeros_local[v] = zeros_buffer[
                            pid_n * stride_n + vi,
                            (k * stride_k + vj) // group_size,
                        ]
                elif zeros_mode == "quantized":
                    for v in T.vectorized(0, local_qzeros_size):
                        vi = ni
                        vj = kr * local_size + v
                        dequant_qzeros_local[v] = _tir_packed_to_unsigned_convert(
                            storage_type, storage_nbit)(
                                num_bits,
                                qzeros_buffer[
                                    (k * stride_k + vj) // group_size,
                                    (pid_n * stride_n + vi) // num_elems_per_byte,
                                ],
                                (pid_n * stride_n + vi) % num_elems_per_byte,
                                dtype=storage_dtype,
                            )
                else:
                    raise ValueError(f"Unsupported zeros_mode: {zeros_mode}")

            if not with_scaling:
                T.call_extern(
                    func_name,
                    T.address_of(compressed_weight_local[0]),
                    T.address_of(dequant_weight_local[0]),
                    dtype=in_dtype,
                )
            elif not with_zeros:
                T.call_extern(
                    func_name,
                    T.address_of(compressed_weight_local[0]),
                    T.address_of(dequant_weight_local[0]),
                    T.address_of(scale_local[0]),
                    dtype=in_dtype,
                )
            elif zeros_mode in ["original", "rescale"]:
                T.call_extern(
                    func_name,
                    T.address_of(compressed_weight_local[0]),
                    T.address_of(dequant_weight_local[0]),
                    T.address_of(scale_local[0]),
                    T.address_of(zeros_local[0]),
                    dtype=in_dtype,
                )
            elif zeros_mode == "quantized":
                T.call_extern(
                    func_name,
                    T.address_of(compressed_weight_local[0]),
                    T.address_of(dequant_weight_local[0]),
                    T.address_of(scale_local[0]),
                    T.address_of(dequant_qzeros_local[0]),
                    8,
                    dtype=in_dtype,
                )

        return _normal_fast_dequant_impl(
            compressed_weight_local,
            dequant_weight_local,
            scale_buffer,
            zeros_buffer,
            qzeros_buffer,
        )

    def dequantize(
        self,
        compressed_weight_local: T.Buffer,
        scale_local: T.Buffer,
        zeros_local: T.Buffer,
        dequant_qzeros_local: T.Buffer,
        dequant_weight_local: T.Buffer,
        lut_buffer: T.Buffer,
        scale_buffer: T.Buffer,
        zeros_buffer: T.Buffer,
        qzeros_buffer: T.Buffer,
        local_size: int,
        pid_n: T.Var,
        ni: T.Var,
        kr: T.Var,
        k: T.Var,
        stride_n: int,
        stride_k: int,
        fast_decoding: bool = False,
        func_name: str = "",
    ):
        if fast_decoding is True:
            return self._normal_fast_dequant(
                compressed_weight_local,
                scale_local,
                zeros_local,
                dequant_qzeros_local,
                dequant_weight_local,
                scale_buffer,
                zeros_buffer,
                qzeros_buffer,
                func_name,
                local_size,
                pid_n,
                ni,
                kr,
                k,
                stride_n,
                stride_k,
            )
        else:
            return self._normal_dequant(
                compressed_weight_local,
                scale_local,
                zeros_local,
                dequant_qzeros_local,
                dequant_weight_local,
                lut_buffer,
                scale_buffer,
                zeros_buffer,
                qzeros_buffer,
                local_size,
                pid_n,
                ni,
                kr,
                k,
                stride_n,
                stride_k,
            )
