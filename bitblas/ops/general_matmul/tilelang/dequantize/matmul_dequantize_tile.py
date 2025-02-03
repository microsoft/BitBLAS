# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Tile represents Tile Library

from bitblas import tvm as tvm
from tvm import DataType
from bitblas import tilelang as tilelang
import tilelang.language as T
from typing import Optional, List
from bitblas.base.arch import TileDevice
from bitblas.base.roller.hint import Hint
from bitblas.base.roller.rasterization import NoRasterization
from bitblas.base.utils import get_roller_hints_from_func
from dataclasses import dataclass
from bitblas.ops.general_matmul.tirscript import (
    matmul_dequantize_select_implementation,)
from bitblas.base.operator_common import QuantizationMemoryStage
from bitblas.tl.base_hint import BaseTLHint
from bitblas.quantization import (
    _tir_packed_int_to_int_convert,
    _tir_packed_to_signed_convert,
    _tir_packed_to_unsigned_convert,
    _tir_packed_to_fp4_to_f16,
    _tir_u8_to_f8_e4m3_to_f16,
    _tir_packed_to_unsigned_convert_with_zeros,
)

from .base import MatmulDequantizeBaseParams

# GPU warp configuration for NVIDIA GPUs
warp_size = 32


@dataclass
class MatmulDequantizeBaseScheduler(MatmulDequantizeBaseParams):

    def get_roller_configs(self, arch: TileDevice = None, topk: int = 10):
        layout = f"{'t' if self.trans_A else 'n'}{'t' if self.trans_B else 'n'}"

        # Simple TIR Compute Expression
        ir_module = matmul_dequantize_select_implementation(
            M=self.M,
            N=self.N,
            K=self.K,
            in_dtype=self.in_dtype,
            out_dtype=self.out_dtype,
            accum_dtype=self.accum_dtype,
            layout=layout,
            bit=self.num_bits,
            storage_dtype=self.storage_dtype,
            source_format=self.source_format,
            with_scaling=self.with_scaling,
            with_zeros=self.with_zeros,
            group_size=self.group_size,
            fast_decoding=self.fast_decoding,
            with_bias=self.with_bias,
            zeros_mode=self.zeros_mode,
        )

        roller_hints = get_roller_hints_from_func(
            ir_module,
            arch,
            topk,
            tensorcore_only=True,
            allow_gemv=True,
        )

        if roller_hints is None:
            raise ValueError("No Roller Hints Found for TensorCore Scheduling")

        return self.serialize_hints_to_configs(roller_hints)

    def get_hardware_aware_configs(self, arch: TileDevice = None, topk=10):
        if arch is None:
            arch = self.arch
        return self.get_roller_configs(arch, topk)

    # check if required shared memory cache
    def check_require_cache(self) -> bool:
        with_bias = self.with_bias

        conditions: List[bool] = []
        conditions.append(False)
        # Bias Add should be done in shared memory
        conditions.append(with_bias)
        return any(conditions)  # Always set to False Currently

    @property
    def _decode_func(self):
        with_zeros = self.with_zeros
        zeros_mode = self.zeros_mode
        storage_dtype = self.storage_dtype

        in_dtype = self.in_dtype
        source_format = self.source_format
        storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
        storage_type = str("".join(c for c in storage_dtype if not c.isdigit()))
        num_bits = self.num_bits

        dequant_func = None

        def naive_cast_dequant(x):
            return x.astype(in_dtype)

        if with_zeros and zeros_mode == "quantized":
            dequant_func = _tir_packed_to_unsigned_convert_with_zeros(storage_type, storage_nbit)
        elif source_format == "uint":
            if num_bits == 8:
                # 8 num_bits does not need to be compressed
                dequant_func = naive_cast_dequant
            else:
                dequant_func = _tir_packed_to_unsigned_convert(storage_type, storage_nbit)
        elif source_format == "int":
            if num_bits == 1:
                # Dequantize int1 to -1 and 1. Without this step, the values would be 0 and 1, identical to uint1.
                dequant_func = _tir_packed_int_to_int_convert(storage_type, storage_nbit)
            elif num_bits == 8:
                # 8 num_bits does not need to be compressed
                dequant_func = naive_cast_dequant
            else:
                dequant_func = _tir_packed_to_signed_convert(storage_type, storage_nbit)
        elif source_format == "fp":
            dequant_func = _tir_packed_to_fp4_to_f16(storage_type, storage_nbit)
        elif source_format == "fp_e4m3":
            dequant_func = _tir_u8_to_f8_e4m3_to_f16
        else:
            raise ValueError("Unsupported source_format: {}".format(source_format))

        return dequant_func

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
        tx: T.Var,
        k: T.Var,
        i: T.Var,
        stride_n: int,
        stride_k: int,
        threads: int,
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
                        # TODO: Enhance all to index2coord
                        index = i * threads * local_size + tx * local_size + v
                        vi = index // stride_k
                        vj = index % stride_k
                        scale_local[v] = scale_buffer[
                            pid_n * stride_n + vi,
                            (k * stride_k + vj) // group_size,
                        ]

                if with_scaling and with_zeros:
                    if zeros_mode in ["original", "rescale"]:
                        for v in T.vectorized(0, local_zeros_size):
                            index = i * threads * local_size + tx * local_size + v
                            vi = index // stride_k
                            vj = index % stride_k
                            zeros_local[v] = zeros_buffer[
                                pid_n * stride_n + vi,
                                (k * stride_k + vj) // group_size,
                            ]
                    elif zeros_mode == "quantized":
                        for v in T.vectorized(0, local_qzeros_size):
                            index = i * threads * local_size + tx * local_size + v
                            vi = index // stride_k
                            vj = index % stride_k
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
                    index = i * threads * local_size + tx * local_size + v
                    vi = index // stride_k
                    vj = index % stride_k
                    if not with_scaling:
                        dequant_weight_local[v] = self._decode_func(
                            num_bits,
                            compressed_weight_local[v // num_elems_per_byte],
                            v % num_elems_per_byte,
                            dtype=in_dtype,
                        )
                    elif not with_zeros:
                        # Scaling only
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
        tx: T.Var,
        k: T.Var,
        i: T.Var,
        stride_n: int,
        stride_k: int,
        threads: int,
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
                    index = i * threads * local_size + tx * local_size + v
                    vi = index // stride_k
                    vj = index % stride_k
                    scale_local[v] = scale_buffer[
                        pid_n * stride_n + vi,
                        (k * stride_k + vj) // group_size,
                    ]

            if with_scaling and with_zeros:
                if zeros_mode in ["original", "rescale"]:
                    for v in T.vectorized(0, local_zeros_size):
                        index = i * threads * local_size + tx * local_size + v
                        vi = index // stride_k
                        vj = index % stride_k
                        zeros_local[v] = zeros_buffer[
                            pid_n * stride_n + vi,
                            (k * stride_k + vj) // group_size,
                        ]
                elif zeros_mode == "quantized":
                    for v in T.vectorized(0, local_qzeros_size):
                        index = i * threads * local_size + tx * local_size + v
                        vi = index // stride_k
                        vj = index % stride_k
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
        tx: T.Var,
        k: T.Var,
        i: T.Var,
        stride_n: int,
        stride_k: int,
        threads: int,
        fast_decoding: bool = False,
        func_name: str = "",
    ):
        if fast_decoding:
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
                tx,
                k,
                i,
                stride_n,
                stride_k,
                threads,
            )
        else:
            return self._normal_dequant(compressed_weight_local, scale_local, zeros_local,
                                        dequant_qzeros_local, dequant_weight_local, lut_buffer,
                                        scale_buffer, zeros_buffer, qzeros_buffer, local_size,
                                        pid_n, tx, k, i, stride_n, stride_k, threads)

    @property
    def num_elems_per_byte(self):
        storage_nbit = int("".join(c for c in self.storage_dtype if c.isdigit()))
        num_bits = self.num_bits
        return storage_nbit // num_bits


@dataclass
class MatmulDequantizeTileLibraryScheduler(MatmulDequantizeBaseScheduler):

    # Default Tile Related Params
    block_M: int = 128
    block_N: int = 128
    block_K: int = 32
    num_stages: int = 2
    threads: int = 128
    enable_rasterization: bool = False  # Enhance L2 Locality

    class TLHint(BaseTLHint):
        hint_type: str = "MatmulDequantizeTileLibraryScheduler"

        def __init__(self):
            super().__init__()

        @classmethod
        def from_roller_hint(cls, hint: Hint):
            tl_hint = cls()
            for key, value in hint.__dict__.items():
                setattr(tl_hint, key, value)

            block = hint.block
            warp = hint.warp
            rstep = hint.rstep
            num_stages = hint.pipeline_stage
            rasterization_plan = hint.rasterization_plan
            enable_rasterization = not isinstance(rasterization_plan, NoRasterization)

            block_row_warps = block[0] // warp[0]
            block_col_warps = block[1] // warp[1]
            warp_size = 32  # NVIDIA GPU warp size is 32
            if num_stages == 1:
                num_stages = 0  # disable pipelining

            tl_hint.block_M = block[0]
            tl_hint.block_N = block[1]
            tl_hint.block_K = rstep[0]
            tl_hint.num_stages = num_stages
            tl_hint.threads = warp_size * block_row_warps * block_col_warps
            tl_hint.enable_rasterization = enable_rasterization

            return tl_hint

        def get_config_params(self):
            return {
                "block_M": self.block_M,
                "block_N": self.block_N,
                "block_K": self.block_K,
                "num_stages": self.num_stages,
                "threads": self.threads,
                "enable_rasterization": self.enable_rasterization,
            }

        def __repr__(self):
            return ("{"
                    f"block_M={self.block_M},"
                    f"block_N={self.block_N},"
                    f"block_K={self.block_K},"
                    f"num_stages={self.num_stages},"
                    f"threads={self.threads},"
                    f"enable_rasterization={self.enable_rasterization}"
                    "}")

    def get_hint_type(self) -> str:
        return self.TLHint.hint_type

    def serialize_hints_to_configs(self, hints: List[Hint]):
        configs = []
        for hint in hints:
            config = self.TLHint.from_roller_hint(hint)
            configs.append(config)
        return configs

    def with_default_config(self):
        block_M = getattr(self, "block_M", 64)
        block_N = getattr(self, "block_N", 64)
        block_K = getattr(self, "block_K", 32)
        num_stages = getattr(self, "num_stages", 2)
        threads = getattr(self, "threads", 128)
        enable_rasterization = getattr(self, "enable_rasterization", False)

        return self.apply_config(
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            num_stages=num_stages,
            threads=threads,
            enable_rasterization=enable_rasterization,
        )

    def apply_config(
        self,
        block_M: Optional[int] = None,
        block_N: Optional[int] = None,
        block_K: Optional[int] = None,
        num_stages: Optional[int] = None,
        threads: Optional[int] = None,
        # Enhance L2 Locality
        enable_rasterization: bool = False,
    ):
        assert block_M is not None, "block_M is required"
        assert block_N is not None, "block_N is required"
        assert block_K is not None, "block_K is required"
        assert num_stages is not None, "num_stages is required"
        assert threads is not None, "threads is required"

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
        local_size = MAX_TRANSACTION_SIZE_IN_BITS // DataType(in_dtype).bits
        local_size_compressed = local_size // num_elems_per_byte

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

        A_shared_shape = (block_M, block_K)
        B_shared_shape = (block_N, block_K // num_elems_per_byte)
        B_dequantize_shared_shape = (block_N, block_K)

        local_scale_size = max(1, local_size // group_size)
        local_zeros_size = max(1, local_size // group_size)
        local_qzeros_size = max(1, local_size // group_size)

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

        cache_write_required = self.check_require_cache()

        @T.prim_func
        def general_shared_dequant_matmul(
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
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared(A_shared_shape, in_dtype)
                B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
                B_local = T.alloc_local([local_size_compressed], storage_dtype)
                scale_local = T.alloc_local([local_scale_size], in_dtype)
                zeros_local = T.alloc_local([local_zeros_size], in_dtype)
                dequant_qzeros_local = T.alloc_local([local_qzeros_size], storage_dtype)
                B_dequantize_local = T.alloc_local([local_size], in_dtype)
                B_dequantize_shared = T.alloc_shared(B_dequantize_shared_shape, in_dtype)
                C_shared = T.alloc_shared([block_M, block_N], out_dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                tx = T.thread_binding(0, threads, thread="threadIdx.x")

                T.use_swizzle(10, enable=enable_rasterization)

                T.import_source(import_source)

                T.clear(C_local)

                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                    T.copy(B[bx * block_N, ko * block_K // num_elems_per_byte], B_shared)

                    for i in T.serial(block_N * block_K // num_elems_per_byte //
                                      (threads * local_size_compressed)):
                        for v in T.vectorized(0, local_size_compressed):
                            index = (
                                i * threads * local_size_compressed + tx * local_size_compressed +
                                v)
                            vi = index // (block_K // num_elems_per_byte)
                            vj = index % (block_K // num_elems_per_byte)
                            B_local[v] = B_shared[vi, vj]

                        self.dequantize(
                            B_local,
                            scale_local,
                            zeros_local,
                            dequant_qzeros_local,
                            B_dequantize_local,
                            LUT,
                            Scale,
                            Zeros,
                            Qzeros,
                            local_size,
                            bx,
                            tx,
                            ko,
                            i,
                            block_N,
                            block_K,
                            threads,
                            fast_decoding,
                            func_name,
                        )
                        for v in T.vectorized(0, local_size):
                            index = i * threads * local_size + tx * local_size + v
                            vi = index // block_K
                            vj = index % block_K
                            B_dequantize_shared[vi, vj] = B_dequantize_local[v]

                    T.gemm(A_shared, B_dequantize_shared, C_local, transpose_B=True)

                if cache_write_required:
                    T.copy(C_local, C_shared)
                    if with_bias:
                        for i, j in T.grid(block_M, block_N):
                            C_shared[i, j] += Bias[bx * block_N + j]

                    T.copy(C_shared, C[by * block_M, bx * block_N])
                else:
                    if with_bias:
                        for i, j in T.grid(block_M, block_N):
                            C_local[i, j] += Bias[bx * block_N + j]
                    T.copy(C_local, C[by * block_M, bx * block_N])

        return self.post_process(general_shared_dequant_matmul)

    def infer_default_quantization_memory_stage(self):
        # Dequantize Stage
        # We automatically set the quantization memory stage by set the value to None
        # By default we dequantize in shared memory
        quantization_memory_stage = QuantizationMemoryStage.Shared
        return quantization_memory_stage
