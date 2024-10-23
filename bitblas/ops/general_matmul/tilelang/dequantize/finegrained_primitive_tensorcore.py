# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm as tvm
from tvm import DataType
import tvm.tl.language as T
from typing import Optional, List, Literal
from bitblas.tl.utils import (
    get_mma_micro_size,  # noqa: F401
    make_swizzle_layout,  # noqa: F401
)

from bitblas.tl.macro_generator import (
    TensorCoreIntrinEmitter,  # noqa: F401
)
from bitblas.ops.common import TransformKind  # noqa: F401
from bitblas.ops.base_scheduler import BaseScheduler
from bitblas.base.arch import TileDevice
from bitblas.base.roller.hint import Hint
from bitblas.base.roller.rasterization import NoRasterization
from bitblas.base.utils import get_roller_hints_from_func
from dataclasses import dataclass
from bitblas.ops.general_matmul.tirscript import (
    matmul_dequantize_select_implementation,)
from bitblas.tl.base_hint import BaseTLHint
from bitblas.quantization import (
    _tir_packed_int_to_int_convert,
    _tir_packed_to_signed_convert,
    _tir_packed_to_unsigned_convert,
    _tir_u32_to_f4_to_f16,
    _tir_u8_to_f8_e4m3_to_f16,
    _tir_packed_to_unsigned_convert_with_zeros,
)
from bitblas.gpu.intrin.lop3 import get_lop3_intrin_group

# GPU warp configuration for NVIDIA GPUs
warp_size = 32


@dataclass
class MatmulDequantizeFineGrainedScheduler(BaseScheduler):

    # OP Related Config
    M: Optional[int] = None
    N: Optional[int] = None
    K: Optional[int] = None
    trans_A: bool = False
    trans_B: bool = False
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float16"

    # Dequantize Config
    num_bits: int = 4
    storage_dtype: str = "int8"
    source_format: str = "uint"
    with_scaling: bool = False
    with_zeros: bool = False
    group_size: int = -1
    fast_decoding: bool = False
    with_bias: bool = False
    zeros_mode: Literal["original", "rescale", "quantized"] = "original",

    # Tensor Core Warp Configuration
    block_row_warps: int = 2
    block_col_warps: int = 2
    warp_row_tiles: int = 64
    warp_col_tiles: int = 64
    chunk: int = 32  # Usually determines the K-dimension split size

    # Other Optimization Parameters
    num_stages: int = 2
    enable_rasterization: bool = False  # Enhance L2 Locality

    class TLHint(BaseTLHint):

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
            warp_row_tiles = warp[0]
            warp_col_tiles = warp[1]
            chunk = rstep[0]

            if num_stages == 1:
                num_stages = 0  # disable pipelining

            tl_hint.block_row_warps = block_row_warps
            tl_hint.block_col_warps = block_col_warps
            tl_hint.warp_row_tiles = warp_row_tiles
            tl_hint.warp_col_tiles = warp_col_tiles
            tl_hint.chunk = chunk
            tl_hint.num_stages = num_stages
            tl_hint.enable_rasterization = enable_rasterization

            return tl_hint

        def get_config_params(self):
            return {
                "block_row_warps": self.block_row_warps,
                "block_col_warps": self.block_col_warps,
                "warp_row_tiles": self.warp_row_tiles,
                "warp_col_tiles": self.warp_col_tiles,
                "chunk": self.chunk,
                "num_stages": self.num_stages,
                "enable_rasterization": self.enable_rasterization,
            }

        def __repr__(self):
            return ("{"
                    f"block_M={self.block_row_warps * self.warp_row_tiles},"
                    f"block_N={self.block_col_warps * self.warp_col_tiles},"
                    f"warp_M={self.warp_row_tiles},"
                    f"warp_N={self.warp_col_tiles},"
                    f"block_K={self.chunk},"
                    f"threads={self.block_row_warps * self.block_col_warps * warp_size},"
                    f"num_stages={self.num_stages},"
                    f"enable_rasterization={self.enable_rasterization}"
                    "}")

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
            zeros_mode=self.zeros_mode)

        roller_hints = get_roller_hints_from_func(
            ir_module["main"],
            arch,
            topk,
            tensorcore_only=True,
            allow_gemv=True,
        )

        if roller_hints is None:
            raise ValueError("No Roller Hints Found for TensorCore Scheduling")

        def serialze_hints_to_configs(hints: List[Hint]):
            configs = []
            for hint in hints:
                config = self.TLHint.from_roller_hint(hint)
                configs.append(config)
            return configs

        return serialze_hints_to_configs(roller_hints)

    def get_hardware_aware_configs(self, arch: TileDevice = None, topk=10):
        return self.get_roller_configs(arch, topk)

    def with_default_config(self):
        block_row_warps = getattr(self, "block_row_warps", 2)
        block_col_warps = getattr(self, "block_col_warps", 2)
        warp_row_tiles = getattr(self, "warp_row_tiles", 32)
        warp_col_tiles = getattr(self, "warp_col_tiles", 32)
        chunk = getattr(self, "chunk", 32)
        num_stages = getattr(self, "num_stages", 2)
        enable_rasterization = getattr(self, "enable_rasterization", False)

        return self.apply_config(
            block_row_warps=block_row_warps,
            block_col_warps=block_col_warps,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=chunk,
            num_stages=num_stages,
            enable_rasterization=enable_rasterization,
        )

    def apply_config(
        self,
        block_row_warps: Optional[int] = None,
        block_col_warps: Optional[int] = None,
        warp_row_tiles: Optional[int] = None,
        warp_col_tiles: Optional[int] = None,
        chunk: Optional[int] = None,
        num_stages: Optional[int] = None,
        enable_rasterization=False,
    ):
        assert block_row_warps is not None, "block_row_warps is required"
        assert block_col_warps is not None, "block_col_warps is required"
        assert warp_row_tiles is not None, "warp_row_tiles is required"
        assert warp_col_tiles is not None, "warp_col_tiles is required"
        assert chunk is not None, "chunk is required"
        assert num_stages is not None, "num_stages is required"

        M, N, K = self.M, self.N, self.K
        trans_A, trans_B = self.trans_A, self.trans_B

        assert trans_A is False, "Dequantize only implement for trans_A=False currently"
        assert trans_B is True, "Dequantize only implement for trans_B=TRue currently"

        in_dtype, out_dtype, accum_dtype = (
            self.in_dtype,
            self.out_dtype,
            self.accum_dtype,
        )
        # Calculate the micro size per warp using a helper function
        micro_size_x, micro_size_y, micro_size_k = get_mma_micro_size(in_dtype)

        block_M = block_row_warps * warp_row_tiles
        block_N = block_col_warps * warp_col_tiles
        block_K = chunk
        threads = warp_size * (block_row_warps * block_col_warps)

        fragement_size = (micro_size_x * micro_size_y) // warp_size
        warp_rows = warp_row_tiles // micro_size_x
        warp_cols = warp_col_tiles // micro_size_y

        fast_decoding = self.fast_decoding

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
        B_shape = (N, K // num_elems_per_byte)
        LUT_shape = (group_size, K // num_elems_per_byte)
        Scale_shape = (N, K // group_size)
        Zeros_shape = (N, K // group_size)
        Qzeros_shape = ((K // group_size), N // storage_nbit * num_bits)
        Bias_shape = (N,)

        A_shared_shape = (block_M, block_K)
        B_shared_shape = (block_N, block_K // num_elems_per_byte)
        B_dequantize_shared_shape = (block_N, block_K)
        C_shared_shape = (
            block_M // micro_size_x,
            block_N // micro_size_y,
            micro_size_x,
            micro_size_y,
        )

        import_source: Optional[str] = None
        func_name: str = ""
        if fast_decoding is True:
            lop3_intrin_info = get_lop3_intrin_group(
                out_dtype=out_dtype,
                source_format=source_format,
                source_bit=num_bits,
                storage_dtype=storage_dtype,
                with_scaling=self.with_scaling,
                with_zeros=self.with_zeros,
            )
            import_source = lop3_intrin_info["c_source"]
            func_name = lop3_intrin_info["func_name"]
            assert import_source is not None, "lop3_intrin_info is not found"
            assert func_name is not None, "lop3_intrin_info is not found"
            import_source = self.common_header + import_source

        # Configure the tensor core intrinsic emitter
        mma_emitter = TensorCoreIntrinEmitter(
            a_dtype=in_dtype,
            b_dtype=in_dtype,
            accum_dtype=accum_dtype,
            a_transposed=trans_A,
            b_transposed=trans_B,
            block_row_warps=block_row_warps,
            block_col_warps=block_col_warps,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=chunk,
        )

        @T.prim_func
        def general_dequant_matmul(
                A: T.Buffer(A_shape, in_dtype),
                B: T.Buffer(B_shape, storage_dtype),
                LUT: T.Buffer(LUT_shape, in_dtype),
                Scale: T.Buffer(Scale_shape, in_dtype),
                Qzeros: T.Buffer(Qzeros_shape, storage_dtype),
                Zeros: T.Buffer(Zeros_shape, in_dtype),
                Bias: T.Buffer(Bias_shape, in_dtype),
                C: T.Buffer((M, N), out_dtype),
        ):
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared(A_shared_shape, in_dtype)
                B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
                B_dequantize_shared = T.alloc_shared(B_dequantize_shared_shape, in_dtype)
                C_shared = T.alloc_shared(C_shared_shape, out_dtype)

                A_frag = T.alloc_local((warp_rows * fragement_size), in_dtype)
                B_frag = T.alloc_local((warp_cols * fragement_size), in_dtype)
                C_frag = T.alloc_local((warp_rows * warp_cols * fragement_size), accum_dtype)

                B_local = T.alloc_local([local_size_compressed], storage_dtype)
                B_dequantize_local = T.alloc_local([local_size], in_dtype)

                tx = T.thread_binding(0, threads, thread="threadIdx.x")

                T.annotate_layout({
                    A_shared: make_swizzle_layout(A_shared),
                    B_dequantize_shared: make_swizzle_layout(B_dequantize_shared),
                })

                T.use_swizzle(10, enable=enable_rasterization)

                T.import_source(import_source)

                T.clear(C_frag)

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

                        if fast_decoding is True:
                            self._normal_fast_dequant(
                                B_local,
                                B_dequantize_local,
                                Scale,
                                Zeros,
                                Qzeros,
                                func_name,
                                by,
                                tx,
                                ko,
                                i,
                                block_N,
                                block_K,
                                threads,
                            )
                        else:
                            self._normal_dequant(
                                B_local,
                                B_dequantize_local,
                                Scale,
                                Zeros,
                                Qzeros,
                                local_size,
                                local_size_compressed,
                                bx,
                                tx,
                                ko,
                                i,
                                block_N,
                                block_K,
                                threads,
                            )
                        for v in T.vectorized(0, local_size):
                            index = i * threads * local_size + tx * local_size + v
                            vi = index // block_K
                            vj = index % block_K
                            B_dequantize_shared[vi, vj] = B_dequantize_local[v]

                    # Perform the matrix multiplication on tensor core fragments
                    for ki in T.serial(0, (block_K // micro_size_k)):

                        # Load A fragment
                        mma_emitter.ldmatrix_a(
                            A_frag,
                            A_shared,
                            ki,
                            thread_bindings=tx,
                        )

                        # Load B fragment
                        mma_emitter.ldmatrix_b(
                            B_frag,
                            B_dequantize_shared,
                            ki,
                            thread_bindings=tx,
                        )

                        # Matrix multiplication on fragments
                        mma_emitter.mma(A_frag, B_frag, C_frag)

                # Store the result back to C shared memory
                mma_emitter.stmatrix(
                    C_frag,
                    C_shared,
                    thread_bindings=tx,
                )

                # Store results from shared memory to global memory
                for i, j in T.Parallel(block_M, block_N):
                    C[by * block_M + i, bx * block_N + j] = C_shared[
                        i // micro_size_x,
                        j // micro_size_y,
                        i % micro_size_x,
                        j % micro_size_y,
                    ]

        return self.maybe_simplify(general_dequant_matmul)

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
            dequant_func = _tir_u32_to_f4_to_f16
        elif source_format == "fp_e4m3":
            dequant_func = _tir_u8_to_f8_e4m3_to_f16
        else:
            raise ValueError("Unsupported source_format: {}".format(source_format))

        return dequant_func

    def _normal_dequant(
        self,
        compressed_weight_local: T.Buffer,
        dequant_weight_local: T.Buffer,
        scale_buffer: T.Buffer,
        zeros_buffer: T.Buffer,
        qzeros_buffer: T.Buffer,
        local_size: int,
        local_size_compressed: int,
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

        @T.macro
        def _normal_dequant_impl(
            compressed_weight_local: T.Buffer,
            dequant_weight_local: T.Buffer,
            scale_buffer: T.Buffer,
            zeros_buffer: T.Buffer,
            qzeros_buffer: T.Buffer,
        ):
            for v in T.serial(0, local_size):
                index = (i * threads * local_size + tx * local_size + v)
                vi = index // (stride_k)
                vj = index % (stride_k)
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
                        ) * scale_buffer[pid_n * stride_n + vi, (k * stride_k + vj) // group_size])
                elif zeros_mode == "original":
                    dequant_weight_local[v] = (self._decode_func(
                        num_bits,
                        compressed_weight_local[v // num_elems_per_byte],
                        v % num_elems_per_byte,
                        dtype=in_dtype,
                    ) - zeros_buffer[pid_n * stride_n + vi, (k * stride_k + vj) //
                                     group_size]) * scale_buffer[pid_n * stride_n + vi,
                                                                 (k * stride_k + vj) // group_size]
                elif zeros_mode == "rescale":
                    dequant_weight_local[v] = (
                        self._decode_func(
                            num_bits,
                            compressed_weight_local[v // num_elems_per_byte],
                            v % num_elems_per_byte,
                            dtype=in_dtype,
                        ) * scale_buffer[pid_n * stride_n + vi, (k * stride_k + vj) // group_size] -
                        zeros_buffer[pid_n * stride_n + vi, (k * stride_k + vj) // group_size])
                elif zeros_mode == "quantized":
                    dequant_qzeros = _tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
                        num_bits,
                        qzeros_buffer[
                            (k * stride_k + vj) // group_size,
                            (pid_n * stride_n + vi) // num_elems_per_byte,
                        ],
                        (pid_n * stride_n + vi) % num_elems_per_byte,
                        dtype=storage_dtype,
                    )

                    dequant_weight_local[v] = (self._decode_func(
                        num_bits,
                        compressed_weight_local[v // num_elems_per_byte],
                        v % num_elems_per_byte,
                        zero=dequant_qzeros,
                        dtype=in_dtype,
                    )) * scale_buffer[pid_n * stride_n + vi, (k * stride_k + vj) // group_size]

        return _normal_dequant_impl(
            compressed_weight_local,
            dequant_weight_local,
            scale_buffer,
            zeros_buffer,
            qzeros_buffer,
        )

    def _normal_fast_dequant(
        self,
        compressed_weight_local: T.Buffer,
        dequant_weight_local: T.Buffer,
        scale_buffer: T.Buffer,
        zeros_buffer: T.Buffer,
        qzeros_buffer: T.Buffer,
        func_name: str,
        pid_n: T.Var,
        tx: T.Var,
        k: T.Var,
        i: T.Var,
        stride_n: int,
        stride_k: int,
        threads: int,
    ):
        # TODO(lei): un-used arguments should be removed
        num_elems_per_byte = self.num_elems_per_byte
        with_scaling = self.with_scaling
        with_zeros = self.with_zeros
        zeros_mode = self.zeros_mode
        in_dtype = self.in_dtype
        group_size = self.group_size

        @T.macro
        def _normal_fast_dequant_impl(
            compressed_weight_local: T.Buffer,
            dequant_weight_local: T.Buffer,
            scale_buffer: T.Buffer,
            zeros_buffer: T.Buffer,
            qzeros_buffer: T.Buffer,
        ):
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
                    T.address_of(scale_buffer[pid_n * stride_n, k * stride_k // group_size]),
                    dtype=in_dtype,
                )
            elif zeros_mode in ["original", "rescale"]:
                T.call_extern(
                    func_name,
                    T.address_of(compressed_weight_local[0]),
                    T.address_of(dequant_weight_local[0]),
                    T.address_of(scale_buffer[pid_n * stride_n, k * stride_k // group_size]),
                    T.address_of(zeros_buffer[pid_n * stride_n, k * stride_k // group_size]),
                    dtype=in_dtype,
                )
            elif zeros_mode == "quantized":
                T.call_extern(
                    func_name,
                    T.address_of(compressed_weight_local[0]),
                    T.address_of(dequant_weight_local[0]),
                    T.address_of(scale_buffer[pid_n * stride_n, k * stride_k // group_size]),
                    T.address_of(zeros_buffer[pid_n * stride_n, k * stride_k // group_size]),
                    T.address_of(qzeros_buffer[k * stride_k // group_size,
                                               pid_n * stride_n // num_elems_per_byte]),
                    dtype=in_dtype,
                )

        return _normal_fast_dequant_impl(
            compressed_weight_local,
            dequant_weight_local,
            scale_buffer,
            zeros_buffer,
            qzeros_buffer,
        )

    @property
    def num_elems_per_byte(self):
        storage_nbit = int("".join(c for c in self.storage_dtype if c.isdigit()))
        num_bits = self.num_bits
        return storage_nbit // num_bits

    def __post_init__(self):
        # Legalize group_size
        if self.with_scaling and self.group_size == -1:
            object.__setattr__(self, "group_size", self.K)
