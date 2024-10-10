# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm as tvm
from bitblas import tilelang as tilelang
from tvm import DataType
import tilelang.language as T
from typing import Optional, List, Literal
from bitblas.tl.utils import (
    get_mma_micro_size,  # noqa: F401
    make_swizzle_layout,  # noqa: F401
)

from bitblas.tl.macro_generator import (
    TensorCoreIntrinEmitter,  # noqa: F401
    TensorCoreIntrinEmitterWithLadderTransform,  # noqa: F401
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

# GPU warp configuration for NVIDIA GPUs
warp_size = 32


@dataclass
class MatmulDequantizeScheduler(BaseScheduler):

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

    # Default Tile Related Params
    block_M: int = 64
    block_N: int = 64
    block_K: int = 32
    num_stages: int = 2
    threads: int = 128
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

    def _apply_config_dequant_only(
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
        M, N, K = self.M, self.N, self.K
        trans_A, trans_B = self.trans_A, self.trans_B
        assert trans_A is False, "Dequantize only implement for trans_A=False currently"
        assert trans_B is True, "Dequantize only implement for trans_B=TRue currently"

        # check is dequantize only

        def check_is_dequantize_only():
            return not self.with_scaling

        if not check_is_dequantize_only():
            raise ValueError("Not a Dequantize Only Configuration")

        in_dtype, out_dtype, accum_dtype = self.in_dtype, self.out_dtype, self.accum_dtype

        num_bits = self.num_bits
        storage_dtype = self.storage_dtype
        storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
        num_elems_per_byte = 8 // num_bits

        MAX_TRANSACTION_SIZE_IN_BITS = 128
        local_size = MAX_TRANSACTION_SIZE_IN_BITS // DataType(in_dtype).bits
        local_size_compressed = local_size // num_elems_per_byte

        group_size = self.group_size
        if group_size == -1:
            group_size = K

        A_shape = (M, K)
        B_shape = (N, K // storage_nbit * num_bits)

        A_shared_shape = (block_M, block_K)
        B_shared_shape = (block_N, block_K // num_elems_per_byte)
        B_dequantize_shared_shape = (block_N, block_K)

        @T.prim_func
        def main(
                A: T.Buffer(A_shape, in_dtype),
                B: T.Buffer(B_shape, storage_dtype),
                C: T.Buffer((M, N), out_dtype),
        ):
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared(A_shared_shape, in_dtype)
                B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
                B_local = T.alloc_local([local_size_compressed], storage_dtype)
                B_dequantize_local = T.alloc_local([local_size], in_dtype)
                B_dequantize_shared = T.alloc_shared(B_dequantize_shared_shape, in_dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                tx = T.thread_binding(0, threads, thread="threadIdx.x")

                T.use_swizzle(10, enable=enable_rasterization)

                T.clear(C_local)

                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)

                    for i in T.serial(block_N * block_K // num_elems_per_byte //
                                      (threads * local_size_compressed)):
                        for v in T.vectorized(0, local_size_compressed):
                            index = i * threads * local_size_compressed + tx * local_size_compressed + v
                            vi = index // (block_K // num_elems_per_byte)
                            vj = index % (block_K // num_elems_per_byte)
                            B_local[v] = B_shared[vi, vj]
                        for v in T.serial(0, local_size):
                            B_dequantize_local[v] = self._decode_func(
                                num_bits,
                                B_local[v // num_elems_per_byte],
                                v % num_elems_per_byte,
                                dtype=in_dtype,
                            )
                        for v in T.vectorized(0, local_size):
                            index = i * threads * local_size + tx * local_size + v
                            vi = index // block_K
                            vj = index % block_K
                            B_dequantize_shared[vi, vj] = B_dequantize_local[v]

                    T.gemm(A_shared, B_dequantize_shared, C_local, transpose_B=True)

                T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    def _apply_config_with_scaling(
        self,
        block_M: Optional[int] = None,
        block_N: Optional[int] = None,
        block_K: Optional[int] = None,
        num_stages: Optional[int] = None,
        threads: Optional[int] = None,
        # Enhance L2 Locality
        enable_rasterization: bool = False,
    ):
        raise NotImplementedError("Scaling Configuration is not implemented")

    def _apply_config_with_scaling_zeros_original_or_rescale(
        self,
        block_M: Optional[int] = None,
        block_N: Optional[int] = None,
        block_K: Optional[int] = None,
        num_stages: Optional[int] = None,
        threads: Optional[int] = None,
        # Enhance L2 Locality
        enable_rasterization: bool = False,
    ):
        raise NotImplementedError("Scaling and Zeros Original Configuration is not implemented")

    def _apply_config_with_scaling_zeros_quantized(
        self,
        block_M: Optional[int] = None,
        block_N: Optional[int] = None,
        block_K: Optional[int] = None,
        num_stages: Optional[int] = None,
        threads: Optional[int] = None,
        # Enhance L2 Locality
        enable_rasterization: bool = False,
    ):
        raise NotImplementedError("Scaling and Zeros Rescale Configuration is not implemented")

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
        trans_A, trans_B = self.trans_A, self.trans_B

        assert trans_A is False, "Dequantize only implement for trans_A=False currently"
        assert trans_B is True, "Dequantize only implement for trans_B=TRue currently"

        with_scaling = self.with_scaling
        with_zeros = self.with_zeros
        zeros_mode = self.zeros_mode

        args = [block_M, block_N, block_K, num_stages, threads, enable_rasterization]

        dequant_prim_func = None
        if not with_scaling:
            dequant_prim_func = self._apply_config_dequant_only(*args)

        if not with_zeros:
            dequant_prim_func = self._apply_config_with_scaling(*args)

        if zeros_mode in ["original", "rescale"]:
            dequant_prim_func = self._apply_config_with_scaling_zeros_original_or_rescale(*args)
        elif zeros_mode == "quantized":
            dequant_prim_func = self._apply_config_with_scaling_zeros_quantized(*args)
        else:
            raise ValueError("Unsupported zeros_mode: {}".format(zeros_mode))

        if dequant_prim_func is None:
            raise ValueError("Unsupported Configuration")

        return self.maybe_simplify(dequant_prim_func)

    @property
    def _decode_func(self):
        with_zeros = self.with_zeros
        zeros_mode = self.zeros_mode
        storage_dtype = self.storage_dtype

        in_dtype = self.in_dtype
        source_format = self.source_format
        storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
        storage_type = str("".join(c for c in storage_dtype if not c.isdigit()))
        bit = self.bit

        dequant_func = None

        def naive_cast_dequant(x):
            return x.astype(in_dtype)

        if with_zeros and zeros_mode == "quantized":
            dequant_func = _tir_packed_to_unsigned_convert_with_zeros(storage_type, storage_nbit)
        elif source_format == "uint":
            if bit == 8:
                # 8 bit does not need to be compressed
                dequant_func = naive_cast_dequant
            else:
                dequant_func = _tir_packed_to_unsigned_convert(storage_type, storage_nbit)
        elif source_format == "int":
            if bit == 1:
                # Dequantize int1 to -1 and 1. Without this step, the values would be 0 and 1, identical to uint1.
                dequant_func = _tir_packed_int_to_int_convert(storage_type, storage_nbit)
            elif bit == 8:
                # 8 bit does not need to be compressed
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

    def __post_init__(self):
        # Add Config Validation
        return
