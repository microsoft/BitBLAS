# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# tile represents tile library

from bitblas import tvm as tvm
from bitblas import tilelang as tilelang
import tilelang.language as T
from typing import Optional, List
from tilelang.intrinsics.utils import (
    get_mma_micro_size,
    make_mma_swizzle_layout as make_swizzle_layout,
)

from bitblas.tl.mma_macro_generator import (TensorCoreIntrinEmitter,
                                            TensorCoreIntrinEmitterWithLadderTransform)
from bitblas.base.operator_common import TransformKind
from bitblas.base.arch import TileDevice
from bitblas.base.roller.hint import Hint
from bitblas.base.roller.rasterization import NoRasterization
from bitblas.base.utils import get_roller_hints_from_func
from dataclasses import dataclass
from bitblas.ops.general_matmul.tirscript import (matmul_select_implementation)
from bitblas.tl.base_hint import BaseTLHint
from .base import MatmulBaseParams
# GPU warp configuration for NVIDIA GPUs
warp_size = 32


@dataclass
class MatmulBaseScheduler(MatmulBaseParams):

    def get_roller_configs(self, arch: TileDevice = None, topk: int = 10):
        layout = f"{'t' if self.trans_A else 'n'}{'t' if self.trans_B else 'n'}"

        # Simple TIR Compute Expression
        ir_module = matmul_select_implementation(
            M=self.M,
            N=self.N,
            K=self.K,
            in_dtype=self.in_dtype,
            out_dtype=self.out_dtype,
            accum_dtype=self.accum_dtype,
            layout=layout,
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
        # Bias Add should be performed in shared memory
        conditions.append(with_bias)
        return any(conditions)  # Always set to False Currently


@dataclass
class MatmulTileLibraryScheduler(MatmulBaseScheduler):

    # Default Tile Related Params
    block_M: int = 64
    block_N: int = 64
    block_K: int = 32
    num_stages: int = 2
    threads: int = 128
    enable_rasterization: bool = False  # Enhance L2 Locality

    class TLHint(BaseTLHint):

        hint_type = "MatmulTileLibraryScheduler"

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

    def get_configs_sm80(self):
        num_stages = 2
        configs = [
            {
                'block_M': 128,
                'block_N': 256,
                'block_K': 32,
                'threads': 128
            },
            {
                'block_M': 256,
                'block_N': 128,
                'block_K': 32,
                'threads': 128
            },
            {
                'block_M': 128,
                'block_N': 128,
                'block_K': 32,
                'threads': 128
            },
        ]
        configs = [{**c, 'num_stages': num_stages} for c in configs]
        return configs

    def get_hint_type(self):
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
        in_dtype, out_dtype, accum_dtype = self.in_dtype, self.out_dtype, self.accum_dtype
        with_bias = self.with_bias

        A_shape = (K, M) if trans_A else (M, K)
        B_shape = (N, K) if trans_B else (K, N)
        C_shape = (M, N)
        Bias_shape = (N,)
        A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
        B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

        @T.prim_func
        def main(
                A: T.Buffer(A_shape, in_dtype),
                B: T.Buffer(B_shape, in_dtype),
                Bias: T.Buffer(Bias_shape, out_dtype),
                C: T.Buffer(C_shape, out_dtype),
        ):
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared(A_shared_shape, in_dtype)
                B_shared = T.alloc_shared(B_shared_shape, in_dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                T.use_swizzle(10, enable=enable_rasterization)

                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    if trans_A:
                        T.copy(A[k * block_K, by * block_M], A_shared)
                    else:
                        T.copy(A[by * block_M, k * block_K], A_shared)
                    if trans_B:
                        T.copy(B[bx * block_N, k * block_K], B_shared)
                    else:
                        T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)

                if with_bias:
                    for i, j in T.Parallel(block_M, block_N):
                        C_local[i, j] += Bias[bx * block_N + j]

                T.copy(C_local, C[by * block_M, bx * block_N])

        return self.post_process(main)

    def __post_init__(self):
        # Add Config Validation
        return


# TODO(lei): remove these legacy functions in the future
def matmul_blocked(
        M,
        N,
        K,
        block_M=64,
        block_N=64,
        block_K=32,
        trans_A=False,
        trans_B=False,
        in_dtype="float16",
        out_dtype="float16",
        accum_dtype="float16",
        num_stages=2,
        threads=128,
        enable_rasterization: bool = False,  # Enhance L2 Locality
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    C_shape = (M, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
            A: T.Buffer(A_shape, in_dtype),
            B: T.Buffer(B_shape, in_dtype),
            C: T.Buffer(C_shape, out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.use_swizzle(10, enable=enable_rasterization)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def matmul_macro_tensorcore(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    trans_A,
    trans_B,
    accum_dtype,
    block_row_warps,
    block_col_warps,
    warp_row_tiles,
    warp_col_tiles,
    chunk,
    num_stages=2,
    enable_rasterization: bool = False,
):
    assert trans_A is False, "Currently only support Matrix A is not transposed"
    assert trans_B is True, "Currently only support Matrix B is transposed"

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    micro_size_x, micro_size_y, micro_size_k = get_mma_micro_size(in_dtype)

    A_shape = (M, K)
    B_shape = (N, K)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)
    C_shared_shape = (
        block_M // micro_size_x,
        block_N // micro_size_y,
        micro_size_x,
        micro_size_y,
    )

    warp_size = 32  # nvidia gpu warp size is 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size_a = (micro_size_x * micro_size_k) // warp_size
    local_size_b = (micro_size_y * micro_size_k) // warp_size
    local_size_c = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    shared_scope = "shared.dyn"  # Literal["shared", "shared.dyn"] while shared for static shared memory
    mma_emitter = TensorCoreIntrinEmitter(
        a_dtype=in_dtype,
        b_dtype=in_dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
    )

    @T.prim_func
    def main(
            A: T.Buffer(A_shape, in_dtype),
            B: T.Buffer(B_shape, in_dtype),
            C: T.Buffer((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):

            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, out_dtype, scope=shared_scope)
            A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)
            B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size_c), accum_dtype)
            thread_bindings = T.thread_binding(0, threads, "threadIdx.x")

            T.annotate_layout({
                A_shared: make_swizzle_layout(A_shared),
                B_shared: make_swizzle_layout(B_shared),
            })

            T.use_swizzle(panel_size=10, enable=enable_rasterization)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):

                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

                for ki in T.serial(0, (block_K // micro_size_k)):

                    # Load A into fragment
                    mma_emitter.ldmatrix_a(
                        A_local,
                        A_shared,
                        ki,
                        thread_bindings=thread_bindings,
                    )

                    # Load B into fragment
                    mma_emitter.ldmatrix_b(
                        B_local,
                        B_shared,
                        ki,
                        thread_bindings=thread_bindings,
                    )

                    mma_emitter.mma(A_local, B_local, C_local)

            mma_emitter.stmatrix(
                C_local,
                C_shared,
                thread_bindings=thread_bindings,
            )

            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = C_shared[
                    i // micro_size_x,
                    j // micro_size_y,
                    i % micro_size_x,
                    j % micro_size_y,
                ]

    return main


def matmul_macro_tensorcore_weight_propagation_level_ldmatrix(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    trans_A,
    trans_B,
    accum_dtype,
    block_row_warps,
    block_col_warps,
    warp_row_tiles,
    warp_col_tiles,
    chunk,
    num_stages=2,
    enable_rasterization: bool = False,
):
    assert trans_A is False, "Currently only support Matrix A is not transposed"
    assert trans_B is True, "Currently only support Matrix B is transposed"

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    micro_size_x, micro_size_y, micro_size_k = get_mma_micro_size(in_dtype)

    A_shape = (M, K)
    B_shape = (N // micro_size_y, K // micro_size_k, micro_size_y, micro_size_k)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (
        block_N // micro_size_y,
        block_K // micro_size_k,
        micro_size_y,
        micro_size_k,
    )
    C_shared_shape = (
        block_M // micro_size_x,
        block_N // micro_size_y,
        micro_size_x,
        micro_size_y,
    )

    warp_size = 32  # nvidia gpu warp size is 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size_a = (micro_size_x * micro_size_k) // warp_size
    local_size_b = (micro_size_y * micro_size_k) // warp_size
    local_size_c = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    shared_scope = "shared.dyn"  # Literal["shared", "shared.dyn"] while shared for static shared memory
    mma_emitter = TensorCoreIntrinEmitterWithLadderTransform(
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
        transform_kind_b=TransformKind.LDMatrixTransform,
    )

    @T.prim_func
    def main(
            A: T.Buffer(A_shape, in_dtype),
            B: T.Buffer(B_shape, in_dtype),
            C: T.Buffer((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):

            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, out_dtype, scope=shared_scope)
            A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)
            B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size_c), accum_dtype)
            thread_bindings = T.thread_binding(0, threads, "threadIdx.x")

            T.annotate_layout({
                A_shared: make_swizzle_layout(A_shared),
                B_shared: make_swizzle_layout(B_shared),
            })

            T.use_swizzle(panel_size=10, enable=enable_rasterization)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):

                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                for j, k, jj, kk in T.Parallel(
                        block_N // micro_size_y,
                        block_K // micro_size_k,
                        micro_size_y,
                        micro_size_k,
                ):
                    B_shared[j, k, jj, kk] = B[
                        bx * (block_N // micro_size_y) + j,
                        ko * (block_K // micro_size_k) + k,
                        jj,
                        kk,
                    ]

                for ki in T.serial(0, (block_K // micro_size_k)):

                    # Load A into fragment
                    mma_emitter.ldmatrix_a(
                        A_local,
                        A_shared,
                        ki,
                        thread_bindings=thread_bindings,
                    )

                    # Load B into fragment
                    mma_emitter.ldmatrix_b(
                        B_local,
                        B_shared,
                        ki,
                        thread_bindings=thread_bindings,
                    )

                    mma_emitter.mma(A_local, B_local, C_local)

            mma_emitter.stmatrix(
                C_local,
                C_shared,
                thread_bindings=thread_bindings,
            )

            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = C_shared[
                    i // micro_size_x,
                    j // micro_size_y,
                    i % micro_size_x,
                    j % micro_size_y,
                ]

    return main
