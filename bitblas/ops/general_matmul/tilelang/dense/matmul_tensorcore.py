# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm as tvm
from tvm import DataType
import tvm.tl.language as T
from typing import Optional, List
from bitblas.tl.utils import (
    get_mma_micro_size,
    make_mma_swizzle_layout as make_swizzle_layout,
)

from bitblas.tl.mma_macro_generator import (
    TensorCoreIntrinEmitter,
    TensorCoreIntrinEmitterWithLadderTransform,
)
from bitblas.ops.common import TransformKind
from bitblas.ops.base_scheduler import BaseScheduler
from bitblas.base.arch import TileDevice
from bitblas.base.roller.hint import Hint
from bitblas.base.roller.rasterization import NoRasterization
from bitblas.base.utils import get_roller_hints_from_func
from dataclasses import dataclass
from bitblas.ops.general_matmul.tirscript import (matmul_select_implementation)
from bitblas.tl.base_hint import BaseTLHint

# GPU warp configuration for NVIDIA GPUs
warp_size = 32


class MatmulBaseScheduler(BaseScheduler):
    # OP Related Config
    M: Optional[int] = None
    N: Optional[int] = None
    K: Optional[int] = None
    trans_A: bool = False
    trans_B: bool = False
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float16"
    with_bias: bool = False

    def serialze_hints_to_configs(self, hints: List[Hint]) -> List[BaseTLHint]:
        # Convert Roller Hints to TileLang Hints
        raise NotImplementedError

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

        return self.serialze_hints_to_configs(roller_hints)

    def get_hardware_aware_configs(self, arch: TileDevice = None, topk=10):
        return self.get_roller_configs(arch, topk)


@dataclass
class MatmulBlockScheduler(MatmulBaseScheduler):

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

    def serialze_hints_to_configs(self, hints: List[Hint]):
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

        M, N, K = self.M, self.N, self.K
        trans_A, trans_B = self.trans_A, self.trans_B
        in_dtype, out_dtype, accum_dtype = self.in_dtype, self.out_dtype, self.accum_dtype
        with_bias = self.with_bias

        A_shape = (K, M) if trans_A else (M, K)
        B_shape = (N, K) if trans_B else (K, N)
        C_shape = (M, N)
        Bias_shape = (N,) if with_bias else None
        A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
        B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

        @T.prim_func
        def main(
                A: T.Buffer(A_shape, in_dtype),
                B: T.Buffer(B_shape, in_dtype),
                C: T.Buffer(C_shape, out_dtype),
                Bias: T.Buffer(Bias_shape, out_dtype),
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

        return self.maybe_simplify(main)

    def __post_init__(self):
        # Add Config Validation
        return


@dataclass
class MatmulFineGrainScheduler(MatmulBaseScheduler):
    # Fine-grained matrix multiplication scheduler
    # Allows for more detailed configuration.

    # Tensor Core Warp Configuration
    block_row_warps: int = 2
    block_col_warps: int = 2
    warp_row_tiles: int = 32
    warp_col_tiles: int = 32
    chunk: int = 32  # Usually determines the K-dimension split size

    # Other Optimization Parameters
    num_stages: int = 2
    enable_rasterization: bool = False

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

    def serialze_hints_to_configs(self, hints: List[Hint]):
        configs = []
        for hint in hints:
            config = self.TLHint.from_roller_hint(hint)
            configs.append(config)
        return configs

    def with_default_config(self):
        block_row_warps = getattr(self, "block_row_warps", 2)
        block_col_warps = getattr(self, "block_col_warps", 2)
        warp_row_tiles = getattr(self, "warp_row_tiles", 32)
        warp_col_tiles = getattr(self, "warp_col_tiles", 32)
        chunk = getattr(self, "chunk", 32)
        # Swizzle size for INT8 Storage is 64
        if DataType(self.in_dtype).bits <= 8:
            chunk = 64
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
        in_dtype, out_dtype, accum_dtype = self.in_dtype, self.out_dtype, self.accum_dtype

        # Calculate the micro size per warp using a helper function
        micro_size_x, micro_size_y, micro_size_k = get_mma_micro_size(in_dtype)

        block_M = block_row_warps * warp_row_tiles
        block_N = block_col_warps * warp_col_tiles
        block_K = chunk

        # Define the shapes of matrices and shared memory buffers
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

        threads = warp_size * (block_row_warps * block_col_warps)

        # Calculate local fragment sizes for tensor core
        local_size_a = (micro_size_x * micro_size_k) // warp_size
        local_size_b = (micro_size_y * micro_size_k) // warp_size
        local_size_c = (micro_size_x * micro_size_y) // warp_size
        warp_rows = warp_row_tiles // micro_size_x
        warp_cols = warp_col_tiles // micro_size_y

        shared_scope = "shared.dyn"

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

        # Define the main kernel using the generated configuration
        @T.prim_func
        def main(
                A: T.Buffer(A_shape, in_dtype),
                B: T.Buffer(B_shape, in_dtype),
                C: T.Buffer((M, N), out_dtype),
        ):
            # Grid and thread configuration for CUDA kernel
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):

                # Allocate shared memory and local fragments
                A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
                B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope=shared_scope)
                C_shared = T.alloc_shared(C_shared_shape, out_dtype, scope=shared_scope)
                A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)
                B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
                C_local = T.alloc_local((warp_rows * warp_cols * local_size_c), accum_dtype)

                # Thread-level parallelism for Tensor Cores
                thread_bindings = T.thread_binding(0, threads, "threadIdx.x")

                # Apply memory layout optimizations
                T.annotate_layout({
                    A_shared: make_swizzle_layout(A_shared),
                    B_shared: make_swizzle_layout(B_shared),
                })

                # Optional rasterization for L2 locality enhancement
                T.use_swizzle(panel_size=10, enable=enable_rasterization)

                # Initialize accumulation buffer to zero
                T.clear(C_local)

                # Main matrix multiplication pipeline with multiple stages
                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):

                    # Load A matrix into shared memory
                    for i, k in T.Parallel(block_M, block_K):
                        A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                    # Load B matrix into shared memory
                    for j, k in T.Parallel(block_N, block_K):
                        B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

                    # Perform the matrix multiplication on tensor core fragments
                    for ki in T.serial(0, (block_K // micro_size_k)):

                        # Load A fragment
                        mma_emitter.ldmatrix_a(
                            A_local,
                            A_shared,
                            ki,
                            thread_bindings=thread_bindings,
                        )

                        # Load B fragment
                        mma_emitter.ldmatrix_b(
                            B_local,
                            B_shared,
                            ki,
                            thread_bindings=thread_bindings,
                        )

                        # Matrix multiplication on fragments
                        mma_emitter.mma(A_local, B_local, C_local)

                # Store the result back to C shared memory
                mma_emitter.stmatrix(
                    C_local,
                    C_shared,
                    thread_bindings=thread_bindings,
                )

                # Store results from shared memory to global memory
                for i, j in T.Parallel(block_M, block_N):
                    C[by * block_M + i, bx * block_N + j] = C_shared[
                        i // micro_size_x,
                        j // micro_size_y,
                        i % micro_size_x,
                        j % micro_size_y,
                    ]

        return self.maybe_simplify(main)

    def __post_init__(self):
        # Validate the matrix transpose settings
        assert self.trans_A is False, "Currently only support Matrix A not transposed"
        assert self.trans_B is True, "Currently only support Matrix B transposed"

        return


@dataclass
class MatmulWeightPropagationScheduler(MatmulFineGrainScheduler):

    # Ladder Transform Config
    weight_transform_kind: TransformKind = TransformKind.LDMatrixTransform

    def apply_config(
        self,
        block_row_warps=2,
        block_col_warps=2,
        warp_row_tiles=32,
        warp_col_tiles=32,
        chunk=16,
        num_stages=2,
        enable_rasterization=False,
    ):

        M, N, K = self.M, self.N, self.K
        trans_A, trans_B = self.trans_A, self.trans_B
        in_dtype, out_dtype, accum_dtype = self.in_dtype, self.out_dtype, self.accum_dtype

        # Calculate the micro size per warp using a helper function
        micro_size_x, micro_size_y, micro_size_k = get_mma_micro_size(in_dtype)

        block_M = block_row_warps * warp_row_tiles
        block_N = block_col_warps * warp_col_tiles
        block_K = chunk

        # TODO(lei): Can be generalized to analyzed from bank size
        pad_factor = 8 if in_dtype == "float16" else 16

        can_swizzle_a = block_K * DataType(in_dtype).bits == 512
        apply_pad_a = not can_swizzle_a

        micro_size_x, micro_size_y, micro_size_k = get_mma_micro_size(in_dtype)

        # Define the shapes of matrices and shared memory buffers
        A_shape = (M, K)
        B_shape = (N // micro_size_y, K // micro_size_k, micro_size_y, micro_size_k)
        A_shared_shape = (block_M, (block_K + pad_factor) if apply_pad_a else block_K)
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

        # GPU warp configuration for NVIDIA GPUs
        warp_size = 32
        threads = warp_size * (block_row_warps * block_col_warps)

        # Calculate local fragment sizes for tensor core
        local_size_a = (micro_size_x * micro_size_k) // warp_size
        local_size_b = (micro_size_y * micro_size_k) // warp_size
        local_size_c = (micro_size_x * micro_size_y) // warp_size
        warp_rows = warp_row_tiles // micro_size_x
        warp_cols = warp_col_tiles // micro_size_y

        shared_scope = "shared.dyn"

        # Configure the tensor core intrinsic emitter
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
            transform_kind_b=self.weight_transform_kind,
        )

        # Define the main kernel using the generated configuration
        @T.prim_func
        def main(
                A: T.Buffer(A_shape, in_dtype),
                B: T.Buffer(B_shape, in_dtype),
                C: T.Buffer((M, N), out_dtype),
        ):
            # Grid and thread configuration for CUDA kernel
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):

                # Allocate shared memory and local fragments
                A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
                B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope=shared_scope)
                C_shared = T.alloc_shared(C_shared_shape, out_dtype, scope=shared_scope)
                A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)
                B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
                C_local = T.alloc_local((warp_rows * warp_cols * local_size_c), accum_dtype)

                # Thread-level parallelism for Tensor Cores
                thread_bindings = T.thread_binding(0, threads, "threadIdx.x")

                # Apply memory layout optimizations
                T.annotate_layout({
                    A_shared: make_swizzle_layout(A_shared),
                    B_shared: make_swizzle_layout(B_shared),
                })

                T.use_swizzle(panel_size=10, enable=enable_rasterization)

                # Initialize accumulation buffer to zero
                T.clear(C_local)

                # Main matrix multiplication pipeline with multiple stages
                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):

                    # Load A matrix into shared memory
                    for i, k in T.Parallel(block_M, block_K):
                        A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                    # Load B matrix into shared memory
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

                    # Perform the matrix multiplication on tensor core fragments
                    for ki in T.serial(0, (block_K // micro_size_k)):

                        # Load A fragment
                        mma_emitter.ldmatrix_a(
                            A_local,
                            A_shared,
                            ki,
                            thread_bindings=thread_bindings,
                        )

                        # Load B fragment
                        mma_emitter.ldmatrix_b(
                            B_local,
                            B_shared,
                            ki,
                            thread_bindings=thread_bindings,
                        )

                        # Matrix multiplication on fragments
                        mma_emitter.mma(A_local, B_local, C_local)

                # Store the result back to C shared memory
                mma_emitter.stmatrix(
                    C_local,
                    C_shared,
                    thread_bindings=thread_bindings,
                )

                # Store results from shared memory to global memory
                for i, j in T.Parallel(block_M, block_N):
                    C[by * block_M + i, bx * block_N + j] = C_shared[
                        i // micro_size_x,
                        j // micro_size_y,
                        i % micro_size_x,
                        j % micro_size_y,
                    ]

        return self.maybe_simplify(main)

    def __post_init__(self):
        # Validate the matrix transpose settings
        assert self.trans_A is False, "Currently only support Matrix A not transposed"
        assert self.trans_B is True, "Currently only support Matrix B transposed"

        return


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
        enable_rasterization=False,  # Enhance L2 Locality
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
            A: T.Buffer(A_shape, in_dtype),
            B: T.Buffer(B_shape, in_dtype),
            C: T.Buffer((M, N), out_dtype),
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
    enable_rasterization=False,
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
    enable_rasterization=False,
):
    assert trans_A is False, "Currently only support Matrix A is not transposed"
    assert trans_B is True, "Currently only support Matrix B is transposed"

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    # TODO(lei): Can be generalized to analyzed from bank size
    pad_factor = 8 if in_dtype == "float16" else 16

    can_swizzle_a = block_K * DataType(in_dtype).bits == 512
    apply_pad_a = not can_swizzle_a

    micro_size_x, micro_size_y, micro_size_k = get_mma_micro_size(in_dtype)

    A_shape = (M, K)
    B_shape = (N // micro_size_y, K // micro_size_k, micro_size_y, micro_size_k)
    A_shared_shape = (block_M, (block_K + pad_factor) if apply_pad_a else block_K)
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
