# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# tile represents tile library

from bitblas import tvm as tvm
from bitblas import tilelang as tilelang
from tvm import DataType
import tilelang.language as T
from typing import Optional, List
from tilelang.intrinsics.utils import (
    get_mma_micro_size,
    make_mma_swizzle_layout as make_swizzle_layout,
)

from bitblas.tl.mma_macro_generator import (
    TensorCoreIntrinEmitter,
    TensorCoreIntrinEmitterWithLadderTransform,
    INT4TensorCoreIntrinEmitter,
    INT4TensorCoreIntrinEmitterWithLadderTransform,
)
from bitblas.base.operator_common import TransformKind
from bitblas.base.arch import TileDevice
from bitblas.base.roller.hint import Hint
from bitblas.base.roller.rasterization import NoRasterization
from bitblas.base.utils import get_roller_hints_from_func
from dataclasses import dataclass
from bitblas.ops.general_matmul.tirscript import (matmul_select_implementation)
from bitblas.tl.base_hint import BaseTLHint
from .matmul_tile import MatmulBaseScheduler

# GPU warp configuration for NVIDIA GPUs
warp_size = 32


@dataclass
class MatmulMMAScheduler(MatmulBaseScheduler):
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

        hint_type: str = "MatmulMMAScheduler"

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

    def get_hint_type(self):
        return self.TLHint.hint_type

    def serialize_hints_to_configs(self, hints: List[Hint]):
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
        enable_rasterization: bool = False,
    ):
        assert block_row_warps is not None, "block_row_warps is required"
        assert block_col_warps is not None, "block_col_warps is required"
        assert warp_row_tiles is not None, "warp_row_tiles is required"
        assert warp_col_tiles is not None, "warp_col_tiles is required"
        assert chunk is not None, "chunk is required"
        assert num_stages is not None, "num_stages is required"

        M = self.maybe_dynamic(self.M, "m")
        N, K = self.N, self.K
        assert isinstance(N, int) and isinstance(K, int), "Do not support dynamic N and K Currently"

        trans_A, trans_B = self.trans_A, self.trans_B
        in_dtype, out_dtype, accum_dtype = self.in_dtype, self.out_dtype, self.accum_dtype
        with_bias = self.with_bias

        # Calculate the micro size per warp using a helper function
        micro_size_x, micro_size_y, micro_size_k = get_mma_micro_size(in_dtype)

        block_M = block_row_warps * warp_row_tiles
        block_N = block_col_warps * warp_col_tiles
        block_K = chunk

        # Define the shapes of matrices and shared memory buffers
        A_shape = (M, K)
        B_shape = (N, K)
        C_shape = (M, N)
        Bias_shape = (N,)
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

        cache_write_required = self.check_require_cache()

        # Define the main kernel using the generated configuration
        @T.prim_func
        def main(
                A: T.Buffer(A_shape, in_dtype),
                B: T.Buffer(B_shape, in_dtype),
                Bias: T.Buffer(Bias_shape, out_dtype),
                C: T.Buffer(C_shape, out_dtype),
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

                if cache_write_required:
                    # Store the result back to C shared memory
                    mma_emitter.stmatrix(
                        C_local,
                        C_shared,
                        thread_bindings=thread_bindings,
                    )

                    # Do bias addition
                    if with_bias:
                        for i, j in T.Parallel(block_M, block_N):
                            C_shared[
                                i // micro_size_x,
                                j // micro_size_y,
                                i % micro_size_x,
                                j % micro_size_y,
                            ] += Bias[bx * block_N + j]

                    # Store results from shared memory to global memory
                    for i, j in T.Parallel(block_M, block_N):
                        C[by * block_M + i, bx * block_N + j] = C_shared[
                            i // micro_size_x,
                            j // micro_size_y,
                            i % micro_size_x,
                            j % micro_size_y,
                        ]
                else:
                    # Store the result directly to global memory
                    mma_emitter.stmatrix(
                        C_local,
                        C,
                        thread_bindings=thread_bindings,
                        pid_m=by,
                        pid_n=bx,
                    )

        return self.post_process(main)

    def __post_init__(self):
        # Validate the matrix transpose settings
        assert self.trans_A is False, "Currently only support Matrix A not transposed"
        assert self.trans_B is True, "Currently only support Matrix B transposed"

        return


@dataclass
class MatmulMMAWeightPropagationScheduler(MatmulMMAScheduler):

    # force set default weight transform kind to LDMatrixTransform
    weight_transform_kind: TransformKind = TransformKind.LDMatrixTransform

    class TLHint(MatmulMMAScheduler.TLHint):
        hint_type: str = "MatmulMMAWeightPropagationScheduler"

    def apply_config(
        self,
        block_row_warps=2,
        block_col_warps=2,
        warp_row_tiles=32,
        warp_col_tiles=32,
        chunk=16,
        num_stages=2,
        enable_rasterization: bool = False,
    ):

        M = self.maybe_dynamic(self.M, "m")
        N, K = self.N, self.K
        assert isinstance(N, int) and isinstance(K, int), "Do not support dynamic N and K Currently"

        trans_A, trans_B = self.trans_A, self.trans_B
        in_dtype, out_dtype, accum_dtype = self.in_dtype, self.out_dtype, self.accum_dtype
        with_bias = self.with_bias
        input_transform_kind, weight_transform_kind = self.input_transform_kind, self.weight_transform_kind

        # Calculate the micro size per warp using a helper function
        micro_size_x, micro_size_y, micro_size_k = get_mma_micro_size(in_dtype)

        block_M = block_row_warps * warp_row_tiles
        block_N = block_col_warps * warp_col_tiles
        block_K = chunk

        micro_size_x, micro_size_y, micro_size_k = get_mma_micro_size(in_dtype)

        # Define the shapes of matrices and shared memory buffers
        B_shape = (N // micro_size_y, K // micro_size_k, micro_size_y, micro_size_k)
        C_shape = (M, N)
        Bias_shape = (N,)

        is_a_smooth = self.is_a_smooth
        is_b_smooth = self.is_b_smooth

        if is_a_smooth:
            A_shape = (M // micro_size_x, K // micro_size_k, micro_size_x, micro_size_k)
            A_shared_shape = (
                block_M // micro_size_x,
                block_K // micro_size_k,
                micro_size_x,
                micro_size_k,
            )
        else:
            A_shape = (M, K)
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
            transform_kind_a=input_transform_kind,
            transform_kind_b=weight_transform_kind,
        )

        cache_write_required = self.check_require_cache()
        # Define the main kernel using the generated configuration
        @T.prim_func
        def main(
                A: T.Buffer(A_shape, in_dtype),
                B: T.Buffer(B_shape, in_dtype),
                Bias: T.Buffer(Bias_shape, out_dtype),
                C: T.Buffer(C_shape, out_dtype),
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
                    A_shared: make_swizzle_layout(A_shared, is_smooth=is_a_smooth),
                    B_shared: make_swizzle_layout(B_shared, is_smooth=is_b_smooth),
                })

                T.use_swizzle(panel_size=10, enable=enable_rasterization)

                # Initialize accumulation buffer to zero
                T.clear(C_local)

                # Main matrix multiplication pipeline with multiple stages
                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):

                    if is_a_smooth:
                        for i, k, ii, kk in T.Parallel(
                                block_M // micro_size_x,
                                block_K // micro_size_k,
                                micro_size_x,
                                micro_size_k,
                        ):
                            A_shared[i, k, ii, kk] = A[by * (block_M // micro_size_x) + i,
                                                       ko * (block_K // micro_size_k) + k, ii, kk]
                    else:
                        T.copy(A[by * block_M, ko * block_K], A_shared)

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

                if cache_write_required:
                    # Store the result back to C shared memory
                    mma_emitter.stmatrix(
                        C_local,
                        C_shared,
                        thread_bindings=thread_bindings,
                    )

                    # Do bias addition
                    if with_bias:
                        for i, j in T.Parallel(block_M, block_N):
                            C_shared[
                                i // micro_size_x,
                                j // micro_size_y,
                                i % micro_size_x,
                                j % micro_size_y,
                            ] += Bias[bx * block_N + j]

                    # Store results from shared memory to global memory
                    for i, j in T.Parallel(block_M, block_N):
                        C[by * block_M + i, bx * block_N + j] = C_shared[
                            i // micro_size_x,
                            j // micro_size_y,
                            i % micro_size_x,
                            j % micro_size_y,
                        ]
                else:
                    # Store the result directly to global memory
                    mma_emitter.stmatrix(
                        C_local,
                        C,
                        thread_bindings=thread_bindings,
                        pid_m=by,
                        pid_n=bx,
                    )

        return self.post_process(main)

    @property
    def is_a_smooth(self):
        return self.input_transform_kind > TransformKind.NonTransform

    @property
    def is_b_smooth(self):
        return self.weight_transform_kind > TransformKind.NonTransform


@dataclass
class MatmulINT4MMAScheduler(MatmulMMAScheduler):

    @dataclass
    class TLHint(MatmulMMAScheduler.TLHint):
        hint_type: str = "MatmulINT4MMAScheduler"

    def get_roller_configs(self, arch: TileDevice = None, topk: int = 10):
        layout = f"{'t' if self.trans_A else 'n'}{'t' if self.trans_B else 'n'}"
        M = self.M
        K = self.K // 2  # 2xint4 should be packed into one single int8
        # Simple TIR Compute Expression
        storage_dtype = "int8"

        # This is a hack to utilize tensor core
        if isinstance(M, int) and M < 16:
            M = 16

        ir_module = matmul_select_implementation(
            M=M,
            N=self.N,
            K=K,
            in_dtype=storage_dtype,
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

        def serialize_hints_to_configs(hints: List[Hint]):
            configs = []
            for hint in hints:
                config = self.TLHint.from_roller_hint(hint)
                configs.append(config)
            return configs

        return serialize_hints_to_configs(roller_hints)

    def apply_config(
        self,
        block_row_warps: Optional[int] = None,
        block_col_warps: Optional[int] = None,
        warp_row_tiles: Optional[int] = None,
        warp_col_tiles: Optional[int] = None,
        chunk: Optional[int] = None,
        num_stages: Optional[int] = None,
        enable_rasterization: bool = False,
    ):
        assert block_row_warps is not None, "block_row_warps is required"
        assert block_col_warps is not None, "block_col_warps is required"
        assert warp_row_tiles is not None, "warp_row_tiles is required"
        assert warp_col_tiles is not None, "warp_col_tiles is required"
        assert chunk is not None, "chunk is required"
        assert num_stages is not None, "num_stages is required"

        M = self.maybe_dynamic(self.M, "m")
        N, K = self.N, self.K
        assert isinstance(N, int) and isinstance(K, int), "Do not support dynamic N and K Currently"
        K = K // 2  # 2xint4 should be packed into one single int8
        trans_A, trans_B = self.trans_A, self.trans_B
        in_dtype, out_dtype, accum_dtype = self.in_dtype, self.out_dtype, self.accum_dtype
        assert in_dtype == "int4", "Only support int4 input"
        assert accum_dtype == "int32", "Only support int32 accumulation"
        with_bias = self.with_bias
        assert not with_bias, "Currently do not support bias"
        storage_dtype = "int8"

        # Calculate the micro size per warp using a helper function
        micro_size_x, micro_size_y, micro_size_k = get_mma_micro_size(storage_dtype)

        block_M = block_row_warps * warp_row_tiles
        block_N = block_col_warps * warp_col_tiles
        block_K = chunk

        # Define the shapes of matrices and shared memory buffers
        A_shape = (M, K)
        B_shape = (N, K)
        Bias_shape = (N,)
        C_shape = (M, N)
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
        mma_emitter = INT4TensorCoreIntrinEmitter(
            a_dtype=storage_dtype,
            b_dtype=storage_dtype,
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
                A: T.Buffer(A_shape, storage_dtype),
                B: T.Buffer(B_shape, storage_dtype),
                Bias: T.Buffer(Bias_shape, out_dtype),
                C: T.Buffer(C_shape, out_dtype),
        ):
            # Grid and thread configuration for CUDA kernel
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):

                # Allocate shared memory and local fragments
                A_shared = T.alloc_shared(A_shared_shape, storage_dtype, scope=shared_scope)
                B_shared = T.alloc_shared(B_shared_shape, storage_dtype, scope=shared_scope)
                C_shared = T.alloc_shared(C_shared_shape, out_dtype, scope=shared_scope)
                A_local = T.alloc_local((warp_rows * local_size_a), storage_dtype)
                B_local = T.alloc_local((warp_cols * local_size_b), storage_dtype)
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

        return self.post_process(main)

    def __post_init__(self):
        # Validate the matrix transpose settings
        assert self.trans_A is False, "Currently only support Matrix A not transposed"
        assert self.trans_B is True, "Currently only support Matrix B transposed"

        return


@dataclass
class MatmulINT4MMAWeightPropagationScheduler(MatmulMMAWeightPropagationScheduler):

    class TLHint(MatmulMMAWeightPropagationScheduler.TLHint):
        hint_type: str = "MatmulINT4MMAWeightPropagationScheduler"

    def get_roller_configs(self, arch: TileDevice = None, topk: int = 10):
        layout = f"{'t' if self.trans_A else 'n'}{'t' if self.trans_B else 'n'}"
        M = self.M
        K = self.K // 2  # 2xint4 should be packed into one single int8
        # Simple TIR Compute Expression
        storage_dtype = "int8"

        # This is a hack to utilize tensor core
        if isinstance(M, int) and M < 16:
            M = 16

        ir_module = matmul_select_implementation(
            M=M,
            N=self.N,
            K=K,
            in_dtype=storage_dtype,
            out_dtype=self.out_dtype,
            accum_dtype=self.accum_dtype,
            layout=layout,
            propagate_b=self.weight_transform_kind)

        roller_hints = get_roller_hints_from_func(
            ir_module,
            arch,
            topk,
            tensorcore_only=True,
            allow_gemv=True,
        )

        if roller_hints is None:
            raise ValueError("No Roller Hints Found for TensorCore Scheduling")

        def serialize_hints_to_configs(hints: List[Hint]):
            configs = []
            for hint in hints:
                config = self.TLHint.from_roller_hint(hint)
                configs.append(config)
            return configs

        return serialize_hints_to_configs(roller_hints)

    def apply_config(
        self,
        block_row_warps=2,
        block_col_warps=2,
        warp_row_tiles=32,
        warp_col_tiles=32,
        chunk=16,
        num_stages=2,
        enable_rasterization: bool = False,
    ):

        M = self.maybe_dynamic(self.M, "m")
        N, K = self.N, self.K
        assert isinstance(N, int) and isinstance(K, int), "Do not support dynamic N and K Currently"
        K = K // 2  # 2xint4 should be packed into one single int8
        trans_A, trans_B = self.trans_A, self.trans_B
        in_dtype, out_dtype, accum_dtype = self.in_dtype, self.out_dtype, self.accum_dtype
        assert in_dtype == "int4", "Only support int4 input"
        assert accum_dtype == "int32", "Only support int32 accumulation"
        storage_dtype = "int8"

        # Calculate the micro size per warp using a helper function
        micro_size_x, micro_size_y, micro_size_k = get_mma_micro_size(storage_dtype)

        block_M = block_row_warps * warp_row_tiles
        block_N = block_col_warps * warp_col_tiles
        block_K = chunk

        is_a_smooth = self.is_a_smooth
        is_b_smooth = self.is_b_smooth

        if is_a_smooth:
            A_shape = (M // micro_size_x, K // micro_size_k, micro_size_x, micro_size_k)
            A_shared_shape = (
                block_M // micro_size_x,
                block_K // micro_size_k,
                micro_size_x,
                micro_size_k,
            )
        else:
            A_shape = (M, K)
            A_shared_shape = (block_M, block_K)

        # Define the shapes of matrices and shared memory buffers
        B_shape = (N // micro_size_y, K // micro_size_k, micro_size_y, micro_size_k)
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
        mma_emitter = INT4TensorCoreIntrinEmitterWithLadderTransform(
            a_dtype=storage_dtype,
            b_dtype=storage_dtype,
            accum_dtype=accum_dtype,
            a_transposed=trans_A,
            b_transposed=trans_B,
            block_row_warps=block_row_warps,
            block_col_warps=block_col_warps,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=chunk,
            transform_kind_a=self.input_transform_kind,
            transform_kind_b=self.weight_transform_kind,
        )

        # Define the main kernel using the generated configuration
        @T.prim_func
        def main(
                A: T.Buffer(A_shape, storage_dtype),
                B: T.Buffer(B_shape, storage_dtype),
                C: T.Buffer((M, N), out_dtype),
        ):
            # Grid and thread configuration for CUDA kernel
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):

                # Allocate shared memory and local fragments
                A_shared = T.alloc_shared(A_shared_shape, storage_dtype, scope=shared_scope)
                B_shared = T.alloc_shared(B_shared_shape, storage_dtype, scope=shared_scope)
                C_shared = T.alloc_shared(C_shared_shape, out_dtype, scope=shared_scope)
                A_local = T.alloc_local((warp_rows * local_size_a), storage_dtype)
                B_local = T.alloc_local((warp_cols * local_size_b), storage_dtype)
                C_local = T.alloc_local((warp_rows * warp_cols * local_size_c), accum_dtype)

                # Thread-level parallelism for Tensor Cores
                thread_bindings = T.thread_binding(0, threads, "threadIdx.x")

                # Apply memory layout optimizations
                T.annotate_layout({
                    A_shared: make_swizzle_layout(A_shared, is_smooth=is_a_smooth),
                    B_shared: make_swizzle_layout(B_shared, is_smooth=is_b_smooth),
                })

                T.use_swizzle(panel_size=10, enable=enable_rasterization)

                # Initialize accumulation buffer to zero
                T.clear(C_local)

                # Main matrix multiplication pipeline with multiple stages
                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):

                    if is_a_smooth:
                        for i, k, ii, kk in T.Parallel(
                                block_M // micro_size_x,
                                block_K // micro_size_k,
                                micro_size_x,
                                micro_size_k,
                        ):
                            A_shared[i, k, ii, kk] = A[by * (block_M // micro_size_x) + i,
                                                       ko * (block_K // micro_size_k) + k, ii, kk]
                    else:
                        T.copy(A[by * block_M, ko * block_K], A_shared)

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

        return self.post_process(main)

    def __post_init__(self):
        # Validate the matrix transpose settings
        assert self.trans_A is False, "Currently only support Matrix A not transposed"
        assert self.trans_B is True, "Currently only support Matrix B transposed"

        return
