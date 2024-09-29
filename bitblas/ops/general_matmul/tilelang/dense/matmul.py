# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm as tvm
from tvm import DataType
import tvm.tl.language as T
from typing import Optional
from bitblas.tl.utils import (
    get_mma_micro_size,
    make_swizzle_layout,
)

from bitblas.tl.macro_generator import (
    TensorCoreIntrinEmitter,
    TensorCoreIntrinEmitterWithLadderTransform,
)
from bitblas.ops.common import TransformKind
from bitblas.ops.base_scheduler import BaseScheduler

from dataclasses import dataclass


@dataclass
class MatmulScheduler(BaseScheduler):

    # OP Related Config
    M: Optional[int] = None
    N: Optional[int] = None
    K: Optional[int] = None
    trans_A: bool = False
    trans_B: bool = False
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float16"

    # Default Tile Related Params
    block_M: int = 64
    block_N: int = 64
    block_K: int = 32
    num_stages: int = 2
    threads: int = 128
    enable_rasterization: bool = False  # Enhance L2 Locality

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
        block_M=64,
        block_N=64,
        block_K=32,
        num_stages=2,
        threads=128,
        # Enhance L2 Locality
        enable_rasterization=False,
    ):
        M, N, K = self.M, self.N, self.K
        trans_A, trans_B = self.trans_A, self.trans_B
        in_dtype, out_dtype, accum_dtype = self.in_dtype, self.out_dtype, self.accum_dtype

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
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared(A_shared_shape, in_dtype)
                B_shared = T.alloc_shared(B_shared_shape, in_dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                if enable_rasterization:
                    # rasterization factor
                    T.use_swizzle(10)

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

        return self.maybe_simplify(main)

    def __post_init__(self):
        # Add Config Validation
        return


@dataclass
class MatmulFineGrainScheduler(BaseScheduler):
    # Fine-grained matrix multiplication scheduler
    # Allows for more detailed configuration.

    # Operation Configuration
    M: Optional[int] = None
    N: Optional[int] = None
    K: Optional[int] = None
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    trans_A: bool = False
    trans_B: bool = True
    accum_dtype: str = "float16"

    # Tensor Core Warp Configuration
    block_row_warps: int = 2
    block_col_warps: int = 2
    warp_row_tiles: int = 32
    warp_col_tiles: int = 32
    chunk: int = 32  # Usually determines the K-dimension split size

    # Tiling and Other Optimization Parameters
    num_stages: int = 2
    enable_rasterization: bool = False

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

        # GPU warp configuration for NVIDIA GPUs
        warp_size = 32
        threads = warp_size * (block_row_warps * block_col_warps)

        # Calculate local fragment sizes for tensor core
        local_size = (micro_size_x * micro_size_y) // warp_size
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
                A_local = T.alloc_local((warp_rows * local_size), in_dtype)
                B_local = T.alloc_local((warp_cols * local_size), in_dtype)
                C_local = T.alloc_local((warp_rows * warp_cols * local_size), accum_dtype)

                # Thread-level parallelism for Tensor Cores
                thread_bindings = T.thread_binding(0, threads, "threadIdx.x")

                # Apply memory layout optimizations
                T.annotate_layout({
                    A_shared: make_swizzle_layout(A_shared),
                    B_shared: make_swizzle_layout(B_shared),
                })

                # Optional rasterization for L2 locality enhancement
                if enable_rasterization:
                    T.use_swizzle(panel_size=10)

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
                    C[by * block_M + i,
                      bx * block_N + j] = C_shared[i // micro_size_x, j // micro_size_y,
                                                   i % micro_size_x, j % micro_size_y,]

        return self.maybe_simplify(main)

    def __post_init__(self):
        # Validate the matrix transpose settings
        assert self.trans_A is False, "Currently only support Matrix A not transposed"
        assert self.trans_B is True, "Currently only support Matrix B transposed"

        return


@dataclass
class MatmulWeightPropagationScheduler(BaseScheduler):
    # Fine-grained matrix multiplication scheduler
    # Allows for more detailed configuration.

    # Operation Configuration
    M: Optional[int] = None
    N: Optional[int] = None
    K: Optional[int] = None
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    trans_A: bool = False
    trans_B: bool = True
    accum_dtype: str = "float16"

    # Tensor Core Warp Configuration
    block_row_warps: int = 2
    block_col_warps: int = 2
    warp_row_tiles: int = 32
    warp_col_tiles: int = 32
    chunk: int = 32  # Usually determines the K-dimension split size

    # Tiling and Other Optimization Parameters
    num_stages: int = 2
    enable_rasterization: bool = False

    def with_default_config(self):
        block_row_warps = getattr(self, "block_row_warps", 2)
        block_col_warps = getattr(self, "block_col_warps", 2)
        warp_row_tiles = getattr(self, "warp_row_tiles", 4)
        warp_col_tiles = getattr(self, "warp_col_tiles", 4)
        chunk = getattr(self, "chunk", 16)
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
        local_size = (micro_size_x * micro_size_y) // warp_size
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
            transform_kind_b=TransformKind.LDMatrixTransform,
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
                A_local = T.alloc_local((warp_rows * local_size), in_dtype)
                B_local = T.alloc_local((warp_cols * local_size), in_dtype)
                C_local = T.alloc_local((warp_rows * warp_cols * local_size), accum_dtype)

                # Thread-level parallelism for Tensor Cores
                thread_bindings = T.thread_binding(0, threads, "threadIdx.x")

                # Apply memory layout optimizations
                T.annotate_layout({
                    A_shared: make_swizzle_layout(A_shared),
                    B_shared: make_swizzle_layout(B_shared),
                })

                # Optional rasterization for L2 locality enhancement
                if enable_rasterization:
                    T.use_swizzle(panel_size=10)

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
                        B_shared[j, k, jj, kk] = B[bx * (block_N // micro_size_y) + j,
                                                   ko * (block_K // micro_size_k) + k, jj, kk,]

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
                    C[by * block_M + i,
                      bx * block_N + j] = C_shared[i // micro_size_x, j // micro_size_y,
                                                   i % micro_size_x, j % micro_size_y,]

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

            if enable_rasterization:
                # rasterization factor
                T.use_swizzle(10)

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
    local_size = (micro_size_x * micro_size_y) // warp_size
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
            A_local = T.alloc_local((warp_rows * local_size), in_dtype)
            B_local = T.alloc_local((warp_cols * local_size), in_dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size), accum_dtype)
            thread_bindings = T.thread_binding(0, threads, "threadIdx.x")

            T.annotate_layout({
                A_shared: make_swizzle_layout(A_shared),
                B_shared: make_swizzle_layout(B_shared),
            })

            if enable_rasterization:
                T.use_swizzle(panel_size=10)

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
                C[by * block_M + i,
                  bx * block_N + j] = C_shared[i // micro_size_x, j // micro_size_y,
                                               i % micro_size_x, j % micro_size_y,]

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
    local_size = (micro_size_x * micro_size_y) // warp_size
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
            A_local = T.alloc_local((warp_rows * local_size), in_dtype)
            B_local = T.alloc_local((warp_cols * local_size), in_dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size), accum_dtype)
            thread_bindings = T.thread_binding(0, threads, "threadIdx.x")

            T.annotate_layout({
                A_shared: make_swizzle_layout(A_shared),
                B_shared: make_swizzle_layout(B_shared),
            })

            if enable_rasterization:
                T.use_swizzle(panel_size=10)

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
                    B_shared[j, k, jj, kk] = B[bx * (block_N // micro_size_y) + j,
                                               ko * (block_K // micro_size_k) + k, jj, kk,]

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
                C[by * block_M + i,
                  bx * block_N + j] = C_shared[i // micro_size_x, j // micro_size_y,
                                               i % micro_size_x, j % micro_size_y,]

    return main
