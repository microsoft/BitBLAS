# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# INT4 Tensor Core Implementation for NVIDIA GPUs
from bitblas import tvm as tvm
from tvm import DataType
import tvm.tl.language as T
from typing import Optional, List
from bitblas.tl.utils import (
    get_mma_micro_size,
    make_mma_swizzle_layout as make_swizzle_layout,
)
from bitblas.ops.general_matmul.tilelang.dense.matmul_tensorcore import (
    MatmulFineGrainScheduler,
    MatmulWeightPropagationScheduler,
)
from bitblas.tl.mma_macro_generator import (
    INT4TensorCoreIntrinEmitter,
    INT4TensorCoreIntrinEmitterWithLadderTransform,
)
from bitblas.base.arch import TileDevice
from bitblas.base.roller.hint import Hint
from bitblas.base.utils import get_roller_hints_from_func
from dataclasses import dataclass
from bitblas.ops.general_matmul.tirscript import (matmul_select_implementation)

# GPU warp configuration for NVIDIA GPUs
warp_size = 32


@dataclass
class MatmulINT4FineGrainScheduler(MatmulFineGrainScheduler):

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

        def serialze_hints_to_configs(hints: List[Hint]):
            configs = []
            for hint in hints:
                config = self.TLHint.from_roller_hint(hint)
                configs.append(config)
            return configs

        return serialze_hints_to_configs(roller_hints)

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
                    A_shared: make_swizzle_layout(A_shared),
                    B_shared: make_swizzle_layout(B_shared, is_smooth=True),
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
class MatmulINT4WeightPropagationScheduler(MatmulWeightPropagationScheduler):

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

        def serialze_hints_to_configs(hints: List[Hint]):
            configs = []
            for hint in hints:
                config = self.TLHint.from_roller_hint(hint)
                configs.append(config)
            return configs

        return serialze_hints_to_configs(roller_hints)

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

        # TODO(lei): Can be generalized to analyzed from bank size
        pad_factor = 8 if storage_dtype == "float16" else 16

        can_swizzle_a = block_K * DataType(storage_dtype).bits == 512
        apply_pad_a = not can_swizzle_a

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
                    A_shared: make_swizzle_layout(A_shared),
                    # B_shared: make_swizzle_layout(B_shared),
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
