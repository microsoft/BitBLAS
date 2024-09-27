# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm as tvm
from tvm import DataType
import tvm.tl.language as T

from bitblas.tl.utils import (
    get_mma_micro_size,
    make_swizzle_layout,
)

from bitblas.tl.macro_generator import (
    TensorCoreIntrinEmitter,
    TensorCoreIntrinEmitterWithLadderTransform,
)

from bitblas.ops.operator import TransformKind


def matmul_blocked(
        M,
        N,
        K,
        block_M=64,
        block_N=64,
        block_K=32,
        trans_A=False,
        trans_B=False,
        dtypeAB="float16",
        dtypeC="float16",
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
            A: T.Buffer(A_shape, dtypeAB),
            B: T.Buffer(B_shape, dtypeAB),
            C: T.Buffer((M, N), dtypeC),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, dtypeAB)
            B_shared = T.alloc_shared(B_shared_shape, dtypeAB)
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
    dtypeAB,
    dtypeC,
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

    micro_size_x, micro_size_y, micro_size_k = get_mma_micro_size(dtypeAB)

    A_shape = (M, K)
    B_shape = (N, K)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)
    C_shared_shape = (block_M // micro_size_x, block_N // micro_size_y, micro_size_x, micro_size_y)

    warp_size = 32  # nvidia gpu warp size is 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    shared_scope = "shared.dyn"  # Literal["shared", "shared.dyn"] while shared for static shared memory
    mma_emitter = TensorCoreIntrinEmitter(
        a_dtype=dtypeAB,
        b_dtype=dtypeAB,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk)

    @T.prim_func
    def main(
            A: T.Buffer(A_shape, dtypeAB),
            B: T.Buffer(B_shape, dtypeAB),
            C: T.Buffer((M, N), dtypeC),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):

            A_shared = T.alloc_shared(A_shared_shape, dtypeAB, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, dtypeAB, scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, dtypeC, scope=shared_scope)
            A_local = T.alloc_local((warp_rows * local_size), dtypeAB)
            B_local = T.alloc_local((warp_cols * local_size), dtypeAB)
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
                                               i % micro_size_x, j % micro_size_y]

    return main


def matmul_macro_tensorcore_weight_propagation_level_ldmatrix(
    M,
    N,
    K,
    dtypeAB,
    dtypeC,
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
    pad_factor = 8 if dtypeAB == "float16" else 16

    can_swizzle_a = block_K * DataType(dtypeAB).bits == 512
    apply_pad_a = not can_swizzle_a

    micro_size_x, micro_size_y, micro_size_k = get_mma_micro_size(dtypeAB)

    A_shape = (M, K)
    B_shape = (N // micro_size_y, K // micro_size_k, micro_size_y, micro_size_k)
    A_shared_shape = (block_M, (block_K + pad_factor) if apply_pad_a else block_K)
    B_shared_shape = (block_N // micro_size_y, block_K // micro_size_k, micro_size_y, micro_size_k)
    C_shared_shape = (block_M // micro_size_x, block_N // micro_size_y, micro_size_x, micro_size_y)

    warp_size = 32  # nvidia gpu warp size is 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    shared_scope = "shared.dyn"  # Literal["shared", "shared.dyn"] while shared for static shared memory
    mma_emitter = TensorCoreIntrinEmitterWithLadderTransform(
        a_dtype=dtypeAB,
        b_dtype=dtypeAB,
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
            A: T.Buffer(A_shape, dtypeAB),
            B: T.Buffer(B_shape, dtypeAB),
            C: T.Buffer((M, N), dtypeC),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):

            A_shared = T.alloc_shared(A_shared_shape, dtypeAB, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, dtypeAB, scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, dtypeC, scope=shared_scope)
            A_local = T.alloc_local((warp_rows * local_size), dtypeAB)
            B_local = T.alloc_local((warp_cols * local_size), dtypeAB)
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

                for j, k, jj, kk in T.Parallel(block_N // micro_size_y, block_K // micro_size_k,
                                               micro_size_y, micro_size_k):
                    B_shared[j, k, jj, kk] = B[bx * (block_N // micro_size_y) + j,
                                               ko * (block_K // micro_size_k) + k, jj, kk]

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
                                               i % micro_size_x, j % micro_size_y]

    return main
