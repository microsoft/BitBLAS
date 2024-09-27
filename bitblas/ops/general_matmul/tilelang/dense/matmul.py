# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas import tvm as tvm
import tvm.tl.language as T

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
    enable_rasterization=False, # Enhance L2 Locality
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    def maybe_pipeline(
        iterable, num_stages,
    ):
        enable_pipeline = num_stages > 1
        if enable_pipeline:
            return T.Pipelined(iterable, num_stages=num_stages)
        else:
            return T.serial(iterable)
        
    @T.prim_func
    def main(
        A: T.Buffer(A_shape, dtypeAB),
        B: T.Buffer(B_shape, dtypeAB),
        C: T.Buffer((M, N), dtypeC),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, dtypeAB)
            B_shared = T.alloc_shared(B_shared_shape, dtypeAB)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            if enable_rasterization:
                # rasterization factor
                T.use_swizzle(10)

            T.clear(C_local)
            for k in maybe_pipeline(T.ceildiv(K, block_K), num_stages):
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