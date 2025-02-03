# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.backends
from bitblas import tvm as tvm
import bitblas.testing
from bitblas import tilelang as tilelang
import tilelang.language as T
from bitblas.tl.utils import make_mfma_swizzle_layout as make_swizzle_layout
from bitblas.tl.mfma_macro_generator import (
    MatrixCoreIntrinEmitter,)
from bitblas.base import simplify_prim_func

torch.manual_seed(0)


@simplify_prim_func
def tl_matmul(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
):
    assert in_dtype in [
        "float16",
        "int8",
    ], "Currently only float16 and int8 are supported"
    assert out_dtype in [
        "float16",
        "float32",
        "int32",
    ], "Currently only float16, float32 and int32 are supported"

    micro_size_x = micro_size_y = micro_size_k = 16

    if out_dtype == "int32":
        micro_size_k = 32

    block_row_warps = 1
    block_col_warps = 1
    warp_row_tiles = 16
    warp_col_tiles = 16
    chunk = 32
    shared_scope = "shared.dyn"
    cache_write_shared = False

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

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

    warp_size = 64
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size_a = (micro_size_x * micro_size_k) // warp_size
    local_size_b = (micro_size_y * micro_size_k) // warp_size
    local_size_c = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    # MMA Wrapper to Auto Generate Code for MMA
    mfma_emitter = MatrixCoreIntrinEmitter(
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

            # Improve L2 Cache
            T.use_swizzle(panel_size=10)

            T.clear(C_local)

            for ko in T.Pipelined((K // block_K), num_stages=0):

                # Load A into shared memory
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                # Load B into shared memory
                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

                for ki in T.serial(0, (block_K // micro_size_k)):

                    # Load A into fragment
                    mfma_emitter.ldmatrix_a(
                        A_local,
                        A_shared,
                        ki,
                        thread_bindings=thread_bindings,
                    )

                    # Load B into fragment
                    mfma_emitter.ldmatrix_b(
                        B_local,
                        B_shared,
                        ki,
                        thread_bindings=thread_bindings,
                    )

                    # Perform Matrix Multiplication
                    mfma_emitter.mfma(A_local, B_local, C_local)

            # Perform STMatrix
            if cache_write_shared:
                mfma_emitter.stmatrix(
                    C_local,
                    C_shared,
                    thread_bindings=thread_bindings,
                )

                # Store shared into global
                for i, j in T.Parallel(block_M, block_N):
                    C[by * block_M + i, bx * block_N + j] = C_shared[
                        i // micro_size_x,
                        j // micro_size_y,
                        i % micro_size_x,
                        j % micro_size_y,
                    ]
            else:
                mfma_emitter.stmatrix(
                    C_local,
                    C,
                    thread_bindings=thread_bindings,
                    pid_m=by,
                    pid_n=bx,
                )

    return main


def assert_tl_matmul_correctness(M, N, K, in_dtype, out_dtype, accum_dtype="float32"):
    matmul = tl_matmul(M, N, K, in_dtype, out_dtype, accum_dtype)
    mod, params = tilelang.lower(matmul)
    src_code = mod.imported_modules[0].get_source()
    # src_code is the generated cuda source
    assert src_code is not None

    if in_dtype == "int8":
        A = torch.randint(-128, 127, (M, K), device="cuda", dtype=torch.int8)
        B = torch.randint(-128, 127, (N, K), device="cuda", dtype=torch.int8)
    else:
        A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype))
        B = torch.rand(N, K, device="cuda", dtype=getattr(torch, in_dtype))

    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, out_dtype))

    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)

    mod(A, B, C)

    latency = mod.do_bench(mod.func, warmup=25)

    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, out_dtype))
    print(C)
    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


@bitblas.testing.requires_rocm
def test_assert_tl_matmul():
    assert_tl_matmul_correctness(128, 128, 128, "float16", "float16")
    assert_tl_matmul_correctness(128, 256, 256, "float16", "float32")


if __name__ == "__main__":
    bitblas.testing.main()
