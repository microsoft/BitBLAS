# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.backends
from bitblas import tvm as tvm
import bitblas.testing
from tvm import DataType
from tvm import tl as TL
import tvm.tl.language as T
from bitblas.tl.utils import (make_swizzle_layout, index_to_coordinates)
from bitblas.gpu.intrin.lop3 import decode_i2s_to_i4s

from bitblas.tl.mma_macro_generator import (
    INT4TensorCoreIntrinEmitter,)
from bitblas.ops.base_scheduler import simplify_prim_func

torch.manual_seed(0)


@simplify_prim_func
def tl_matmul(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    fast_decoding=True,
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

    K = K // 2

    micro_size_x = micro_size_y = micro_size_k = 16

    if accum_dtype == "int32":
        micro_size_k = 32

    num_elems_per_byte = 2
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    local_size = MAX_TRANSACTION_SIZE_IN_BITS // DataType(in_dtype).bits
    local_size_compressed = local_size // num_elems_per_byte

    # This is a debug config
    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 32
    warp_col_tiles = 32
    chunk = 32 if in_dtype == "float16" else 64
    shared_scope = "shared.dyn"
    storage_dtype = "int8"

    # Pipeline Stage
    stage = 2

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    A_shape = (M, K)  # int8 storage represents int4*2
    B_shape = (N, K // num_elems_per_byte)  # int8 storage represents int4*2
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K // num_elems_per_byte)
    B_dequantize_shared_shape = (block_N, block_K)
    C_shared_shape = (
        block_M // micro_size_x,
        block_N // micro_size_y,
        micro_size_x,
        micro_size_y,
    )

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    fragement_size_a = (micro_size_x * micro_size_k) // warp_size
    fragement_size_b = (micro_size_y * micro_size_k) // warp_size
    fragement_size_c = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    # MMA Wrapper to Auto Generate Code for MMA
    mma_emitter = INT4TensorCoreIntrinEmitter(
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
            B: T.Buffer(B_shape, storage_dtype),
            C: T.Buffer((M, N), out_dtype),
    ):
        with T.Kernel(
                T.ceildiv(N, block_N),
                T.ceildiv(M, block_M),
                threads=threads,
                prelude=decode_i2s_to_i4s) as (bx, by):

            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, storage_dtype, scope=shared_scope)
            B_dequantize_shared = T.alloc_shared(
                B_dequantize_shared_shape, in_dtype, scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, out_dtype, scope=shared_scope)
            A_frag = T.alloc_local((warp_rows * fragement_size_a), in_dtype)
            B_frag = T.alloc_local((warp_cols * fragement_size_b), in_dtype)
            C_frag = T.alloc_local((warp_rows * warp_cols * fragement_size_c), accum_dtype)

            B_local = T.alloc_local([local_size_compressed], storage_dtype)
            B_dequantize_local = T.alloc_local([local_size], in_dtype)

            thread_bindings = T.thread_binding(0, threads, "threadIdx.x")

            T.annotate_layout({
                A_shared: make_swizzle_layout(A_shared),
                B_dequantize_shared: make_swizzle_layout(B_dequantize_shared),
            })

            # Improve L2 Cache
            T.use_swizzle(panel_size=10)

            T.clear(C_frag)

            for ko in T.Pipelined((K // block_K), num_stages=stage):

                # Load A into shared memory
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                # Load B into shared memory
                for j, k in T.Parallel(block_N, block_K // num_elems_per_byte):
                    B_shared[j, k] = B[bx * block_N + j, ko * (block_K // num_elems_per_byte) + k]

                for i in T.serial(block_N * block_K // num_elems_per_byte //
                                  (threads * local_size_compressed)):
                    for v in T.vectorized(0, local_size_compressed):
                        index = (
                            i * threads * local_size_compressed +
                            thread_bindings * local_size_compressed + v)
                        vi, vj = index_to_coordinates(index, B_shared_shape)
                        B_local[v] = B_shared[vi, vj]

                    if fast_decoding:
                        T.call_extern('handle', 'decode_i2u_to_i4s', T.address_of(B_local[0]),
                                      T.address_of(B_dequantize_local[0]), 32)
                    else:
                        for v in T.serial(0, local_size):
                            int2x2_value = (B_local[v // 2] >> ((v % 2) * 4)) & 0x0F

                            int4_0 = (int2x2_value >> 0) & 0x03
                            int4_1 = (int2x2_value >> 2) & 0x03

                            B_dequantize_local[v] = (int4_1 << 4) | int4_0

                    for v in T.vectorized(0, local_size):
                        index = i * threads * local_size + thread_bindings * local_size + v
                        vi, vj = index_to_coordinates(index, B_dequantize_shared_shape)
                        B_dequantize_shared[vi, vj] = B_dequantize_local[v]

                for ki in T.serial(0, (block_K // micro_size_k)):

                    # Load A into fragment
                    mma_emitter.ldmatrix_a(
                        A_frag,
                        A_shared,
                        ki,
                        thread_bindings=thread_bindings,
                    )

                    # Load B into fragment
                    mma_emitter.ldmatrix_b(
                        B_frag,
                        B_dequantize_shared,
                        ki,
                        thread_bindings=thread_bindings,
                    )

                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_frag, B_frag, C_frag)

            # Perform STMatrix
            mma_emitter.stmatrix(
                C_frag,
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

    return main


def assert_tl_matmul_correctness(M, N, K, in_dtype, out_dtype, accum_dtype, fast_decoding=True):
    matmul = tl_matmul(M, N, K, in_dtype, out_dtype, accum_dtype, fast_decoding)
    print(matmul)
    mod, params = TL.lower(matmul)
    src_code = mod.imported_modules[0].get_source()
    # src_code is the generated cuda source
    assert src_code is not None
    print(src_code)
    # A = torch.ones(M, K, device="cuda", dtype=getattr(torch, in_dtype))
    # B = torch.ones(N, K, device="cuda", dtype=getattr(torch, in_dtype))
    A = torch.randint(0, 4, (M, K), device="cuda", dtype=getattr(torch, in_dtype))
    B = torch.randint(0, 2, (N, K), device="cuda", dtype=getattr(torch, in_dtype))

    lop3_permutate_config = bitblas.ops.LOP3PermutateConfig(
        M=N,
        N=K,
        datatype="int4",
        dequantize_bits=2,
        storage_dtype="int8",
    )
    lop3_permutate = bitblas.ops.LOP3Permutate(
        config=lop3_permutate_config,
        target=tvm.target.Target("llvm"),
    )

    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))

    compressed_A = (A[:, ::2] & 0x0F) + ((A[:, 1::2] & 0x0F) << 4)
    compressed_B = (B[:, ::4] & 0x03) + ((B[:, 1::4] & 0x03) << 2) + ((B[:, 2::4] & 0x03) << 4) + (
        (B[:, 3::4] & 0x03) << 6)

    mod = TL.Profiler(mod, params, [], TL.TensorSupplyType.Integer)
    print(f"{compressed_B=}")
    lop3_compressed_B = lop3_permutate(compressed_B.cpu()).cuda()
    print(f"{lop3_compressed_B=}")
    mod(compressed_A, lop3_compressed_B, C)
    print(C)
    latency = mod.do_bench(mod.func, warmup=25, profiler="tvm")
    print(latency)
    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, accum_dtype))

    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


def test_assert_tl_matmul():
    assert_tl_matmul_correctness(128, 128, 128, "float16", "float16", "float16")
    assert_tl_matmul_correctness(128, 256, 256, "float16", "float32", "float32")


if __name__ == "__main__":
    # bitblas.testing.main()
    # assert_tl_matmul_correctness(128, 128, 128, "float16", "float16", "float16")
    # assert_tl_matmul_correctness(16384, 16384, 16384, "int8", "int32", "int32")
    # assert_tl_matmul_correctness(16384, 16384, 16384, "int8", "int32", "int32")
    assert_tl_matmul_correctness(256, 256, 256, "int8", "int32", "int32")
    # assert_tl_matmul_correctness(16384, 16384, 16384, "int8", "int32", "int32")
