# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.backends
from bitblas import tvm as tvm
import bitblas.testing
from tvm import DataType
from bitblas import tilelang as tilelang
import tilelang.language as T
from bitblas.tl.utils import get_swizzle_layout
from bitblas.tl.mma_macro_generator import (
    TensorCoreIntrinEmitter,
    TensorCoreIntrinEmitterWithLadderTransform,
)
from bitblas.gpu.intrin.lop3 import decode_i4_to_f16
from bitblas.base import simplify_prim_func

torch.manual_seed(0)


def make_swizzle_layout(shared_buf):
    dtype = shared_buf.dtype
    shape = shared_buf.shape

    can_swizzle = shape[-1] * DataType(dtype).bits == 512
    if not can_swizzle:
        return T.Layout(shape, lambda *args: args)

    def transform_func(i, j):
        new_warp_i, new_warp_j = get_swizzle_layout(i, j, shape[-1], dtype)
        return [new_warp_i, new_warp_j]

    return T.Layout(shape, transform_func)


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

    # This is a debug config
    block_row_warps = 1
    block_col_warps = 1
    warp_row_tiles = 16
    warp_col_tiles = 16
    # chunk = 32 if in_dtype == "float16" else 64
    chunk = 32
    shared_scope = "shared.dyn"

    # Pipeline Stage
    stage = 2

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

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size_a = (micro_size_x * micro_size_k) // warp_size
    local_size_b = (micro_size_y * micro_size_k) // warp_size
    local_size_c = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    # MMA Wrapper to Auto Generate Code for MMA
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

            # Improve L2 Cache
            T.use_swizzle(panel_size=10)

            T.clear(C_local)

            for ko in T.Pipelined((K // block_K), num_stages=stage):

                # Load A into shared memory
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                # Load B into shared memory
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

                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_local, B_local, C_local)

            # Perform STMatrix
            mma_emitter.stmatrix(
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

    return main


def assert_tl_matmul_correctness(M, N, K, in_dtype, out_dtype, accum_dtype):
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

    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))

    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)

    mod(A, B, C)

    latency = mod.do_bench(mod.func, warmup=25)

    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, accum_dtype))
    print(C)
    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


def tl_matmul_with_block_reduce(
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

    # This is a debug config
    block_row_warps = 1
    block_col_warps = 1
    warp_row_tiles = 16
    warp_col_tiles = 16
    shared_scope = "shared.dyn"

    # Pipeline Stage
    stage = 2

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = 32 if in_dtype == "float16" else 64
    reduce_k = 2
    chunk = block_K // reduce_k

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

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

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
        reduce_k=reduce_k)

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
            reduced_accum_res = T.alloc_local(0, accum_dtype)

            thread_bindings = T.thread_binding(0, threads, "threadIdx.x")

            rk = T.thread_binding(0, reduce_k, "threadIdx.y")

            if block_K == 32:  # Swizzling only works for chunk size 32
                T.annotate_layout({
                    A_shared: make_swizzle_layout(A_shared),
                    B_shared: make_swizzle_layout(B_shared),
                })

            T.use_swizzle(panel_size=10)

            T.clear(C_local)

            for ko in T.Pipelined((K // block_K), num_stages=stage):

                # Load A into shared memory
                for i, k in T.Parallel(block_M, (block_K // reduce_k)):
                    vk = rk * (block_K // reduce_k) + k
                    A_shared[i, vk] = A[by * block_M + i, ko * block_K + vk]

                # Load B into shared memory
                for j, k in T.Parallel(block_N, (block_K // reduce_k)):
                    vk = rk * (block_K // reduce_k) + k
                    B_shared[j, vk] = B[bx * block_N + j, ko * block_K + vk]

                for ki in T.serial(0, (block_K // (micro_size_k * reduce_k))):

                    # Load A into fragment
                    mma_emitter.ldmatrix_a(
                        A_local,
                        A_shared,
                        ki,
                        thread_bindings=thread_bindings,
                        rk=rk,
                    )

                    # Load B into fragment
                    mma_emitter.ldmatrix_b(
                        B_local,
                        B_shared,
                        ki,
                        thread_bindings=thread_bindings,
                        rk=rk,
                    )

                    mma_emitter.mma(A_local, B_local, C_local)

            for n in T.serial(warp_rows * warp_cols * local_size):
                init_value = getattr(T, accum_dtype)(0)
                T.attr(
                    T.comm_reducer(lambda x, y: x + y, [init_value]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                )
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        C_local[n],
                        True,
                        reduced_accum_res[0],
                        rk,
                        dtype="handle",
                    ))
                if rk == 0:
                    C_local[n] = reduced_accum_res[0]

            if rk == 0:
                mma_emitter.stmatrix(
                    C_local,
                    C_shared,
                    thread_bindings=thread_bindings,
                )

            for i, j in T.Parallel(block_M, (block_N // reduce_k)):
                vj = rk * (block_N // reduce_k) + j
                C[by * block_M + i,
                  bx * block_N + vj] = C_shared[i // micro_size_x, vj // micro_size_y,
                                                i % micro_size_x, vj % micro_size_y]

    return main


def assert_tl_matmul_with_block_reduce_correctness(M, N, K, in_dtype, out_dtype, accum_dtype):
    matmul = tl_matmul_with_block_reduce(M, N, K, in_dtype, out_dtype, accum_dtype)

    mod, params = tilelang.lower(matmul)
    src_code = mod.imported_modules[0].get_source()

    # src_code is the generated cuda source
    assert src_code is not None

    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype))
    B = torch.rand(N, K, device="cuda", dtype=getattr(torch, in_dtype))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))

    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)

    mod(A, B, C)

    latency = mod.do_bench(mod.func, warmup=25)

    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A, B.T).to(getattr(torch, accum_dtype))
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


def tl_matmul_with_ladder_weight_only_transform(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    transform_b,
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

    # This is a debug config
    block_row_warps = 2
    block_col_warps = 2

    warp_rows = 2
    warp_cols = 2
    warp_row_tiles = micro_size_x * warp_rows
    warp_col_tiles = micro_size_y * warp_cols

    chunk = 64 if in_dtype == "float16" else 128
    shared_scope = "shared.dyn"

    # Pipeline Stage
    stage = 2

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    is_smooth_a = False
    can_swizzle = block_K * DataType(in_dtype).bits == 512
    apply_pad_a = not (is_smooth_a or can_swizzle)
    pad_factor = 8

    A_shape = (M, K)
    B_shape = (N // micro_size_y, K // micro_size_k, micro_size_y, micro_size_k)
    A_shared_shape = (block_M, (block_K + pad_factor) if apply_pad_a else block_K)
    B_shared_shape = (block_N // micro_size_y, block_K // micro_size_k, micro_size_y, micro_size_k)
    C_shared_shape = (
        block_M // micro_size_x,
        block_N // micro_size_y,
        micro_size_x,
        micro_size_y,
    )

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size_a = (micro_size_x * micro_size_k) // warp_size
    local_size_b = (micro_size_y * micro_size_k) // warp_size
    local_size_c = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    # MMA Wrapper to Auto Generate Code for MMA
    mma_emitter = TensorCoreIntrinEmitterWithLadderTransform(
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
        transform_kind_b=transform_b)

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
            })

            T.use_swizzle(panel_size=10)

            T.clear(C_local)

            for ko in T.Pipelined((K // block_K), num_stages=stage):
                # Load A into shared memory
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                # Load B into shared memory
                for j, k, jj, kk in T.Parallel(block_N // micro_size_y, block_K // micro_size_k,
                                               micro_size_y, micro_size_k):
                    B_shared[j, k, jj, kk] = B[bx * (block_N // micro_size_y) + j,
                                               ko * (block_K // micro_size_k) + k, jj, kk]

                for ki in T.serial(0, (block_K // micro_size_k)):

                    mma_emitter.ldmatrix_a(
                        A_local,
                        A_shared,
                        ki,
                        thread_bindings=thread_bindings,
                    )

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


def assert_tl_matmul_with_ladder_weight_only_transform_correctness(M, N, K, in_dtype, out_dtype,
                                                                   accum_dtype, transform_b):
    matmul = tl_matmul_with_ladder_weight_only_transform(M, N, K, in_dtype, out_dtype, accum_dtype,
                                                         transform_b)

    mod, params = tilelang.lower(matmul)
    src_code = mod.imported_modules[0].get_source()

    # src_code is the generated cuda source
    assert src_code is not None

    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype))
    B = torch.rand(N, K, device="cuda", dtype=getattr(torch, in_dtype))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))

    ladder_permutate_config = bitblas.ops.LadderPermutateConfig(
        M=N,
        N=K,
        transform_kind=transform_b,
        transpose_matrix=True,
    )

    ladder_permutate = bitblas.ops.LadderPermutate(ladder_permutate_config)

    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)

    LB = ladder_permutate(B.cpu()).cuda()

    mod(A, LB, C)

    latency = mod.do_bench(mod.func, warmup=25)

    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A, B.T).to(getattr(torch, accum_dtype))
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


def tl_matmul_with_ladder_weight_only_transform_block_reduce_int4(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    transform_b,
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
    num_bits = 4
    num_elems_per_byte = 8 // num_bits
    storage_dtype = "int8"

    micro_size_x = micro_size_y = micro_size_k = 16

    if out_dtype == "int32":
        micro_size_k = 32

    # This is a debug config
    block_row_warps = 1
    block_col_warps = 4

    warp_rows = 1
    warp_cols = 2
    warp_row_tiles = micro_size_x * warp_rows
    warp_col_tiles = micro_size_y * warp_cols
    shared_scope = "shared.dyn"

    # Pipeline Stage
    stage = 2
    reduce_k = 2

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = 32 if in_dtype == "float16" else 64
    chunk = block_K // reduce_k

    is_smooth_a = False
    can_swizzle = block_K * DataType(in_dtype).bits == 512
    apply_pad_a = not (is_smooth_a or can_swizzle)
    pad_factor = 8

    A_shape = (M, K)
    B_shape = (N // micro_size_y, K // micro_size_k, micro_size_y,
               micro_size_k // num_elems_per_byte)
    A_shared_shape = (block_M, (block_K + pad_factor) if apply_pad_a else block_K)
    B_shared_shape = (
        block_N // micro_size_y,
        block_K // micro_size_k,
        micro_size_y,
        micro_size_k // num_elems_per_byte,
    )
    C_shared_shape = (
        block_M // micro_size_x,
        block_N // micro_size_y,
        micro_size_x,
        micro_size_y,
    )

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size_a = (micro_size_x * micro_size_k) // warp_size
    local_size_b = (micro_size_y * micro_size_k) // warp_size
    local_size_c = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    # MMA Wrapper to Auto Generate Code for MMA
    mma_emitter = TensorCoreIntrinEmitterWithLadderTransform(
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
        reduce_k=reduce_k,
        transform_kind_b=transform_b,
        num_elems_per_byte=num_elems_per_byte)

    vec_load_qb = 16
    if block_N * (block_K // reduce_k) // num_elems_per_byte // threads < vec_load_qb:
        vec_load_qb = block_N * (block_K // reduce_k) // num_elems_per_byte // threads

    @T.prim_func
    def main(
            A: T.Buffer(A_shape, in_dtype),
            B: T.Buffer(B_shape, storage_dtype),
            C: T.Buffer((M, N), out_dtype),
    ):
        with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads,
                prelude=decode_i4_to_f16) as (bx, by):

            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, storage_dtype, scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, out_dtype, scope=shared_scope)
            A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)
            B_local = T.alloc_local((warp_cols * local_size_b // num_elems_per_byte), storage_dtype)
            B_dequantize_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size_c), accum_dtype)
            reduced_accum_res = T.alloc_local(0, accum_dtype)
            thread_bindings = T.thread_binding(0, threads, "threadIdx.x")
            rk = T.thread_binding(0, reduce_k, "threadIdx.y")

            T.annotate_layout({
                A_shared: make_swizzle_layout(A_shared),
            })

            T.use_swizzle(panel_size=10)

            T.clear(C_local)

            for ko in T.Pipelined((K // block_K), num_stages=stage):

                # Load A into shared memory
                for i, k in T.Parallel(block_M, (block_K // reduce_k)):
                    vk = rk * (block_K // reduce_k) + k
                    A_shared[i, vk] = A[by * block_M + i, ko * block_K + vk]

                # TODO(lei): Layout Inference Pass is not efficient to handle the four dims int8 load
                for i in T.serial(block_N * (block_K // reduce_k) // num_elems_per_byte //
                                  (threads * vec_load_qb)):
                    for v in T.vectorized(0, vec_load_qb):
                        t = thread_bindings
                        idx = i * threads * vec_load_qb * reduce_k + rk * threads * vec_load_qb + t * vec_load_qb + v
                        vkk = idx % (micro_size_k // num_elems_per_byte)
                        vjj = (idx // (micro_size_k // num_elems_per_byte)) % micro_size_y
                        vk = (idx // (micro_size_k // num_elems_per_byte) // micro_size_y) % (
                            block_K // micro_size_k)
                        vj = (idx // (micro_size_k // num_elems_per_byte) // micro_size_y //
                              (block_K // micro_size_k)) % (
                                  block_N // micro_size_y)
                        B_shared[vj, vk, vjj,
                                 vkk] = B[bx * (block_N // micro_size_y) + vj,
                                          ko * (block_K // micro_size_k) + vk, vjj, vkk]

                for ki in T.serial(0, (block_K // (micro_size_k * reduce_k))):

                    # Load A into fragment
                    mma_emitter.ldmatrix_a(
                        A_local,
                        A_shared,
                        ki,
                        thread_bindings=thread_bindings,
                        rk=rk,
                    )

                    # Load B into fragment
                    mma_emitter.ldmatrix_b(
                        B_local,
                        B_shared,
                        ki,
                        thread_bindings=thread_bindings,
                        rk=rk,
                    )

                    for j in T.serial(warp_cols):
                        T.call_extern(
                            'handle', 'decode_i4u_to_f16',
                            T.address_of(B_local[j * mma_emitter.local_size_b //
                                                 num_elems_per_byte]),
                            T.address_of(B_dequantize_local[j * mma_emitter.local_size_b]), 8)

                    mma_emitter.mma(A_local, B_dequantize_local, C_local)

            if reduce_k > 1:
                for n in T.serial(warp_rows * warp_cols * local_size_c):
                    T.attr(
                        T.comm_reducer(lambda x, y: x + y, [T.float16(0)]),
                        "reduce_scope",
                        T.reinterpret(T.uint64(0), dtype="handle"),
                    )
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1),
                            C_local[n],
                            True,
                            reduced_accum_res[0],
                            rk,
                            dtype="handle",
                        ))
                    if rk == 0:
                        C_local[n] = reduced_accum_res[0]

            if rk == 0:
                mma_emitter.stmatrix(
                    C_local,
                    C_shared,
                    thread_bindings=thread_bindings,
                )

            for i, j in T.Parallel(block_M, (block_N // reduce_k)):
                vj = rk * (block_N // reduce_k) + j
                C[by * block_M + i,
                  bx * block_N + vj] = C_shared[i // micro_size_x, vj // micro_size_y,
                                                i % micro_size_x, vj % micro_size_y]

    return main


def assert_tl_matmul_with_ladder_weight_only_transform_block_reduce_int4_correctness(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    transform_b,
):
    matmul = tl_matmul_with_ladder_weight_only_transform_block_reduce_int4(
        M, N, K, in_dtype, out_dtype, accum_dtype, transform_b)

    mod, params = tilelang.lower(matmul)
    src_code = mod.imported_modules[0].get_source()

    # src_code is the generated cuda source
    assert src_code is not None
    num_bits = 4
    num_elems_per_byte = 8 // num_bits
    storage_dtype = "int8"

    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype))
    qB = torch.randint(
        0, 127, (N, K // num_elems_per_byte), device="cuda", dtype=getattr(torch, storage_dtype))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))

    ladder_permutate_config = bitblas.ops.LadderPermutateConfig(
        M=N,
        N=K,
        transform_kind=transform_b,
        transpose_matrix=True,
        dequantize_bits=num_bits,
        storage_dtype=storage_dtype,
    )

    ladder_permutate = bitblas.ops.LadderPermutate(ladder_permutate_config)

    lop3_permutate_config = bitblas.ops.LOP3PermutateConfig(
        M=N,
        N=K,
        datatype=in_dtype,
        dequantize_bits=num_bits,
        storage_dtype=storage_dtype,
    )
    lop3_permutate = bitblas.ops.LOP3Permutate(
        config=lop3_permutate_config,
        target=tvm.target.Target("llvm"),
    )
    QLB = ladder_permutate(qB.cpu()).cuda()
    QLB = lop3_permutate(QLB.cpu()).cuda()

    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)

    mod(A, QLB, C)

    latency = mod.do_bench(mod.func, warmup=25)

    # Ensure that the latency is not None
    assert latency is not None

    B = (
        torch.zeros(qB.shape[0], qB.shape[1] * 8 // 4,
                    dtype=torch.half).to(torch.half).to(A.device))
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i][j] = ((qB[i][j // 2] >> (4 * (j % 2))) & 0xF).to(torch.half)
    # Get Reference Result
    ref_c = torch.matmul(A, B.T).to(getattr(torch, accum_dtype))
    print(C)
    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


@simplify_prim_func
def tl_matmul_with_ladder_input_weight_transform(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    transform_a,
    transform_b,
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
    assert transform_a > 0, "transform_a should be greater than 0"
    micro_size_x = micro_size_y = micro_size_k = 16

    if out_dtype == "int32":
        micro_size_k = 32

    # This is a debug config
    block_row_warps = 2
    block_col_warps = 2

    warp_rows = 1
    warp_cols = 1
    warp_row_tiles = micro_size_x * warp_rows
    warp_col_tiles = micro_size_y * warp_cols

    chunk = 32 if in_dtype == "float16" else 64
    shared_scope = "shared"

    # Pipeline Stage
    stage = 2

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    A_shape = (M // micro_size_x, K // micro_size_k, micro_size_x, micro_size_k)
    B_shape = (N // micro_size_y, K // micro_size_k, micro_size_y, micro_size_k)
    A_shared_shape = (block_M // micro_size_x, block_K // micro_size_k, micro_size_x, micro_size_k)
    B_shared_shape = (block_N // micro_size_y, block_K // micro_size_k, micro_size_y, micro_size_k)
    C_shared_shape = (
        block_M // micro_size_x,
        block_N // micro_size_y,
        micro_size_x,
        micro_size_y,
    )

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size_a = (micro_size_x * micro_size_k) // warp_size
    local_size_b = (micro_size_y * micro_size_k) // warp_size
    local_size_c = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    # MMA Wrapper to Auto Generate Code for MMA
    mma_emitter = TensorCoreIntrinEmitterWithLadderTransform(
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
        transform_kind_a=transform_a,
        transform_kind_b=transform_b)

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

            T.use_swizzle(panel_size=10)

            T.clear(C_local)

            for ko in T.Pipelined((K // block_K), num_stages=stage):
                # Load A into shared memory
                for i, k, ii, kk in T.Parallel(block_M // micro_size_x, block_K // micro_size_k,
                                               micro_size_x, micro_size_k):
                    A_shared[i, k, ii, kk] = A[by * (block_M // micro_size_x) + i,
                                               ko * (block_K // micro_size_k) + k, ii, kk]

                # Load B into shared memory
                for j, k, jj, kk in T.Parallel(block_N // micro_size_y, block_K // micro_size_k,
                                               micro_size_y, micro_size_k):
                    B_shared[j, k, jj, kk] = B[bx * (block_N // micro_size_y) + j,
                                               ko * (block_K // micro_size_k) + k, jj, kk]

                for ki in T.serial(0, (block_K // micro_size_k)):

                    mma_emitter.ldmatrix_a(
                        A_local,
                        A_shared,
                        ki,
                        thread_bindings=thread_bindings,
                    )

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


def assert_tl_matmul_with_ladder_input_weight_transform_correctness(M, N, K, in_dtype, out_dtype,
                                                                    accum_dtype, transform_a,
                                                                    transform_b):
    matmul = tl_matmul_with_ladder_input_weight_transform(M, N, K, in_dtype, out_dtype, accum_dtype,
                                                          transform_a, transform_b)

    mod, params = tilelang.lower(matmul)
    src_code = mod.imported_modules[0].get_source()
    # src_code is the generated cuda source
    assert src_code is not None

    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype))
    B = torch.rand(N, K, device="cuda", dtype=getattr(torch, in_dtype))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))

    ladder_permutate_config_A = bitblas.ops.LadderPermutateConfig(
        M=M,
        N=K,
        datatype=in_dtype,
        storage_dtype=in_dtype,
        propagate_kind="A",
        transpose_matrix=False,
        transform_kind=transform_a,
    )

    ladder_permutate_a = bitblas.ops.LadderPermutate(ladder_permutate_config_A)

    ladder_permutate_config_B = bitblas.ops.LadderPermutateConfig(
        M=N,
        N=K,
        datatype=in_dtype,
        storage_dtype=in_dtype,
        propagate_kind="B",
        transform_kind=transform_b,
        transpose_matrix=True,
    )

    ladder_permutate_b = bitblas.ops.LadderPermutate(ladder_permutate_config_B)

    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)

    LA = ladder_permutate_a(A.cpu()).cuda()
    LB = ladder_permutate_b(B.cpu()).cuda()

    mod(LA, LB, C)

    latency = mod.do_bench(mod.func, warmup=25)

    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A, B.T).to(getattr(torch, accum_dtype))
    # print("Ref C: ", ref_c)
    # print("C: ", C)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


def test_assert_tl_matmul():
    assert_tl_matmul_correctness(128, 128, 128, "float16", "float16", "float16")
    assert_tl_matmul_correctness(128, 256, 256, "float16", "float32", "float32")
    assert_tl_matmul_correctness(128, 256, 256, "int8", "int32", "int32")


def test_assert_tl_matmul_with_block_reduce():
    assert_tl_matmul_with_block_reduce_correctness(128, 128, 128, "float16", "float16", "float16")
    assert_tl_matmul_with_block_reduce_correctness(128, 256, 256, "float16", "float32", "float32")


def test_assert_assert_tl_matmul_with_ladder_weight_only_transform():
    assert_tl_matmul_with_ladder_weight_only_transform_correctness(256, 256, 256, "float16",
                                                                   "float16", "float16", 3)
    assert_tl_matmul_with_ladder_weight_only_transform_correctness(256, 256, 256, "float16",
                                                                   "float32", "float32", 3)


def test_assert_tl_matmul_with_ladder_weight_only_transform_block_reduce_int4():
    assert_tl_matmul_with_ladder_weight_only_transform_block_reduce_int4_correctness(
        256, 1024, 512, "float16", "float16", "float16", 3)


def test_assert_tl_matmul_with_ladder_input_weight_transform():
    assert_tl_matmul_with_ladder_input_weight_transform_correctness(256, 256, 256, "float16",
                                                                    "float16", "float16", 2, 3)


if __name__ == "__main__":
    # bitblas.testing.main()
    test_assert_tl_matmul_with_ladder_input_weight_transform()
