# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.backends
from bitblas import tvm as tvm
from tvm import tl as TL
import tvm.tl.language as T
from bitblas.base import simplify_prim_func


def make_pad_layout(shared_buf, pad_offset=4):
    shape = shared_buf.shape
    stride = shape[-1]

    def transform(i, j):
        idx = i * (stride + pad_offset) + j
        return idx

    return T.Layout(shape, transform)


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

    # This is a debug config
    # block_row_warps = 2
    # block_col_warps = 2
    # warp_row_tiles = 64
    # warp_col_tiles = 64
    # chunk = 32

    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 32
    warp_col_tiles = 32
    chunk = 32

    # shared_scope = "shared.dyn"
    shared_scope = "shared"

    # Pipeline Stage
    stage = 1

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    A_shape = (M, K)
    B_shape = (N, K)
    C_shape = (M, N)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)
    C_shared_shape = (block_M, block_N)

    warp_size = 64
    threads = warp_size * (block_row_warps * block_col_warps)

    @T.prim_func
    def main(A: T.Buffer(A_shape, in_dtype), B: T.Buffer(B_shape, in_dtype),
             C: T.Buffer(C_shape, out_dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            A_local = T.alloc_fragment(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope=shared_scope)
            C_local = T.alloc_fragment(C_shared_shape, accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=stage):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.copy(A_shared, A_local)
                T.gemm(A_local, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def assert_tl_matmul_correctness(M, N, K, in_dtype, out_dtype, accum_dtype):
    matmul = tl_matmul(M, N, K, in_dtype, out_dtype, accum_dtype)
    print(matmul)
    mod, params = TL.lower(matmul, target="hip")
    src_code = mod.imported_modules[0].get_source()
    # src_code is the generated cuda source
    assert src_code is not None
    print(src_code)
    torch.random.manual_seed(0)
    if in_dtype == "int8":
        A = torch.randint(-128, 127, (M, K), device="cuda", dtype=torch.int8)
        B = torch.randint(-128, 127, (N, K), device="cuda", dtype=torch.int8)
    else:
        A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype))
        B = torch.rand(N, K, device="cuda", dtype=getattr(torch, in_dtype))
        # B = torch.ones((N, K), device="cuda", dtype=getattr(torch, in_dtype))
    print(f"{A=}")
    print(f"{B=}")
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))

    mod = TL.Profiler(mod, params, [], TL.TensorSupplyType.Integer)

    mod(A, B, C)

    # latency = mod.do_bench(mod.func, warmup=5, rep=10)

    # # Ensure that the latency is not None
    # assert latency is not None
    # print(f"{latency=}")
    # Get Reference Result
    ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, accum_dtype))
    print(C)
    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


def test_assert_tl_matmul():
    assert_tl_matmul_correctness(128, 256, 256, "float16", "float32", "float32")


if __name__ == "__main__":
    assert_tl_matmul_correctness(256, 256, 256, "float16", "float32", "float32")
