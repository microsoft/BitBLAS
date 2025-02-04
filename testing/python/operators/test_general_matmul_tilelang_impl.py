# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas import tvm as tvm
from bitblas import tilelang as tilelang
import bitblas.testing
from bitblas.ops.general_matmul.tilelang.dense.matmul_tile import (
    matmul_blocked,
    matmul_macro_tensorcore,
    matmul_macro_tensorcore_weight_propagation_level_ldmatrix,
)

import torch
import torch.backends

torch.manual_seed(0)


def assert_matmul_blocked_correctness(M,
                                      N,
                                      K,
                                      block_M=64,
                                      block_N=64,
                                      block_K=32,
                                      trans_A=False,
                                      trans_B=True,
                                      in_dtype="float16",
                                      out_dtype="float16",
                                      accum_dtype="float32",
                                      num_stages=2,
                                      threads=128,
                                      enable_rasterization=False):
    matmul = matmul_blocked(
        M,
        N,
        K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        num_stages=num_stages,
        threads=threads,
        enable_rasterization=enable_rasterization,
    )

    mod, params = tilelang.lower(matmul)
    src_code = mod.imported_modules[0].get_source()

    # src_code is the generated cuda source
    assert src_code is not None

    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype))
    B = torch.rand(N, K, device="cuda", dtype=getattr(torch, in_dtype))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, out_dtype))

    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)

    mod(A, B, C)

    latency = mod.do_bench(mod.func, warmup=25)

    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A, B.T).to(getattr(torch, out_dtype))
    torch.testing.assert_close(C, ref_c, rtol=1e-1, atol=1e0)


def assert_matmul_macro_tensorcore_correctness(
    M,
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    trans_A=False,
    trans_B=True,
    accum_dtype="float16",
    block_row_warps=1,
    block_col_warps=1,
    warp_row_tiles=16,
    warp_col_tiles=16,
    chunk=32,
    num_stages=2,
    enable_rasterization: bool = False,
):
    matmul = matmul_macro_tensorcore(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        trans_A=trans_A,
        trans_B=trans_B,
        accum_dtype=accum_dtype,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
        num_stages=num_stages,
        enable_rasterization=enable_rasterization,
    )
    mod, params = tilelang.lower(matmul)
    src_code = mod.imported_modules[0].get_source()

    # src_code represents generated cuda source
    assert src_code is not None

    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype)) - 0.5
    B = torch.rand(N, K, device="cuda", dtype=getattr(torch, in_dtype)) - 0.5
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))

    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)

    mod(A, B, C)

    latency = mod.do_bench(mod.func, warmup=25)

    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A, B.T).to(getattr(torch, accum_dtype))
    torch.testing.assert_close(C, ref_c, rtol=1e-1, atol=1e0)


def assert_tl_matmul_with_ladder_weight_only_transform_correctness(
    M,
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    trans_A=False,
    trans_B=True,
    accum_dtype="float16",
    block_row_warps=1,
    block_col_warps=1,
    warp_row_tiles=16,
    warp_col_tiles=16,
    chunk=32,
    num_stages=2,
    enable_rasterization: bool = False,
):
    matmul = matmul_macro_tensorcore_weight_propagation_level_ldmatrix(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        trans_A=trans_A,
        trans_B=trans_B,
        accum_dtype=accum_dtype,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
        num_stages=num_stages,
        enable_rasterization=enable_rasterization,
    )

    mod, params = tilelang.lower(matmul)
    src_code = mod.imported_modules[0].get_source()

    # src_code is the generated cuda source
    assert src_code is not None

    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype)) - 0.5
    B = torch.rand(N, K, device="cuda", dtype=getattr(torch, in_dtype)) - 0.5
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))

    ladder_permutate_config = bitblas.ops.LadderPermutateConfig(
        M=N,
        N=K,
        transform_kind=3,
        transpose_matrix=True,
    )

    ladder_permutate = bitblas.ops.LadderPermutate(ladder_permutate_config)

    LB = ladder_permutate(B.cpu()).cuda()

    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)

    mod(A, LB, C)

    latency = mod.do_bench(mod.func, warmup=25)
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A, B.T).to(getattr(torch, accum_dtype))
    torch.testing.assert_close(C, ref_c, rtol=1e0, atol=1e0)


def test_matmul_blocked():
    # Pipeline
    assert_matmul_blocked_correctness(1024, 1024, 1024, num_stages=2)
    assert_matmul_blocked_correctness(1024, 1024, 1024, num_stages=1)
    # L2 Cache
    assert_matmul_blocked_correctness(1024, 1024, 1024, enable_rasterization=True)


def test_matmul_macro_tensorcore():
    # Pipeline
    assert_matmul_macro_tensorcore_correctness(1024, 1024, 1024, num_stages=2)
    assert_matmul_macro_tensorcore_correctness(1024, 1024, 1024, num_stages=1)
    assert_matmul_macro_tensorcore_correctness(1024, 1024, 1024, num_stages=0)
    # L2 Cache
    assert_matmul_macro_tensorcore_correctness(1024, 1024, 1024, enable_rasterization=True)


def test_tl_matmul_with_ladder_weight_only_transform():
    # Pipeline
    assert_tl_matmul_with_ladder_weight_only_transform_correctness(1024, 1024, 1024, num_stages=2)
    assert_tl_matmul_with_ladder_weight_only_transform_correctness(1024, 1024, 1024, num_stages=1)
    assert_tl_matmul_with_ladder_weight_only_transform_correctness(1024, 1024, 1024, num_stages=0)
    # L2 Cache
    assert_tl_matmul_with_ladder_weight_only_transform_correctness(
        1024, 1024, 1024, enable_rasterization=True)


if __name__ == "__main__":
    bitblas.testing.main()
