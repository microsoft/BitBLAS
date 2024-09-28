# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas import tvm as tvm
import bitblas.testing
from tvm import tl
from bitblas.ops.general_matmul.tilelang.dense.matmul import (
    MatmulScheduler,
    MatmulFineGrainScheduler,
    MatmulWeightPropagationScheduler,
)

import torch
import torch.backends

torch.manual_seed(0)


def assert_matmul_blocked_with_default_correctness(M,
                                                   N,
                                                   K,
                                                   trans_A=False,
                                                   trans_B=True,
                                                   dtypeAB="float16",
                                                   dtypeC="float16",
                                                   accum_dtype="float16"):
    matmul = MatmulScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        dtypeAB=dtypeAB,
        dtypeC=dtypeC,
        accum_dtype=accum_dtype,
    ).with_default_config()

    mod, params = tl.lower(matmul)
    src_code = mod.imported_modules[0].get_source()

    # src_code is the generated cuda source
    assert src_code is not None

    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, dtypeAB))
    B = torch.rand(N, K, device="cuda", dtype=getattr(torch, dtypeAB))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))

    mod = tl.Profiler(mod, params, [], tl.TensorSupplyType.Integer)

    mod(A, B, C)

    latency = mod.do_bench(mod.func, warmup=25)

    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A, B.T).to(getattr(torch, accum_dtype))
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


def assert_matmul_blocked_apply_config_correctness(M,
                                                   N,
                                                   K,
                                                   block_M=64,
                                                   block_N=64,
                                                   block_K=32,
                                                   trans_A=False,
                                                   trans_B=True,
                                                   dtypeAB="float16",
                                                   dtypeC="float16",
                                                   accum_dtype="float16",
                                                   num_stages=2,
                                                   threads=128,
                                                   enable_rasterization=False):
    matmul = MatmulScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        dtypeAB=dtypeAB,
        dtypeC=dtypeC,
        accum_dtype=accum_dtype,
    ).apply_config(
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        num_stages=num_stages,
        threads=threads,
        enable_rasterization=enable_rasterization,
    )

    mod, params = tl.lower(matmul)
    src_code = mod.imported_modules[0].get_source()

    # src_code is the generated cuda source
    assert src_code is not None

    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, dtypeAB))
    B = torch.rand(N, K, device="cuda", dtype=getattr(torch, dtypeAB))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))

    mod = tl.Profiler(mod, params, [], tl.TensorSupplyType.Integer)

    mod(A, B, C)

    latency = mod.do_bench(mod.func, warmup=25)

    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A, B.T).to(getattr(torch, accum_dtype))
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


def assert_matmul_fine_grained_with_default_correctness(M,
                                                        N,
                                                        K,
                                                        trans_A=False,
                                                        trans_B=True,
                                                        dtypeAB="float16",
                                                        dtypeC="float16",
                                                        accum_dtype="float16"):

    matmul = MatmulFineGrainScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        dtypeAB=dtypeAB,
        dtypeC=dtypeC,
        accum_dtype=accum_dtype,
    ).with_default_config()

    mod, params = tl.lower(matmul)
    src_code = mod.imported_modules[0].get_source()

    # src_code is the generated cuda source
    assert src_code is not None

    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, dtypeAB)) - 0.5
    B = torch.rand(N, K, device="cuda", dtype=getattr(torch, dtypeAB)) - 0.5
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, dtypeC))

    mod = tl.Profiler(mod, params, [], tl.TensorSupplyType.Integer)

    mod(A, B, C)

    latency = mod.do_bench(mod.func, warmup=25)

    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A, B.T).to(getattr(torch, dtypeC))
    print(C)
    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-1, atol=1e-1)


def assert_matmul_fine_grained_apply_config_correctness(
    M,
    N,
    K,
    trans_A=False,
    trans_B=True,
    dtypeAB="float16",
    dtypeC="float16",
    accum_dtype="float16",
    block_row_warps=1,
    block_col_warps=1,
    warp_row_tiles=16,
    warp_col_tiles=16,
    chunk=32,
    num_stages=2,
    enable_rasterization=False,
):

    matmul = MatmulFineGrainScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        dtypeAB=dtypeAB,
        dtypeC=dtypeC,
        accum_dtype=accum_dtype,
    ).apply_config(
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
        num_stages=num_stages,
        enable_rasterization=enable_rasterization,
    )

    mod, params = tl.lower(matmul)
    src_code = mod.imported_modules[0].get_source()

    # src_code is the generated cuda source
    assert src_code is not None

    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, dtypeAB))
    B = torch.rand(N, K, device="cuda", dtype=getattr(torch, dtypeAB))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))

    mod = tl.Profiler(mod, params, [], tl.TensorSupplyType.Integer)

    mod(A, B, C)

    latency = mod.do_bench(mod.func, warmup=25)

    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A, B.T).to(getattr(torch, accum_dtype))
    torch.testing.assert_close(C, ref_c, rtol=1e-1, atol=1e-1)


def assert_matmul_weight_propagation_with_default_correctness(M,
                                                              N,
                                                              K,
                                                              trans_A=False,
                                                              trans_B=True,
                                                              dtypeAB="float16",
                                                              dtypeC="float16",
                                                              accum_dtype="float16"):

    matmul = MatmulWeightPropagationScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        dtypeAB=dtypeAB,
        dtypeC=dtypeC,
        accum_dtype=accum_dtype,
    ).with_default_config()

    mod, params = tl.lower(matmul)
    src_code = mod.imported_modules[0].get_source()

    # src_code is the generated cuda source
    assert src_code is not None

    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, dtypeAB)) - 0.5
    B = torch.rand(N, K, device="cuda", dtype=getattr(torch, dtypeAB)) - 0.5
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, dtypeC))

    ladder_permutate_config = bitblas.ops.LadderPermutateConfig(
        M=N,
        N=K,
        transform_kind=3,
        transpose_matrix=True,
    )

    ladder_permutate = bitblas.ops.LadderPermutate(ladder_permutate_config)

    LB = ladder_permutate(B.cpu()).cuda()

    mod = tl.Profiler(mod, params, [], tl.TensorSupplyType.Integer)

    mod(A, LB, C)

    latency = mod.do_bench(mod.func, warmup=25)

    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A, B.T).to(getattr(torch, dtypeC))
    print(C)
    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-1, atol=1e-1)


def assert_matmul_weight_propagation_apply_config_correctness(
    M,
    N,
    K,
    trans_A=False,
    trans_B=True,
    dtypeAB="float16",
    dtypeC="float16",
    accum_dtype="float16",
    block_row_warps=1,
    block_col_warps=1,
    warp_row_tiles=16,
    warp_col_tiles=16,
    chunk=32,
    num_stages=2,
    enable_rasterization=False,
):

    matmul = MatmulWeightPropagationScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        dtypeAB=dtypeAB,
        dtypeC=dtypeC,
        accum_dtype=accum_dtype,
    ).apply_config(
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
        num_stages=num_stages,
        enable_rasterization=enable_rasterization,
    )

    mod, params = tl.lower(matmul)
    src_code = mod.imported_modules[0].get_source()

    # src_code is the generated cuda source
    assert src_code is not None

    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, dtypeAB)) - 0.5
    B = torch.rand(N, K, device="cuda", dtype=getattr(torch, dtypeAB)) - 0.5
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, dtypeC))

    ladder_permutate_config = bitblas.ops.LadderPermutateConfig(
        M=N,
        N=K,
        transform_kind=3,
        transpose_matrix=True,
    )

    ladder_permutate = bitblas.ops.LadderPermutate(ladder_permutate_config)

    LB = ladder_permutate(B.cpu()).cuda()

    mod = tl.Profiler(mod, params, [], tl.TensorSupplyType.Integer)

    mod(A, LB, C)

    latency = mod.do_bench(mod.func, warmup=25)

    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A, B.T).to(getattr(torch, dtypeC))
    torch.testing.assert_close(C, ref_c, rtol=1e-1, atol=1e-1)


def test_matmul_blocked():
    # Default
    assert_matmul_blocked_with_default_correctness(1024, 1024, 1024)
    # Pipeline
    assert_matmul_blocked_apply_config_correctness(1024, 1024, 1024, num_stages=2)
    assert_matmul_blocked_apply_config_correctness(1024, 1024, 1024, num_stages=1)
    # L2 Cache
    assert_matmul_blocked_apply_config_correctness(1024, 1024, 1024, enable_rasterization=True)


def test_matmul_fine_grained():
    # Default
    assert_matmul_fine_grained_with_default_correctness(1024, 1024, 1024)
    # Pipeline
    assert_matmul_fine_grained_apply_config_correctness(1024, 1024, 1024, num_stages=2)
    assert_matmul_fine_grained_apply_config_correctness(1024, 1024, 1024, num_stages=1)
    # L2 Cache
    assert_matmul_fine_grained_apply_config_correctness(1024, 1024, 1024, enable_rasterization=True)


def test_matmul_weight_propagation():
    # Default
    assert_matmul_weight_propagation_with_default_correctness(1024, 1024, 1024)
    # Pipeline
    assert_matmul_weight_propagation_apply_config_correctness(1024, 1024, 1024, num_stages=2)
    assert_matmul_weight_propagation_apply_config_correctness(1024, 1024, 1024, num_stages=1)
    # L2 Cache
    assert_matmul_weight_propagation_apply_config_correctness(
        1024, 1024, 1024, enable_rasterization=True)


if __name__ == "__main__":
    bitblas.testing.main()
