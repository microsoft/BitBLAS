# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas import tvm as tvm
import bitblas.testing
from tvm import tl
from bitblas.ops.general_matmul.tilelang.dense.matmul_tensorcore import (
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
                                                   in_dtype="float16",
                                                   out_dtype="float16",
                                                   accum_dtype="float16"):
    matmul = MatmulScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
    ).with_default_config()

    mod, params = tl.lower(matmul)
    src_code = mod.imported_modules[0].get_source()

    # src_code is the generated cuda source
    assert src_code is not None

    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype))
    B = torch.rand(N, K, device="cuda", dtype=getattr(torch, in_dtype))
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
                                                   in_dtype="float16",
                                                   out_dtype="float16",
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
        in_dtype=in_dtype,
        out_dtype=out_dtype,
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

    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype))
    B = torch.rand(N, K, device="cuda", dtype=getattr(torch, in_dtype))
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
                                                        in_dtype="float16",
                                                        out_dtype="float16",
                                                        accum_dtype="float16"):

    matmul = MatmulFineGrainScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
    ).with_default_config()

    mod, params = tl.lower(matmul)
    src_code = mod.imported_modules[0].get_source()
    # src_code is the generated cuda source
    assert src_code is not None

    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype))
    B = torch.rand(N, K, device="cuda", dtype=getattr(torch, in_dtype))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, out_dtype))
    mod = tl.Profiler(mod, params, [], tl.TensorSupplyType.Integer)

    latency = mod.do_bench(mod.func, warmup=25)

    # Ensure that the latency is not None
    assert latency is not None

    mod(A, B, C)

    # Get Reference Result
    ref_c = torch.matmul(A, B.T).to(getattr(torch, out_dtype))
    from bitblas.ops import Matmul, MatmulConfig
    matmul_config = MatmulConfig(
        M=M,
        N=N,
        K=K,
        propagate_a=False,
        propagate_b=False,
    )
    matmul = Matmul(matmul_config, enable_tuning=False)
    prim_func = matmul.prim_func
    intrin_info = bitblas.base.hint.IntrinInfo(
        in_dtype=in_dtype,
        out_dtype=accum_dtype,
        trans_b=True,
        input_transform_kind=0,
        weight_transform_kind=0,
    )

    arch = bitblas.base.CUDA(target="cuda")

    sch = bitblas.gpu.MatmulTensorizationMMA().apply_config(
        prim_func,
        config=bitblas.base.Hint.from_dict({
            "arch": arch,
            "block": [64, 64],
            "warp": [32, 32],
            "rstep": [32],
            "pipeline_stage": 2,
            "use_async": True,
            "intrin_info": intrin_info,
            "shared_scope": "shared.dyn",
            "vectorize": {
                "b": 8,
                "a": 8
            },
        }),
    )

    with tvm.transform.PassContext(config={
            "tir.use_async_copy": True,
            "tir.merge_static_smem": False
    }):
        rt_mod = tvm.build(sch.mod, target="cuda")
    from tvm.contrib.dlpack import to_pytorch_func

    torch_func = to_pytorch_func(rt_mod)

    matmul_c = torch.zeros(M, N, device="cuda", dtype=getattr(torch, out_dtype))
    torch_func(A, B, matmul_c)

    torch.testing.assert_close(matmul_c, ref_c, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


def assert_matmul_fine_grained_apply_config_correctness(
    M,
    N,
    K,
    trans_A=False,
    trans_B=True,
    in_dtype="float16",
    out_dtype="float16",
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
        in_dtype=in_dtype,
        out_dtype=out_dtype,
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

    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype))
    B = torch.rand(N, K, device="cuda", dtype=getattr(torch, in_dtype))
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
                                                              in_dtype="float16",
                                                              out_dtype="float16",
                                                              accum_dtype="float16"):

    matmul = MatmulWeightPropagationScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
    ).with_default_config()

    mod, params = tl.lower(matmul)
    src_code = mod.imported_modules[0].get_source()

    # src_code is the generated cuda source
    assert src_code is not None

    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype)) - 0.5
    B = torch.rand(N, K, device="cuda", dtype=getattr(torch, in_dtype)) - 0.5
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, out_dtype))

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
    ref_c = torch.matmul(A, B.T).to(getattr(torch, out_dtype))
    print(C)
    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e0, atol=1e0)


def assert_matmul_weight_propagation_apply_config_correctness(
    M,
    N,
    K,
    trans_A=False,
    trans_B=True,
    in_dtype="float16",
    out_dtype="float16",
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
        in_dtype=in_dtype,
        out_dtype=out_dtype,
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

    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype)) - 0.5
    B = torch.rand(N, K, device="cuda", dtype=getattr(torch, in_dtype)) - 0.5
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, out_dtype))

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
    ref_c = torch.matmul(A, B.T).to(getattr(torch, out_dtype))
    torch.testing.assert_close(C, ref_c, rtol=1e0, atol=1e0)


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
