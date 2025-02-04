# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas import tvm as tvm
import bitblas.testing
from bitblas import tilelang as tilelang
from bitblas.ops.general_matmul.tilelang.dense.matmul_tile import (
    MatmulTileLibraryScheduler,)

from bitblas.ops.general_matmul.tilelang.dequantize import (
    MatmulDequantizeScheduler,
    MatmulDequantizeMMAScheduler,
    MatmulDequantizeMMAWeightPropagationScheduler,
    MatmulINT4DequantizeMMAScheduler,
    MatmulINT4DequantizeMMAWeightPropagationScheduler,
)

from bitblas.ops.general_matmul.tilelang.dense.matmul_mma import (
    MatmulMMAScheduler,
    MatmulMMAWeightPropagationScheduler,
    MatmulINT4MMAScheduler,
    MatmulINT4MMAWeightPropagationScheduler,
)

import torch
import torch.backends

torch.manual_seed(0)

verbose = False


def assert_matmul_blocked_with_default_correctness(
    M,
    N,
    K,
    trans_A=False,
    trans_B=True,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
):
    matmul = MatmulTileLibraryScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
    ).with_default_config()

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
    torch.testing.assert_close(C, ref_c, rtol=1e0, atol=1e-1)


def assert_matmul_blocked_apply_config_correctness(
    M,
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
    enable_rasterization: bool = False,
):
    matmul = MatmulTileLibraryScheduler(
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
    torch.testing.assert_close(C, ref_c, rtol=1e0, atol=1e-1)


def assert_matmul_fine_grained_with_default_correctness(
    M,
    N,
    K,
    trans_A=False,
    trans_B=True,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
):

    matmul = MatmulMMAScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
    ).with_default_config()

    mod, params = tilelang.lower(matmul)
    src_code = mod.imported_modules[0].get_source()
    # src_code is the generated cuda source
    assert src_code is not None
    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype)) - 0.5
    B = (torch.rand(N, K, device="cuda", dtype=getattr(torch, in_dtype)) if trans_B else torch.rand(
        K, N, device="cuda", dtype=getattr(torch, in_dtype))) - 0.5
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, out_dtype))
    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)

    latency = mod.do_bench(mod.func, warmup=25)

    # Ensure that the latency is not None
    assert latency is not None

    mod(A, B, C)

    # Get Reference Result
    ref_c = (
        torch.matmul(A, B.T).to(getattr(torch, out_dtype)) if trans_B else torch.matmul(A, B).to(
            getattr(torch, out_dtype)))

    torch.testing.assert_close(C, ref_c, rtol=1e0, atol=1e-1)


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
    enable_rasterization: bool = False,
):

    matmul = MatmulMMAScheduler(
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

    mod, params = tilelang.lower(matmul)
    src_code = mod.imported_modules[0].get_source()

    # src_code is the generated cuda source
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
    torch.testing.assert_close(C, ref_c, rtol=1e-1, atol=1e-1)


def assert_matmul_weight_propagation_with_default_correctness(
    M,
    N,
    K,
    trans_A=False,
    trans_B=True,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
):

    matmul = MatmulMMAWeightPropagationScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
    ).with_default_config()

    mod, params = tilelang.lower(matmul)
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

    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)

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
    enable_rasterization: bool = False,
):

    matmul = MatmulMMAWeightPropagationScheduler(
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

    mod, params = tilelang.lower(matmul)
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

    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)

    mod(A, LB, C)

    latency = mod.do_bench(mod.func, warmup=25)

    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A, B.T).to(getattr(torch, out_dtype))
    torch.testing.assert_close(C, ref_c, rtol=1e0, atol=1e0)


def assert_matmul_int4_fine_grained_with_default_correctness(
    M,
    N,
    K,
    trans_A=False,
    trans_B=True,
    in_dtype="int4",
    out_dtype="int32",
    accum_dtype="int32",
):

    matmul = MatmulINT4MMAScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
    ).with_default_config()

    mod, params = tilelang.lower(matmul)
    src_code = mod.imported_modules[0].get_source()
    # src_code is the generated cuda source
    assert src_code is not None
    storage_dtype = "int8"
    A = torch.randint(0, 4, (M, K), device="cuda", dtype=getattr(torch, storage_dtype))
    B = torch.randint(0, 4, (N, K), device="cuda", dtype=getattr(torch, storage_dtype))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))

    compressed_A = (A[:, ::2] & 0x0F) + ((A[:, 1::2] & 0x0F) << 4)
    compressed_B = (B[:, ::2] & 0x0F) + ((B[:, 1::2] & 0x0F) << 4)
    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)

    latency = mod.do_bench(mod.func, warmup=25, profiler="tvm")
    print(latency)

    # Ensure that the latency is not None
    assert latency is not None

    mod(compressed_A, compressed_B, C)
    print(C)

    # Get Reference Result
    ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, out_dtype))
    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-1)


def assert_matmul_int4_fine_grained_apply_config_correctness(
    M,
    N,
    K,
    trans_A=False,
    trans_B=True,
    in_dtype="int4",
    out_dtype="int32",
    accum_dtype="int32",
    block_row_warps=1,
    block_col_warps=1,
    warp_row_tiles=16,
    warp_col_tiles=16,
    chunk=32,
    num_stages=2,
    enable_rasterization: bool = False,
):

    matmul = MatmulINT4MMAScheduler(
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

    mod, params = tilelang.lower(matmul)
    src_code = mod.imported_modules[0].get_source()
    # src_code is the generated cuda source
    assert src_code is not None
    storage_dtype = "int8"
    A = torch.randint(0, 4, (M, K), device="cuda", dtype=getattr(torch, storage_dtype))
    B = torch.randint(0, 4, (N, K), device="cuda", dtype=getattr(torch, storage_dtype))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))

    compressed_A = (A[:, ::2] & 0x0F) + ((A[:, 1::2] & 0x0F) << 4)
    compressed_B = (B[:, ::2] & 0x0F) + ((B[:, 1::2] & 0x0F) << 4)
    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)

    latency = mod.do_bench(mod.func, warmup=25, profiler="tvm")
    print(latency)

    # Ensure that the latency is not None
    assert latency is not None

    mod(compressed_A, compressed_B, C)
    print(C)

    # Get Reference Result
    ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, out_dtype))
    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-1)


def assert_matmul_int4_weight_propagation_with_default_correctness(
    M,
    N,
    K,
    trans_A=False,
    trans_B=True,
    in_dtype="int4",
    out_dtype="int32",
    accum_dtype="int32",
):

    matmul = MatmulINT4MMAWeightPropagationScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
    ).with_default_config()
    print(matmul)
    mod, params = tilelang.lower(matmul)
    src_code = mod.imported_modules[0].get_source()

    # src_code is the generated cuda source
    assert src_code is not None
    storage_dtype = "int8"
    A = torch.randint(0, 4, (M, K), device="cuda", dtype=getattr(torch, storage_dtype))
    B = torch.randint(0, 4, (N, K), device="cuda", dtype=getattr(torch, storage_dtype))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))
    compressed_A = (A[:, ::2] & 0x0F) + ((A[:, 1::2] & 0x0F) << 4)
    compressed_B = (B[:, ::2] & 0x0F) + ((B[:, 1::2] & 0x0F) << 4)

    ladder_permutate_config = bitblas.ops.LadderPermutateConfig(
        M=N,
        N=(K // 2),
        datatype="int8",
        storage_dtype="int8",
        transform_kind=3,
        transpose_matrix=True,
    )

    ladder_permutate = bitblas.ops.LadderPermutate(ladder_permutate_config)

    LB = ladder_permutate(compressed_B.cpu()).cuda()

    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)

    mod(compressed_A, LB, C)

    latency = mod.do_bench(mod.func, warmup=25)

    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, out_dtype))
    print(C)
    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


def assert_matmul_int4_weight_propagation_apply_config__correctness(
    M,
    N,
    K,
    trans_A=False,
    trans_B=True,
    in_dtype="int4",
    out_dtype="int32",
    accum_dtype="int32",
    block_row_warps=1,
    block_col_warps=1,
    warp_row_tiles=16,
    warp_col_tiles=16,
    chunk=32,
    num_stages=2,
    enable_rasterization: bool = False,
):

    matmul = MatmulINT4MMAWeightPropagationScheduler(
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

    print(matmul)
    mod, params = tilelang.lower(matmul)
    src_code = mod.imported_modules[0].get_source()

    # src_code is the generated cuda source
    assert src_code is not None
    storage_dtype = "int8"
    A = torch.randint(0, 4, (M, K), device="cuda", dtype=getattr(torch, storage_dtype))
    B = torch.randint(0, 4, (N, K), device="cuda", dtype=getattr(torch, storage_dtype))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))
    compressed_A = (A[:, ::2] & 0x0F) + ((A[:, 1::2] & 0x0F) << 4)
    compressed_B = (B[:, ::2] & 0x0F) + ((B[:, 1::2] & 0x0F) << 4)

    ladder_permutate_config = bitblas.ops.LadderPermutateConfig(
        M=N,
        N=(K // 2),
        datatype="int8",
        storage_dtype="int8",
        transform_kind=3,
        transpose_matrix=True,
    )

    ladder_permutate = bitblas.ops.LadderPermutate(ladder_permutate_config)

    LB = ladder_permutate(compressed_B.cpu()).cuda()

    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)

    mod(compressed_A, LB, C)

    latency = mod.do_bench(mod.func, warmup=25)

    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, out_dtype))
    print(C)
    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


def assert_matmul_fine_grained_dequant_int4_with_default_correctness(
    M,
    N,
    K,
    trans_A=False,
    trans_B=True,
    in_dtype="int4",
    out_dtype="int32",
    accum_dtype="int32",
    bit=2,
    storage_dtype="int8",
    source_format="int",
    with_scaling=False,
    with_zeros=False,
    group_size=-1,
    fast_decoding=False,
    zeros_mode="original",
):
    matmul = MatmulINT4DequantizeMMAScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        num_bits=bit,
        storage_dtype=storage_dtype,
        source_format=source_format,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        group_size=group_size,
        fast_decoding=fast_decoding,
        zeros_mode=zeros_mode,
    ).with_default_config()

    mod, params = tilelang.lower(matmul)
    src_code = mod.imported_modules[0].get_source()
    # src_code is the generated cuda source
    assert src_code is not None

    A = torch.randint(0, 4, (M, K), device="cuda", dtype=getattr(torch, storage_dtype))
    B = torch.randint(0, 2, (N, K), device="cuda", dtype=getattr(torch, storage_dtype))

    lop3_permutate_config = bitblas.ops.LOP3PermutateConfig(
        M=N,
        N=K,
        datatype=in_dtype,
        dequantize_bits=bit,
        storage_dtype=storage_dtype,
    )
    lop3_permutate = bitblas.ops.LOP3Permutate(
        config=lop3_permutate_config,
        target=tvm.target.Target("llvm"),
    )

    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, out_dtype))

    compressed_A = (A[:, ::2] & 0x0F) + ((A[:, 1::2] & 0x0F) << 4)
    compressed_B = (B[:, ::4] & 0x03) + ((B[:, 1::4] & 0x03) << 2) + ((B[:, 2::4] & 0x03) << 4) + (
        (B[:, 3::4] & 0x03) << 6)

    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)
    print(f"{compressed_B=}")
    if fast_decoding:
        lop3_compressed_B = lop3_permutate(compressed_B.cpu()).cuda()
    else:
        lop3_compressed_B = compressed_B
    print(f"{lop3_compressed_B=}")
    mod(compressed_A, lop3_compressed_B, C)
    print(C)
    latency = mod.do_bench(mod.func, warmup=25, profiler="tvm")
    print(latency)
    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, out_dtype))

    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


def assert_matmul_fine_grained_dequant_int4_apply_config_correctness(
    M,
    N,
    K,
    trans_A=False,
    trans_B=True,
    in_dtype="int4",
    out_dtype="int32",
    accum_dtype="int32",
    bit=2,
    storage_dtype="int8",
    source_format="int",
    with_scaling=False,
    with_zeros=False,
    group_size=-1,
    fast_decoding=False,
    zeros_mode="original",
    block_row_warps=1,
    block_col_warps=1,
    warp_row_tiles=16,
    warp_col_tiles=16,
    chunk=32,
    num_stages=2,
    enable_rasterization: bool = False,
):
    matmul = MatmulINT4DequantizeMMAScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        num_bits=bit,
        storage_dtype=storage_dtype,
        source_format=source_format,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        group_size=group_size,
        fast_decoding=fast_decoding,
        zeros_mode=zeros_mode,
    ).apply_config(
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

    A = torch.randint(0, 4, (M, K), device="cuda", dtype=getattr(torch, storage_dtype))
    B = torch.randint(0, 2, (N, K), device="cuda", dtype=getattr(torch, storage_dtype))

    lop3_permutate_config = bitblas.ops.LOP3PermutateConfig(
        M=N,
        N=K,
        datatype=in_dtype,
        dequantize_bits=bit,
        storage_dtype=storage_dtype,
    )
    lop3_permutate = bitblas.ops.LOP3Permutate(
        config=lop3_permutate_config,
        target=tvm.target.Target("llvm"),
    )

    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, out_dtype))

    compressed_A = (A[:, ::2] & 0x0F) + ((A[:, 1::2] & 0x0F) << 4)
    compressed_B = (B[:, ::4] & 0x03) + ((B[:, 1::4] & 0x03) << 2) + ((B[:, 2::4] & 0x03) << 4) + (
        (B[:, 3::4] & 0x03) << 6)

    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)
    print(f"{compressed_B=}")
    if fast_decoding:
        lop3_compressed_B = lop3_permutate(compressed_B.cpu()).cuda()
    else:
        lop3_compressed_B = compressed_B
    print(f"{lop3_compressed_B=}")
    mod(compressed_A, lop3_compressed_B, C)
    print(C)
    latency = mod.do_bench(mod.func, warmup=25, profiler="tvm")
    print(latency)
    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, out_dtype))

    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


def assert_matmul_weight_transform_dequant_int4_with_default_correctness(
    M,
    N,
    K,
    trans_A=False,
    trans_B=True,
    in_dtype="int4",
    out_dtype="int32",
    accum_dtype="int32",
    bit=2,
    storage_dtype="int8",
    source_format="int",
    with_scaling=False,
    with_zeros=False,
    group_size=-1,
    fast_decoding=False,
    zeros_mode="original",
):
    matmul = MatmulINT4DequantizeMMAWeightPropagationScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        num_bits=bit,
        storage_dtype=storage_dtype,
        source_format=source_format,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        group_size=group_size,
        fast_decoding=fast_decoding,
        zeros_mode=zeros_mode,
    ).with_default_config()
    print(matmul)
    mod, params = tilelang.lower(matmul)
    src_code = mod.imported_modules[0].get_source()
    # src_code is the generated cuda source
    assert src_code is not None
    transform_b = 3  # assume ladder stage 3 transform
    A = torch.randint(0, 4, (M, K), device="cuda", dtype=getattr(torch, storage_dtype))
    B = torch.randint(0, 2, (N, K), device="cuda", dtype=getattr(torch, storage_dtype))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))

    compressed_A = (A[:, ::2] & 0x0F) + ((A[:, 1::2] & 0x0F) << 4)
    compressed_B = (B[:, ::2] & 0x0F) + ((B[:, 1::2] & 0x0F) << 4)

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

    ladder_permutate_config = bitblas.ops.LadderPermutateConfig(
        M=N,
        N=(K // 2),
        datatype="int8",
        storage_dtype="int8",
        transform_kind=transform_b,
        transpose_matrix=True,
    )

    ladder_permutate = bitblas.ops.LadderPermutate(ladder_permutate_config)

    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)
    compressed_B_ladder = ladder_permutate(compressed_B.cpu()).cuda()
    ladder_shape = compressed_B_ladder.shape
    int2_shape = (ladder_shape[:-1] + (ladder_shape[-1] // 2,))
    int2_tensor = torch.zeros(int2_shape, device="cuda", dtype=torch.int8)
    for i in range(int2_tensor.shape[-1]):
        int2_tensor[..., i] = (compressed_B_ladder[..., 2 * i] & 0x03) | (
            (compressed_B_ladder[..., 2 * i] >> 4) & 0x03) << 2 | (
                (compressed_B_ladder[..., 2 * i + 1] & 0x03) << 4) | (
                    (compressed_B_ladder[..., 2 * i + 1] >> 4) << 6)

    raw_tensor_shape = int2_tensor.shape
    print(f"{raw_tensor_shape=}")
    if fast_decoding:
        lop3_compressed_B = lop3_permutate(int2_tensor.cpu()).cuda()
        lop3_compressed_B = lop3_compressed_B.view(raw_tensor_shape)
    else:
        lop3_compressed_B = int2_tensor

    mod(compressed_A, lop3_compressed_B, C)

    latency = mod.do_bench(mod.func, warmup=25)
    print(f"Latency: {latency}")
    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, accum_dtype))
    print(C)
    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


def assert_matmul_weight_transform_dequant_int4_apply_config_correctness(
    M,
    N,
    K,
    trans_A=False,
    trans_B=True,
    in_dtype="int4",
    out_dtype="int32",
    accum_dtype="int32",
    bit=2,
    storage_dtype="int8",
    source_format="int",
    with_scaling=False,
    with_zeros=False,
    group_size=-1,
    fast_decoding=False,
    zeros_mode="original",
    block_row_warps=1,
    block_col_warps=1,
    warp_row_tiles=16,
    warp_col_tiles=16,
    chunk=32,
    num_stages=2,
    enable_rasterization: bool = False,
):
    matmul = MatmulINT4DequantizeMMAWeightPropagationScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        num_bits=bit,
        storage_dtype=storage_dtype,
        source_format=source_format,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        group_size=group_size,
        fast_decoding=fast_decoding,
        zeros_mode=zeros_mode,
    ).apply_config(
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
        num_stages=num_stages,
        enable_rasterization=enable_rasterization,
    )

    print(matmul)
    mod, params = tilelang.lower(matmul)
    src_code = mod.imported_modules[0].get_source()
    # src_code is the generated cuda source
    assert src_code is not None
    transform_b = 3  # assume ladder stage 3 transform
    A = torch.randint(0, 4, (M, K), device="cuda", dtype=getattr(torch, storage_dtype))
    B = torch.randint(0, 2, (N, K), device="cuda", dtype=getattr(torch, storage_dtype))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))

    compressed_A = (A[:, ::2] & 0x0F) + ((A[:, 1::2] & 0x0F) << 4)
    compressed_B = (B[:, ::2] & 0x0F) + ((B[:, 1::2] & 0x0F) << 4)

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

    ladder_permutate_config = bitblas.ops.LadderPermutateConfig(
        M=N,
        N=(K // 2),
        datatype="int8",
        storage_dtype="int8",
        transform_kind=transform_b,
        transpose_matrix=True,
    )

    ladder_permutate = bitblas.ops.LadderPermutate(ladder_permutate_config)

    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)
    compressed_B_ladder = ladder_permutate(compressed_B.cpu()).cuda()
    ladder_shape = compressed_B_ladder.shape
    int2_shape = (ladder_shape[:-1] + (ladder_shape[-1] // 2,))
    int2_tensor = torch.zeros(int2_shape, device="cuda", dtype=torch.int8)
    for i in range(int2_tensor.shape[-1]):
        int2_tensor[..., i] = (compressed_B_ladder[..., 2 * i] & 0x03) | (
            (compressed_B_ladder[..., 2 * i] >> 4) & 0x03) << 2 | (
                (compressed_B_ladder[..., 2 * i + 1] & 0x03) << 4) | (
                    (compressed_B_ladder[..., 2 * i + 1] >> 4) << 6)

    raw_tensor_shape = int2_tensor.shape
    print(f"{raw_tensor_shape=}")
    if fast_decoding:
        lop3_compressed_B = lop3_permutate(int2_tensor.cpu()).cuda()
        lop3_compressed_B = lop3_compressed_B.view(raw_tensor_shape)
    else:
        lop3_compressed_B = int2_tensor

    mod(compressed_A, lop3_compressed_B, C)

    latency = mod.do_bench(mod.func, warmup=25)
    print(f"Latency: {latency}")
    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, accum_dtype))
    print(C)
    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


def assert_matmul_blocked_dequant_with_default_correctness(
    M,
    N,
    K,
    trans_A=False,
    trans_B=True,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    bit=4,
    storage_dtype="int8",
    source_format="uint",
    with_scaling=False,
    with_zeros=False,
    group_size=-1,
    fast_decoding=False,
    zeros_mode="original",
):
    import numpy as np
    from bitblas.quantization import general_compress, interleave_weight

    matmul = MatmulDequantizeScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        num_bits=bit,
        storage_dtype=storage_dtype,
        source_format=source_format,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        group_size=group_size,
        fast_decoding=fast_decoding,
        zeros_mode=zeros_mode,
    ).with_default_config()
    print(matmul)
    mod, params = tilelang.lower(matmul)
    src_code = mod.imported_modules[0].get_source()
    # src_code is the generated cuda source
    assert src_code is not None
    print(src_code)
    input_shape = (M, K)
    weight_shape = (N, K)
    output_shape = (M, N)
    inputs = []
    inputs.append(torch.rand(input_shape, dtype=torch.float16).cuda() - 0.5)
    maxq = 2**(bit - 1)
    zeros = maxq
    if source_format == "uint":
        inputs.append(torch.randint(0, maxq, weight_shape, dtype=torch.int8).cuda())
    elif source_format == "int":
        inputs.append(torch.randint(-maxq, maxq, weight_shape, dtype=torch.int8).cuda())
    else:
        raise NotImplementedError

    inputs.append(torch.rand(output_shape, dtype=torch.float16).cuda())

    intweight = inputs[1]
    intweight = intweight.cpu().to(torch.int8)
    if source_format == "int":
        intweight = intweight + maxq
    if with_zeros:
        inputs[1] = inputs[1] - zeros

    permuted_inputs = []
    permuted_inputs.append(inputs[0])
    qw = general_compress(intweight.cpu().numpy(), source_bits=bit, storage_dtype=np.int8)
    # lop3 transformation
    if fast_decoding:
        qw = interleave_weight(qw, bit, target_dtype=in_dtype)
    permuted_inputs.append(torch.from_numpy(qw).cuda())
    if with_scaling:
        if group_size == -1:
            group_size = K
        permuted_inputs.append(torch.randn((N, K // group_size), dtype=torch.float16).cuda())
    if with_zeros:
        if zeros_mode == "original":
            permuted_inputs.append(
                torch.ones([N, K // group_size], dtype=torch.float16).cuda() * zeros)
        elif zeros_mode == "rescale":
            original_zeros = (torch.ones([N, K // group_size], dtype=torch.float16).cuda() * zeros)
            scaled_zeros = original_zeros * permuted_inputs[-1]
            permuted_inputs.append(scaled_zeros)
        elif zeros_mode == "quantized":
            original_zeros = (torch.ones([K // group_size, N], dtype=torch.int8).cuda() * zeros)
            qzeros = general_compress(
                original_zeros.cpu().numpy(), source_bits=bit, storage_dtype=np.int8)
            permuted_inputs.append(torch.from_numpy(qzeros).cuda())
        else:
            raise NotImplementedError

    permuted_inputs.append(inputs[2])

    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)

    mod(*permuted_inputs)

    print(permuted_inputs[-1])

    args = [inputs[0]]
    b = inputs[1]
    if with_scaling:
        scale = permuted_inputs[2]
        rescale_b = torch.empty_like(b, dtype=torch.float16)
        for i in range(N):
            for j in range(K):
                if with_zeros:
                    zeros = permuted_inputs[3]
                    if zeros_mode == "original":
                        rescale_b[i, j] = (b[i, j] - zeros[i, j // group_size]) * scale[i, j //
                                                                                        group_size]
                    elif zeros_mode == "rescale":
                        rescale_b[i, j] = (
                            b[i, j] * scale[i, j // group_size] + zeros[i, j // group_size])
                    else:
                        raise NotImplementedError
                else:
                    rescale_b[i, j] = b[i, j] * scale[i, j // group_size]
        args.append(rescale_b.t().cuda())
    else:
        args.append(b.t().cuda().to(torch.float16))

    ref_result = torch.matmul(*args)

    print(ref_result)
    if zeros_mode == "rescale":
        torch.testing.assert_close(permuted_inputs[-1], ref_result, rtol=1e2, atol=1e2)
    else:
        torch.testing.assert_close(permuted_inputs[-1], ref_result, rtol=1e2, atol=1e2)


def assert_matmul_fine_grained_dequant_with_default_correctness(
    M,
    N,
    K,
    trans_A=False,
    trans_B=True,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    bit=4,
    storage_dtype="int8",
    source_format="uint",
    with_scaling=False,
    with_zeros=False,
    group_size=-1,
    fast_decoding=False,
    zeros_mode="original",
):
    import numpy as np
    from bitblas.quantization import general_compress, interleave_weight

    matmul = MatmulDequantizeMMAScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        num_bits=bit,
        storage_dtype=storage_dtype,
        source_format=source_format,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        group_size=group_size,
        fast_decoding=fast_decoding,
        zeros_mode=zeros_mode,
    ).with_default_config()

    mod, params = tilelang.lower(matmul)
    src_code = mod.imported_modules[0].get_source()
    # src_code is the generated cuda source
    assert src_code is not None
    input_shape = (M, K)
    weight_shape = (N, K)
    output_shape = (M, N)
    inputs = []
    inputs.append(torch.rand(input_shape, dtype=torch.float16).cuda() - 0.5)
    maxq = 2**(bit - 1)
    zeros = maxq
    if source_format == "uint":
        inputs.append(torch.randint(0, maxq, weight_shape, dtype=torch.int8).cuda())
    elif source_format == "int":
        inputs.append(torch.randint(-maxq, maxq, weight_shape, dtype=torch.int8).cuda())
    else:
        raise NotImplementedError

    inputs.append(torch.rand(output_shape, dtype=torch.float16).cuda())

    intweight = inputs[1]
    intweight = intweight.cpu().to(torch.int8)
    if source_format == "int":
        intweight = intweight + maxq
    if with_zeros:
        inputs[1] = inputs[1] - zeros

    permuted_inputs = []
    permuted_inputs.append(inputs[0])
    qw = general_compress(intweight.cpu().numpy(), source_bits=bit, storage_dtype=np.int8)
    # lop3 transformation
    if fast_decoding:
        qw = interleave_weight(qw, bit, target_dtype=in_dtype)
    permuted_inputs.append(torch.from_numpy(qw).cuda())
    if with_scaling:
        if group_size == -1:
            group_size = K
        permuted_inputs.append(torch.ones([N, K // group_size], dtype=torch.float16).cuda())
    if with_zeros:
        if zeros_mode == "original":
            permuted_inputs.append(torch.randn((N, K // group_size), dtype=torch.float16).cuda())
        elif zeros_mode == "rescale":
            original_zeros = (torch.ones([N, K // group_size], dtype=torch.float16).cuda() * zeros)
            scaled_zeros = original_zeros * permuted_inputs[-1]
            permuted_inputs.append(scaled_zeros)
        elif zeros_mode == "quantized":
            original_zeros = (torch.ones([K // group_size, N], dtype=torch.int8).cuda() * zeros)
            qzeros = general_compress(
                original_zeros.cpu().numpy(), source_bits=bit, storage_dtype=np.int8)
            permuted_inputs.append(torch.from_numpy(qzeros).cuda())
        else:
            raise NotImplementedError

    permuted_inputs.append(inputs[2])

    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)

    mod(*permuted_inputs)

    print(permuted_inputs[-1])

    args = [inputs[0]]
    b = inputs[1]
    if with_scaling:
        scale = permuted_inputs[2]
        rescale_b = torch.empty_like(b, dtype=torch.float16)
        for i in range(N):
            for j in range(K):
                if with_zeros:
                    if zeros_mode == "original":
                        rescale_b[i, j] = (b[i, j] - zeros) * scale[i, j // group_size]
                    elif zeros_mode == "rescale":
                        rescale_b[i, j] = (b[i, j] * scale[i, j // group_size] + zeros)
                    else:
                        raise NotImplementedError
                else:
                    rescale_b[i, j] = b[i, j] * scale[i, j // group_size]
        args.append(rescale_b.t().cuda())
    else:
        args.append(b.t().cuda().to(torch.float16))

    ref_result = torch.matmul(*args)

    print(ref_result)
    if zeros_mode == "rescale":
        torch.testing.assert_close(permuted_inputs[-1], ref_result, rtol=1e2, atol=1e2)
    else:
        torch.testing.assert_close(permuted_inputs[-1], ref_result, rtol=1e2, atol=1e2)


def assert_matmul_weight_transform_dequant_with_default_correctness(
    M,
    N,
    K,
    trans_A=False,
    trans_B=True,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    bit=4,
    storage_dtype="int8",
    source_format="uint",
    with_scaling=False,
    with_zeros=False,
    group_size=-1,
    fast_decoding=False,
    zeros_mode="original",
):
    import numpy as np
    from bitblas.quantization import general_compress, interleave_weight

    matmul = MatmulDequantizeMMAWeightPropagationScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        num_bits=bit,
        storage_dtype=storage_dtype,
        source_format=source_format,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        group_size=group_size,
        fast_decoding=fast_decoding,
        zeros_mode=zeros_mode,
    ).with_default_config()
    if verbose:
        print(matmul)
    mod, params = tilelang.lower(matmul)

    src_code = mod.imported_modules[0].get_source()
    # src_code is the generated cuda source
    assert src_code is not None
    if verbose:
        print(src_code)
    input_shape = (M, K)
    weight_shape = (N, K)
    output_shape = (M, N)
    inputs = []
    inputs.append(torch.rand(input_shape, dtype=torch.float16).cuda() - 0.5)
    maxq = 2**(bit - 1)
    if group_size == -1:
        group_size = K

    if source_format == "uint":
        inputs.append(torch.randint(0, maxq, weight_shape, dtype=torch.int8).cuda())
    elif source_format == "int":
        inputs.append(torch.randint(-maxq, maxq, weight_shape, dtype=torch.int8).cuda())
    else:
        raise NotImplementedError

    inputs.append(torch.rand(output_shape, dtype=torch.float16).cuda())

    intweight = inputs[1]
    intweight = intweight.cpu().to(torch.int8)
    if source_format == "int":
        intweight = intweight + maxq

    ladder_permutate_config = bitblas.ops.LadderPermutateConfig(
        M=N,
        N=K,
        storage_dtype=storage_dtype,
        propagate_kind="B",
        transform_kind=3,
        transpose_matrix=True,
    )

    ladder_permutate = bitblas.ops.LadderPermutate(ladder_permutate_config)

    LB = ladder_permutate(intweight.cpu()).cuda().reshape(N, K)
    permuted_inputs = []
    permuted_inputs.append(inputs[0])
    qw = general_compress(LB.cpu().numpy(), source_bits=bit, storage_dtype=np.int8)

    # lop3 transformation
    if fast_decoding:
        qw = interleave_weight(qw, bit, target_dtype=in_dtype)
    qw_shape = [int(v) for v in matmul.buffer_map[matmul.params[1]].shape]
    qw = qw.reshape(qw_shape)
    permuted_inputs.append(torch.from_numpy(qw).cuda())
    if with_scaling:
        permuted_inputs.append(torch.randn((N, K // group_size), dtype=torch.float16).cuda())

    zeros = None
    if with_zeros:
        if zeros_mode == "original":
            zeros = torch.ones([N, K // group_size], dtype=torch.float16).cuda() * maxq
        elif zeros_mode == "rescale":
            scale = permuted_inputs[2]
            original_zeros = (torch.ones([N, K // group_size], dtype=torch.float16).cuda() * maxq)
            zeros = -(original_zeros * scale.cuda())
        else:
            raise NotImplementedError

    if with_scaling and with_zeros:
        permuted_inputs.append(zeros)

    permuted_inputs.append(inputs[2])

    mod = tilelang.Profiler(mod, params, [], tilelang.TensorSupplyType.Integer)

    mod(*permuted_inputs)

    print(permuted_inputs[-1])

    args = [inputs[0]]
    b = inputs[1]

    if with_scaling:
        scale = permuted_inputs[2]
        rescale_b = torch.empty_like(b, dtype=torch.float16)
        for i in range(N):
            for j in range(K):
                if with_zeros:
                    zeros = permuted_inputs[3]
                    if zeros_mode == "original":
                        rescale_b[i, j] = (b[i, j] - zeros[i, j // group_size]) * scale[i, j //
                                                                                        group_size]
                    elif zeros_mode == "rescale":
                        rescale_b[i, j] = (
                            b[i, j] * scale[i, j // group_size] + zeros[i, j // group_size])
                    else:
                        raise NotImplementedError
                else:
                    rescale_b[i, j] = b[i, j] * scale[i, j // group_size]
        args.append(rescale_b.t().cuda())
    else:
        args.append(b.t().cuda().to(torch.float16))

    ref_result = torch.matmul(*args)
    print(ref_result)
    if zeros_mode == "rescale":
        torch.testing.assert_close(permuted_inputs[-1], ref_result, rtol=1e-2, atol=1e0)
    else:
        torch.testing.assert_close(permuted_inputs[-1], ref_result, rtol=1e-2, atol=1e0)


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


def test_matmul_int4_fine_grained():
    # Default
    assert_matmul_int4_fine_grained_with_default_correctness(256, 256, 256)
    # Pipeline
    assert_matmul_int4_fine_grained_apply_config_correctness(1024, 1024, 1024, num_stages=2)
    assert_matmul_int4_fine_grained_apply_config_correctness(1024, 1024, 1024, num_stages=1)
    # L2 Cache
    assert_matmul_int4_fine_grained_apply_config_correctness(
        1024, 1024, 1024, enable_rasterization=True)


def test_matmul_int4_weight_propagation():
    # Default
    assert_matmul_int4_weight_propagation_with_default_correctness(256, 256, 256)
    # Pipeline
    assert_matmul_int4_weight_propagation_apply_config__correctness(1024, 1024, 1024, num_stages=2)
    assert_matmul_int4_weight_propagation_apply_config__correctness(1024, 1024, 1024, num_stages=1)
    # L2 Cache
    assert_matmul_int4_weight_propagation_apply_config__correctness(
        1024, 1024, 1024, enable_rasterization=True)


def test_matmul_int4xint2_fine_grained():
    # Default
    assert_matmul_fine_grained_dequant_int4_with_default_correctness(256, 256, 256)
    assert_matmul_fine_grained_dequant_int4_with_default_correctness(
        256, 256, 256, fast_decoding=True)
    # Pipeline
    assert_matmul_fine_grained_dequant_int4_apply_config_correctness(1024, 1024, 1024, num_stages=2)
    # L2 Cache
    assert_matmul_fine_grained_dequant_int4_apply_config_correctness(
        1024, 1024, 1024, enable_rasterization=True)


def test_matmul_int4_weight_transform_dequant():
    # Default
    assert_matmul_weight_transform_dequant_int4_with_default_correctness(256, 256, 256)
    assert_matmul_weight_transform_dequant_int4_with_default_correctness(
        256, 256, 256, fast_decoding=True)
    # Pipeline
    assert_matmul_weight_transform_dequant_int4_apply_config_correctness(
        1024, 1024, 1024, num_stages=2)
    assert_matmul_weight_transform_dequant_int4_apply_config_correctness(
        1024, 1024, 1024, num_stages=1)
    # L2 Cache
    assert_matmul_weight_transform_dequant_int4_apply_config_correctness(
        1024, 1024, 1024, enable_rasterization=True)


def test_matmul_weight_propagation():
    # Default
    assert_matmul_weight_propagation_with_default_correctness(1024, 1024, 1024)
    # Pipeline
    assert_matmul_weight_propagation_apply_config_correctness(1024, 1024, 1024, num_stages=2)
    assert_matmul_weight_propagation_apply_config_correctness(1024, 1024, 1024, num_stages=1)
    # L2 Cache
    assert_matmul_weight_propagation_apply_config_correctness(
        1024, 1024, 1024, enable_rasterization=True)


def test_matmul_blocked_dequant_with_default():
    assert_matmul_blocked_dequant_with_default_correctness(
        1024, 1024, 1024, source_format="uint", bit=4)
    assert_matmul_blocked_dequant_with_default_correctness(
        1024, 1024, 1024, source_format="uint", bit=2)
    assert_matmul_blocked_dequant_with_default_correctness(
        1024, 1024, 1024, source_format="uint", bit=4, with_scaling=True)
    assert_matmul_blocked_dequant_with_default_correctness(
        1024,
        1024,
        1024,
        source_format="uint",
        bit=4,
        with_scaling=True,
        with_zeros=True,
    )
    assert_matmul_blocked_dequant_with_default_correctness(
        1024, 1024, 1024, source_format="uint", bit=4, fast_decoding=True)


def test_matmul_fine_grained_dequant_with_default():
    assert_matmul_fine_grained_dequant_with_default_correctness(
        1024, 1024, 1024, source_format="uint", bit=4)
    assert_matmul_fine_grained_dequant_with_default_correctness(
        1024, 1024, 1024, source_format="uint", bit=2)
    assert_matmul_fine_grained_dequant_with_default_correctness(
        1024, 1024, 1024, source_format="uint", bit=4, with_scaling=True)
    assert_matmul_fine_grained_dequant_with_default_correctness(
        1024,
        1024,
        1024,
        source_format="uint",
        bit=4,
        with_scaling=True,
        with_zeros=True,
    )
    assert_matmul_fine_grained_dequant_with_default_correctness(
        1024, 1024, 1024, source_format="uint", bit=4, fast_decoding=True)
    assert_matmul_fine_grained_dequant_with_default_correctness(
        1024,
        1024,
        1024,
        source_format="uint",
        bit=4,
        with_scaling=True,
        fast_decoding=True,
    )
    assert_matmul_fine_grained_dequant_with_default_correctness(
        1024,
        1024,
        1024,
        source_format="uint",
        bit=4,
        with_scaling=True,
        with_zeros=True,
        fast_decoding=True,
    )


def test_matmul_weight_transform_dequant_with_default():
    assert_matmul_weight_transform_dequant_with_default_correctness(
        1024, 1024, 1024, source_format="uint", bit=4)
    assert_matmul_weight_transform_dequant_with_default_correctness(
        1024, 1024, 1024, source_format="uint", bit=2)
    assert_matmul_weight_transform_dequant_with_default_correctness(
        1024, 1024, 1024, source_format="uint", bit=4, with_scaling=True)
    assert_matmul_weight_transform_dequant_with_default_correctness(
        1024, 1024, 1024, source_format="uint", bit=4, with_scaling=True, with_zeros=True)
    assert_matmul_weight_transform_dequant_with_default_correctness(
        1024, 1024, 1024, source_format="uint", bit=4, fast_decoding=True)
    assert_matmul_weight_transform_dequant_with_default_correctness(
        1024,
        1024,
        1024,
        source_format="uint",
        bit=4,
        with_scaling=True,
        fast_decoding=True,
    )
    assert_matmul_weight_transform_dequant_with_default_correctness(
        1024,
        1024,
        1024,
        source_format="uint",
        bit=4,
        with_scaling=True,
        fast_decoding=True,
        with_zeros=True,
    )


if __name__ == "__main__":
    bitblas.testing.main()
