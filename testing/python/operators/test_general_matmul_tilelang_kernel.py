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

from bitblas.ops.general_matmul.tilelang.dequantize import (
    MatmulDequantizeScheduler,)

import torch
import torch.backends

torch.manual_seed(0)


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
    enable_rasterization=False,
):
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
    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype)) - 0.5
    B = (torch.rand(N, K, device="cuda", dtype=getattr(torch, in_dtype)) if trans_B else torch.rand(
        K, N, device="cuda", dtype=getattr(torch, in_dtype))) - 0.5
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, out_dtype))
    mod = tl.Profiler(mod, params, [], tl.TensorSupplyType.Integer)

    latency = mod.do_bench(mod.func, warmup=25)

    # Ensure that the latency is not None
    assert latency is not None

    mod(A, B, C)

    # Get Reference Result
    ref_c = (
        torch.matmul(A, B.T).to(getattr(torch, out_dtype)) if trans_B else torch.matmul(A, B).to(
            getattr(torch, out_dtype)))

    # from bitblas.ops import Matmul, MatmulConfig
    # matmul_config = MatmulConfig(
    #     M=M,
    #     N=N,
    #     K=K,
    #     propagate_a=False,
    #     propagate_b=False,
    # )
    # matmul = Matmul(matmul_config, enable_tuning=False)
    # prim_func = matmul.prim_func
    # intrin_info = bitblas.base.hint.IntrinInfo(
    #     in_dtype=in_dtype,
    #     out_dtype=accum_dtype,
    #     trans_b=True,
    #     input_transform_kind=0,
    #     weight_transform_kind=0,
    # )

    # arch = bitblas.base.CUDA(target="cuda")

    # sch = bitblas.gpu.MatmulTensorizationMMA().apply_config(
    #     prim_func,
    #     config=bitblas.base.Hint.from_dict({
    #         "arch": arch,
    #         "block": [64, 64],
    #         "warp": [32, 32],
    #         "rstep": [32],
    #         "pipeline_stage": 2,
    #         "use_async": True,
    #         "intrin_info": intrin_info,
    #         "shared_scope": "shared.dyn",
    #         "vectorize": {
    #             "b": 8,
    #             "a": 8
    #         },
    #     }),
    # )

    # with tvm.transform.PassContext(config={
    #         "tir.use_async_copy": True,
    #         "tir.merge_static_smem": False
    # }):
    #     rt_mod = tvm.build(sch.mod, target="cuda")
    # from tvm.contrib.dlpack import to_pytorch_func

    # torch_func = to_pytorch_func(rt_mod)

    # matmul_c = torch.zeros(M, N, device="cuda", dtype=getattr(torch, out_dtype))
    # torch_func(A, B, matmul_c)

    # with open("debug/matmul_ref.cu", "w") as f:
    #     f.write(rt_mod.imported_modules[0].get_source())

    # with open("debug/matmul_tl.cu", "w") as f:
    #     f.write(src_code)

    # torch.testing.assert_close(matmul_c, ref_c, rtol=1e0, atol=1e-1)

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

    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype)) - 0.5
    B = torch.rand(N, K, device="cuda", dtype=getattr(torch, in_dtype)) - 0.5
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))

    mod = tl.Profiler(mod, params, [], tl.TensorSupplyType.Integer)

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
        bit=bit,
        storage_dtype=storage_dtype,
        source_format=source_format,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        group_size=group_size,
        fast_decoding=fast_decoding,
        zeros_mode=zeros_mode,
    ).with_default_config()

    mod, params = tl.lower(matmul)
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

    ref_result = torch.matmul(inputs[0], inputs[1].t().to(torch.float16))

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
            permuted_inputs.append(
                torch.ones([N, K // group_size], dtype=torch.float16).cuda() * zeros)
        elif zeros_mode == "rescale":
            original_zeros = torch.ones([N, K // group_size], dtype=torch.float16).cuda() * zeros
            scaled_zeros = original_zeros * permuted_inputs[-1]
            permuted_inputs.append(scaled_zeros)
        elif zeros_mode == "quantized":
            original_zeros = torch.ones([K // group_size, N], dtype=torch.int8).cuda() * zeros
            qzeros = general_compress(
                original_zeros.cpu().numpy(), source_bits=bit, storage_dtype=np.int8)
            permuted_inputs.append(torch.from_numpy(qzeros).cuda())
        else:
            raise NotImplementedError

    permuted_inputs.append(inputs[2])

    mod = tl.Profiler(mod, params, [], tl.TensorSupplyType.Integer)

    mod(*permuted_inputs)

    print(permuted_inputs[-1])
    print(ref_result)
    if zeros_mode == "rescale":
        torch.testing.assert_close(permuted_inputs[-1], ref_result, rtol=1e2, atol=1e0)
    else:
        torch.testing.assert_close(permuted_inputs[-1], ref_result, rtol=1e2, atol=1e0)


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


def test_matmul_blocked_dequant_with_default():
    assert_matmul_blocked_dequant_with_default_correctness(
        1024, 1024, 1024, source_format="uint", bit=4)
    assert_matmul_blocked_dequant_with_default_correctness(
        1024, 1024, 1024, source_format="uint", bit=2)


if __name__ == "__main__":
    bitblas.testing.main()
