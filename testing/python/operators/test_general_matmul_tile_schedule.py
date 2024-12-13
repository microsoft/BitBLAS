# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import bitblas
import bitblas.testing
from bitblas import tvm
from bitblas.ops.general_matmul.tirscript import (
    matmul_select_implementation,
    matmul_dequantize_select_implementation,
)
import logging
from bitblas import set_log_level
import numpy as np

np.random.seed(0)

set_log_level(logging.DEBUG)


def check_reduce(rt_mod):
    # source = rt_mod.imported_modules[0].get_source()
    # assert "red_buf" in source
    # TODO(lei): After improve lower_thraed_all_reduce pass
    # The red_buf has been merged into dynamic shared memory
    # ref to: https://github.com/microsoft/BitBLAS/pull/183
    return True


# fmt: off
def assert_correctness_with_block_reduce(
    M=None,
    N=256,
    K=256,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    propagate_a=0,
    propagate_b=0,
):
    matmul_func = matmul_select_implementation(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        propagate_a=propagate_a,
        propagate_b=propagate_b)["main"]
    target = bitblas.auto_detect_nvidia_target()
    intrin_info = bitblas.base.hint.IntrinInfo(
        in_dtype=in_dtype,
        out_dtype=accum_dtype,
        trans_b=True,
        input_transform_kind=propagate_a,
        weight_transform_kind=propagate_b,
    )
    arch = bitblas.base.CUDA(target=target)
    ref_sch = bitblas.gpu.MatmulTensorizationMMA().apply_config(
        matmul_func,
        config=bitblas.base.Hint.from_dict({
            "arch": arch,
            "block": [16, 128],
            "warp": [16, 32],
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
            "tir.merge_static_smem": True
    }):
        ref_rt_mod = tvm.build(ref_sch.mod, target=target)

    block_reduce_sch = bitblas.gpu.MatmulTensorizationMMA().apply_config(
        matmul_func,
        config=bitblas.base.Hint.from_dict({
            "arch": arch,
            "block": [16, 128],
            "warp": [16, 32],
            "rstep": [32],
            "pipeline_stage": 2,
            "use_async": True,
            "intrin_info": intrin_info,
            "shared_scope": "shared.dyn",
            "vectorize": {
                "b": 8,
                "a": 8
            },
            "block_reduction_depth": 2,
        }),
    )
    with tvm.transform.PassContext(config={
            "tir.use_async_copy": True,
            "tir.merge_static_smem": True
    }):
        block_reduce_rt_mod = tvm.build(block_reduce_sch.mod, target=target)

    tvm_a = tvm.nd.array(np.random.randn(M, K).astype(in_dtype), device=tvm.cuda())
    tvm_b = tvm.nd.array(np.random.randn(N, K).astype(in_dtype), device=tvm.cuda())
    tvm_c = tvm.nd.array(np.random.randn(M, N).astype(out_dtype), device=tvm.cuda())
    tvm_c_ref = tvm.nd.array(np.zeros((M, N)).astype(out_dtype), device=tvm.cuda())

    ref_rt_mod(tvm_a, tvm_b, tvm_c_ref)

    block_reduce_rt_mod(tvm_a, tvm_b, tvm_c)
    np.testing.assert_allclose(tvm_c.asnumpy(), tvm_c_ref.asnumpy(), rtol=1e2, atol=1e-2)


def test_assert_correctness_with_block_reduce():
    assert_correctness_with_block_reduce(
        M=256,
        N=256,
        K=256,
        in_dtype="float16",
        out_dtype="float16",
        accum_dtype="float16",
        propagate_a=0,
        propagate_b=0)


def assert_correctness_with_ladder_ldmatrix_propagate(
    M=None,
    N=256,
    K=256,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    block_reduction_depth=1,
):
    propagate_b = 3
    matmul_func = matmul_select_implementation(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        propagate_a=0,
        propagate_b=propagate_b)["main"]

    target = bitblas.auto_detect_nvidia_target()
    intrin_info = bitblas.base.hint.IntrinInfo(
        in_dtype=in_dtype,
        out_dtype=accum_dtype,
        trans_b=True,
        input_transform_kind=0,
        weight_transform_kind=propagate_b,
    )
    arch = bitblas.base.CUDA(target=target)
    block_reduce_sch = bitblas.gpu.MatmulTensorizationMMA().apply_config(
        matmul_func,
        config=bitblas.base.Hint.from_dict({
            "arch": arch,
            "block": [16, 128],
            "warp": [16, 32],
            "rstep": [128],
            "pipeline_stage": 4,
            "use_async": True,
            "intrin_info": intrin_info,
            "shared_scope": "shared.dyn",
            "vectorize": {
                "b": 8,
                "a": 8
            },
            "block_reduction_depth": block_reduction_depth,
        }),
    )
    with tvm.transform.PassContext(config={
            "tir.use_async_copy": True,
            "tir.merge_static_smem": False
    }):
        block_reduce_rt_mod = tvm.build(block_reduce_sch.mod, target=target)
    # Evaluate the correctness
    import numpy as np
    a = np.random.randn(M, K).astype(np.float16 if in_dtype == "float16" else "int8")
    b = np.random.randn(N, K).astype(np.float16 if in_dtype == "float16" else "int8")
    c = np.random.randn(M, N).astype(np.float16 if in_dtype == "float16" else "int8")

    ladder_permutate_config = bitblas.ops.LadderPermutateConfig(
        M=N,
        N=K,
        datatype=in_dtype,
        storage_dtype=in_dtype,
        transform_kind=propagate_b,
        transpose_matrix=True,
    )

    ladder_permutate = bitblas.ops.LadderPermutate(ladder_permutate_config)

    tvm_b = tvm.nd.array(b)
    tvm_transformed_b = ladder_permutate.get_profile_tensors()[-1]
    ladder_permutate.rt_mod(tvm_b, tvm_transformed_b)
    ladder_transformed_b = tvm_transformed_b.asnumpy()

    # transformed_b = b
    tvm_a = tvm.nd.array(a, device=tvm.cuda(0))
    tvm_b = tvm.nd.array(ladder_transformed_b, device=tvm.cuda(0))
    tvm_c = tvm.nd.array(c, device=tvm.cuda(0))

    block_reduce_rt_mod(tvm_a, tvm_b, tvm_c)
    print("removed ldmatrix b output is \n", tvm_c)

    np_c = np.dot(a, b.T)
    print("numpy output is \n", np_c)
    np.testing.assert_allclose(tvm_c.asnumpy(), np_c, rtol=1e2, atol=1e-1)


def test_assert_correctness_with_ladder_ldmatrix_propagate():
    assert_correctness_with_ladder_ldmatrix_propagate(
        M=256, N=256, K=256, in_dtype="float16", out_dtype="float16", accum_dtype="float16")
    assert_correctness_with_ladder_ldmatrix_propagate(
        M=256, N=256, K=256, in_dtype="int8", out_dtype="int8", accum_dtype="int32")


def assert_dequant_correctness_with_block_reduce(
    M=None,
    N=1024,
    K=1024,
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
    with_bias=False,
    layout="nt",
    zeros_mode="original",
    propagate_b=False,
):
    matmul_func = matmul_dequantize_select_implementation(
        M=M,
        N=N,
        K=K,
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
        with_bias=with_bias,
        layout=layout,
        zeros_mode=zeros_mode,
        propagate_a=False,
        propagate_b=propagate_b)["main"]
    target = bitblas.auto_detect_nvidia_target()
    intrin_info = bitblas.base.hint.IntrinInfo(
        in_dtype=in_dtype,
        out_dtype=accum_dtype,
        trans_b=True,
        input_transform_kind=0,
        weight_transform_kind=propagate_b,
    )

    arch = bitblas.base.CUDA(target=target)

    ref_sch = bitblas.gpu.MatmulTensorizationMMAWithDequantizeInfo().apply_config(
        matmul_func,
        config=bitblas.base.Hint.from_dict({
            "arch": arch,
            "block": [16, 128],
            "warp": [16, 32],
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
        ref_rt_mod = tvm.build(ref_sch.mod, target=target)

    block_reduce_sch = bitblas.gpu.MatmulTensorizationMMAWithDequantizeInfo().apply_config(
        matmul_func,
        config=bitblas.base.Hint.from_dict({
            "arch": arch,
            "block": [16, 128],
            "warp": [16, 32],
            "rstep": [32],
            "pipeline_stage": 2,
            "use_async": True,
            "intrin_info": intrin_info,
            "shared_scope": "shared.dyn",
            "vectorize": {
                "b": 8,
                "a": 8
            },
            "block_reduction_depth": 2,
        }),
    )
    with tvm.transform.PassContext(config={
            "tir.use_async_copy": True,
            "tir.merge_static_smem": False
    }):
        block_reduce_rt_mod = tvm.build(block_reduce_sch.mod, target=target)

    check_reduce(block_reduce_rt_mod)

    # TODO: Should be more generalized.
    # Check correctness
    import numpy as np
    elems_per_byte = 8 // bit
    tvm_a = tvm.nd.array(np.random.randn(M, K).astype(in_dtype), device=tvm.cuda())
    tvm_b = tvm.nd.array(
        np.random.randint(-1, 2, (N, K // elems_per_byte)).astype("int8"), device=tvm.cuda())
    tvm_c = tvm.nd.array(np.random.randn(M, N).astype(out_dtype), device=tvm.cuda())
    tvm_c_ref = tvm.nd.array(np.zeros((M, N)).astype(out_dtype), device=tvm.cuda())

    ref_rt_mod(tvm_a, tvm_b, tvm_c_ref)

    block_reduce_rt_mod(tvm_a, tvm_b, tvm_c)
    np.testing.assert_allclose(tvm_c.asnumpy(), tvm_c_ref.asnumpy(), rtol=1e2, atol=1e0)


def test_assert_dequant_correctness_with_block_reduce():
    assert_dequant_correctness_with_block_reduce(
        M=256,
        N=256,
        K=256,
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
        with_bias=False,
        layout="nt",
        zeros_mode="original",
        propagate_b=False)


def assert_dequantize_correctness_with_ladder_ldmatrix_propagate(
    M=None,
    N=1024,
    K=1024,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    bit=4,
    storage_dtype="int8",
    source_format="uint",
    with_scaling=True,
    with_zeros=False,
    group_size=-1,
    fast_decoding=False,
    with_bias=False,
    layout="nt",
    zeros_mode="original",
):

    if group_size == -1:
        group_size = K
    propagate_b = 3
    matmul_func = matmul_dequantize_select_implementation(
        M=M,
        N=N,
        K=K,
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
        with_bias=with_bias,
        layout=layout,
        zeros_mode=zeros_mode,
        propagate_a=False,
        propagate_b=propagate_b)["main"]
    target = bitblas.auto_detect_nvidia_target()
    intrin_info = bitblas.base.hint.IntrinInfo(
        in_dtype=in_dtype,
        out_dtype=accum_dtype,
        trans_b=True,
        input_transform_kind=0,
        weight_transform_kind=propagate_b,
    )

    arch = bitblas.base.CUDA(target=target)

    block_reduce_sch = bitblas.gpu.MatmulTensorizationMMAWithDequantizeInfo().apply_config(
        matmul_func,
        config=bitblas.base.Hint.from_dict({
            "arch": arch,
            "block": [128, 128],
            "warp": [64, 64],
            "rstep": [32],
            "pipeline_stage": 2,
            "use_async": True,
            "intrin_info": intrin_info,
            "shared_scope": "shared.dyn",
            "vectorize": {
                "b": 8,
                "a": 8
            },
            "block_reduction_depth": 1,
        }),
    )

    with tvm.transform.PassContext(config={
            "tir.use_async_copy": True,
            "tir.merge_static_smem": False,
            "tir.disable_cse_tir": True
    }):
        rt_mod = tvm.build(block_reduce_sch.mod, target=target)
    src_code = rt_mod.imported_modules[0].get_source()
    # print(src_code)
    assert src_code is not None

    check_reduce(rt_mod)

    # TODO: Should be more generalized.
    # Check correctness
    import torch
    torch.manual_seed(0)

    input_shape = (M, K)
    weight_shape = (N, K) if layout == "nt" else (K, N)

    a = torch.randn(input_shape, dtype=torch.float16).cuda() - 0.5
    weight_shape = (N, K)
    maxq = 2**(bit - 1)
    if source_format == "uint":
        b = torch.randint(0, maxq, weight_shape, dtype=torch.int8).cuda()
    elif source_format == "int":
        b = torch.randint(-maxq, maxq, weight_shape, dtype=torch.int8).cuda()
    else:
        raise NotImplementedError
    ref_inputs = []
    ref_inputs.append(a)
    ref_inputs.append(b)
    qb = bitblas.quantization.general_compress(b.cpu().numpy())
    qb = torch.from_numpy(qb)
    scale = torch.randn((N, K // group_size), dtype=torch.float16)
    maxq = 2**(bit - 1)
    zeros = None
    if with_scaling:
        ref_inputs.append(scale.cuda())
    if with_scaling and with_zeros:
        if zeros_mode == "original":
            zeros = torch.ones([N, K // group_size], dtype=torch.float16).cuda() * maxq
        elif zeros_mode == "rescale":
            original_zeros = torch.ones([N, K // group_size], dtype=torch.float16).cuda() * maxq
            zeros = (original_zeros * scale.cuda())
        else:
            raise NotImplementedError
        ref_inputs.append(zeros)

    c = torch.randn(M, N, dtype=torch.float16)

    ladder_permutate_config = bitblas.ops.LadderPermutateConfig(
        M=N,
        N=K,
        dequantize_bits=bit,
        storage_dtype="int8",
        transpose_matrix=True,
        transform_kind=propagate_b,
    )

    ladder_permutate = bitblas.ops.LadderPermutate(ladder_permutate_config)
    from bitblas.utils import tvm_tensor_to_torch
    transformed_b = tvm_tensor_to_torch(ladder_permutate.get_profile_tensors()[-1]).cpu()

    tvm_b = tvm.nd.array(qb.numpy())
    tvm_transformed_b = tvm.nd.array(transformed_b.numpy())
    ladder_permutate.rt_mod(tvm_b, tvm_transformed_b)

    if fast_decoding:
        lop3_permutate_config = bitblas.ops.LOP3PermutateConfig(
            M=N,
            N=K,
            storage_dtype="int8",
            dequantize_bits=bit,
        )

        lop3_permutate = bitblas.ops.LOP3Permutate(lop3_permutate_config)

        tvm_transformed_b_lop3 = lop3_permutate.get_profile_tensors()[-1]
        torch_transformed_b = tvm_tensor_to_torch(tvm_transformed_b).cpu().view(N, K // (8 // bit))
        torch_transformed_b_lop3 = tvm_tensor_to_torch(tvm_transformed_b_lop3).cpu()
        lop3_permutate.forward(torch_transformed_b, torch_transformed_b_lop3)
        tvm_transformed_b = tvm.nd.array(
            torch_transformed_b_lop3.view(torch.int8).view(tvm_transformed_b.shape).numpy())

    transformed_b = tvm_transformed_b.asnumpy()
    transformed_b = torch.from_numpy(transformed_b)

    from tvm.contrib.dlpack import to_pytorch_func

    torch_func = to_pytorch_func(rt_mod)

    a = a.cuda()
    transformed_b = transformed_b.cuda()
    c = c.cuda()
    scale = scale.cuda()
    args = [a, transformed_b]
    if with_scaling:
        args.append(scale)
    if with_scaling and with_zeros:
        zeros = zeros.cuda()
        args.append(zeros)
    args.append(c)

    torch_func(*args)


    def ref_program(A, intweight, scale=None, zeros=None, Bias=None):
        import torch

        B = intweight
        _, K = B.shape

        if with_scaling:
            # Calculate group indices for each column (group_size determines the grouping)
            group_indices = torch.arange(K, device=B.device) // group_size

            # Broadcast zeros and scale to match the shape of B
            scale_expanded = scale[:, group_indices]  # Shape: [N, K]

            if with_zeros:
                if zeros_mode == "original":
                    zeros_expanded = zeros[:, group_indices]  # Shape: [N, K]
                    # Subtract zeros and then scale
                    rescale_b = (B - zeros_expanded) * scale_expanded
                elif zeros_mode == "rescale":
                    zeros_expanded = zeros[:, group_indices]  # Shape: [N, K]
                    # Scale first and then add zeros
                    rescale_b = B * scale_expanded - zeros_expanded
                elif zeros_mode == "quantized":
                    dequant_zeros = (
                        torch.zeros(zeros.shape[0], zeros.shape[1] * 8 // 4,
                                    dtype=torch.half).to(torch.half).to(zeros.device))
                    for i in range(dequant_zeros.shape[0]):
                        for j in range(dequant_zeros.shape[1]):
                            dequant_zeros[i][j] = ((zeros[i][j // 2] >> (4 * (j % 2))) & 0xF).to(torch.half)
                    zeros_expanded = dequant_zeros[group_indices, :]  # Shape: [N, K]
                    # Subtract zeros and then scale
                    rescale_b = (B - zeros_expanded) * scale_expanded
                else:
                    # Raise an error for unsupported zeros_mode
                    raise NotImplementedError(f"Unsupported zeros_mode: {zeros_mode}")
            else:
                # Apply scaling without zeros adjustment
                rescale_b = B * scale_expanded
        else:
            # If scaling is disabled, directly use B
            rescale_b = B

        C = torch.matmul(A.to(torch.float), rescale_b.T.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        if with_bias:
            C = C + Bias
        return C

    print(f"A = {a}")
    print(f"intweight = {b}")
    print(f"scale = {scale if with_scaling else None}")
    ref_result = ref_program(*ref_inputs)
    print("c is \n", c)
    print("ref_c is \n", ref_result)

    bitblas.testing.torch_assert_close(c, ref_result, rtol=1e2, atol=1e0)


def test_assert_dequantize_correctness_with_ladder_ldmatrix_propagate():
    assert_dequantize_correctness_with_ladder_ldmatrix_propagate(
        M=256,
        N=256,
        K=256,
        in_dtype="float16",
        out_dtype="float16",
        accum_dtype="float16",
        bit=4,
        storage_dtype="int8",
        source_format="uint",
        with_scaling=True,
        with_zeros=False,
        group_size=-1,
        fast_decoding=False,
        with_bias=False,
        layout="nt",
        zeros_mode="original")
    assert_dequantize_correctness_with_ladder_ldmatrix_propagate(
        M=256,
        N=256,
        K=256,
        in_dtype="float16",
        out_dtype="float16",
        accum_dtype="float16",
        bit=4,
        storage_dtype="int8",
        source_format="uint",
        with_scaling=True,
        with_zeros=False,
        group_size=32,
        fast_decoding=False,
        with_bias=False,
        layout="nt",
        zeros_mode="original")
    assert_dequantize_correctness_with_ladder_ldmatrix_propagate(
        M=256,
        N=256,
        K=256,
        in_dtype="float16",
        out_dtype="float16",
        accum_dtype="float16",
        bit=4,
        storage_dtype="int8",
        source_format="uint",
        with_scaling=True,
        with_zeros=False,
        group_size=-1,
        fast_decoding=True,
        with_bias=False,
        layout="nt",
        zeros_mode="original")
    assert_dequantize_correctness_with_ladder_ldmatrix_propagate(
        M=256,
        N=256,
        K=256,
        in_dtype="float16",
        out_dtype="float16",
        accum_dtype="float16",
        bit=4,
        storage_dtype="int8",
        source_format="uint",
        with_scaling=True,
        with_zeros=True,
        group_size=-1,
        fast_decoding=True,
        with_bias=False,
        layout="nt",
        zeros_mode="original")
    assert_dequantize_correctness_with_ladder_ldmatrix_propagate(
        M=256,
        N=256,
        K=256,
        in_dtype="float16",
        out_dtype="float16",
        accum_dtype="float16",
        bit=4,
        storage_dtype="int8",
        source_format="uint",
        with_scaling=True,
        with_zeros=True,
        group_size=-1,
        fast_decoding=True,
        with_bias=False,
        layout="nt",
        zeros_mode="rescale")


# fmt: on
if __name__ == "__main__":
    bitblas.testing.main()
