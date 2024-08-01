# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import bitblas
from bitblas import tvm
from bitblas.ops.general_matmul.tirscript import (
    matmul_select_implementation,
    matmul_dequantize_select_implementation,
)
import logging
from bitblas import set_log_level

set_log_level(logging.DEBUG)

def check_reduce(rt_mod):
    source = rt_mod.imported_modules[0].get_source()
    assert "red_buf" in source

# fmt: off
def assert_correctness_with_block_reduce(
    M=None, N=256, K=256,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    propagate_a = 0,
    propagate_b = 0,
):
    matmul_func = matmul_select_implementation(
        M=M, N=N, K=K, in_dtype=in_dtype, out_dtype=out_dtype, accum_dtype=accum_dtype, 
        propagate_a=propagate_a, propagate_b=propagate_b
    )["main"]
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
        config=bitblas.base.Hint.from_dict(
            {
                "arch": arch,
                "block": [16, 128],
                "warp": [16, 32],
                "rstep": [128],
                "pipeline_stage": 4,
                "use_async": True,
                "intrin_info": intrin_info,
                "shared_scope": "shared.dyn",
                "vectorize": {"b": 8, "a": 8},
            }
        ),
    )
    with tvm.transform.PassContext(config={"tir.use_async_copy": True, "tir.merge_static_smem": False}):
        ref_rt_mod = tvm.build(ref_sch.mod, target=target)

    block_reduce_sch = bitblas.gpu.MatmulTensorizationMMA().apply_config(
        matmul_func,
        config=bitblas.base.Hint.from_dict(
            {
                "arch": arch,
                "block": [16, 128],
                "warp": [16, 32],
                "rstep": [128],
                "pipeline_stage": 4,
                "use_async": True,
                "intrin_info": intrin_info,
                "shared_scope": "shared.dyn",
                "vectorize": {"b": 8, "a": 8},
                "block_reduction_depth": 2,
            }
        ),
    )
    with tvm.transform.PassContext(config={"tir.use_async_copy": True, "tir.merge_static_smem": False}):
        block_reduce_rt_mod = tvm.build(block_reduce_sch.mod, target=target)
    
    # Check correctness
    import numpy as np
    tvm_a = tvm.nd.array(np.random.randn(M, K).astype(in_dtype), device=tvm.cuda())
    tvm_b = tvm.nd.array(np.random.randn(N, K).astype(in_dtype), device=tvm.cuda())
    tvm_c = tvm.nd.array(np.random.randn(M, N).astype(out_dtype), device=tvm.cuda())
    tvm_c_ref = tvm.nd.array(np.zeros((M, N)).astype(out_dtype), device=tvm.cuda())
    
    ref_rt_mod(tvm_a, tvm_b, tvm_c_ref)
    
    block_reduce_rt_mod(tvm_a, tvm_b, tvm_c)
    np.testing.assert_allclose(tvm_c.asnumpy(), tvm_c_ref.asnumpy(), rtol=1e-3, atol=1e-3)


def test_assert_correctness_with_block_reduce():
    assert_correctness_with_block_reduce(M=256, N=256, K=256, in_dtype="float16", out_dtype="float16", accum_dtype="float16", propagate_a=0, propagate_b=0)
    assert_correctness_with_block_reduce(M=256, N=256, K=256, in_dtype="float16", out_dtype="float16", accum_dtype="float16", propagate_a=0, propagate_b=2)
    assert_correctness_with_block_reduce(M=256, N=256, K=256, in_dtype="float16", out_dtype="float16", accum_dtype="float16", propagate_a=2, propagate_b=2)


def assert_correctness_with_ladder_ldmatrix_propagate(
    M=None, N=256, K=256,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    block_reduction_depth=1,
):
    matmul_func = matmul_select_implementation(
        M=M, N=N, K=K, in_dtype=in_dtype, out_dtype=out_dtype, accum_dtype=accum_dtype, 
        propagate_a=0, propagate_b=3
    )["main"]
    propagate_b = 3
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
        config=bitblas.base.Hint.from_dict(
            {
                "arch": arch,
                "block": [16, 128],
                "warp": [16, 32],
                "rstep": [128],
                "pipeline_stage": 4,
                "use_async": True,
                "intrin_info": intrin_info,
                "shared_scope": "shared.dyn",
                "vectorize": {"b": 8, "a": 8},
                "block_reduction_depth": block_reduction_depth,
            }
        ),
    )
    with tvm.transform.PassContext(config={"tir.use_async_copy": True, "tir.merge_static_smem": False}):
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
    
    ladder_permutate = bitblas.ops.LadderPermutate(
        ladder_permutate_config
    )

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


def test_assert_correctness_with_ladder_ldmatrix_propagate():
    assert_correctness_with_ladder_ldmatrix_propagate(M=256, N=256, K=256, in_dtype="float16", out_dtype="float16", accum_dtype="float16")
    assert_correctness_with_ladder_ldmatrix_propagate(M=256, N=256, K=256, in_dtype="int8", out_dtype="int8", accum_dtype="int32")


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
        M=M, N=N, K=K, in_dtype=in_dtype, out_dtype=out_dtype, accum_dtype=accum_dtype, 
        bit=bit, storage_dtype=storage_dtype, source_format=source_format, with_scaling=with_scaling, with_zeros=with_zeros, group_size=group_size, fast_decoding=fast_decoding, with_bias=with_bias, layout=layout, zeros_mode=zeros_mode, propagate_a=False, propagate_b=propagate_b
    )["main"]
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
        config=bitblas.base.Hint.from_dict(
            {
                "arch": arch,
                "block": [16, 128],
                "warp": [16, 32],
                "rstep": [128],
                "pipeline_stage": 4,
                "use_async": True,
                "intrin_info": intrin_info,
                "shared_scope": "shared.dyn",
                "vectorize": {"b": 8, "a": 8},
            }
        ),
    )
    with tvm.transform.PassContext(config={"tir.use_async_copy": True, "tir.merge_static_smem": False}):
        ref_rt_mod = tvm.build(ref_sch.mod, target=target)

    block_reduce_sch = bitblas.gpu.MatmulTensorizationMMAWithDequantizeInfo().apply_config(
        matmul_func,
        config=bitblas.base.Hint.from_dict(
            {
                "arch": arch,
                "block": [16, 128],
                "warp": [16, 32],
                "rstep": [128],
                "pipeline_stage": 4,
                "use_async": True,
                "intrin_info": intrin_info,
                "shared_scope": "shared.dyn",
                "vectorize": {"b": 8, "a": 8},
                "block_reduction_depth": 2,
            }
        ),
    )
    with tvm.transform.PassContext(config={"tir.use_async_copy": True, "tir.merge_static_smem": False}):
        block_reduce_rt_mod = tvm.build(block_reduce_sch.mod, target=target)

    check_reduce(block_reduce_rt_mod)

    # TODO: Should be more generalized.
    # Check correctness
    import numpy as np
    elems_per_byte = 8 // bit
    tvm_a = tvm.nd.array(np.random.randn(M, K).astype(in_dtype), device=tvm.cuda())
    tvm_b = tvm.nd.array(np.random.randint(-1, 2, (N, K // elems_per_byte)).astype("int8"), device=tvm.cuda())
    tvm_c = tvm.nd.array(np.random.randn(M, N).astype(out_dtype), device=tvm.cuda())
    tvm_c_ref = tvm.nd.array(np.zeros((M, N)).astype(out_dtype), device=tvm.cuda())
    
    ref_rt_mod(tvm_a, tvm_b, tvm_c_ref)
    
    block_reduce_rt_mod(tvm_a, tvm_b, tvm_c)
    np.testing.assert_allclose(tvm_c.asnumpy(), tvm_c_ref.asnumpy(), rtol=1e0, atol=1e0)


def test_assert_dequant_correctness_with_block_reduce():
    assert_dequant_correctness_with_block_reduce(M=256, N=256, K=256, in_dtype="float16", out_dtype="float16", accum_dtype="float16", bit=4, storage_dtype="int8", source_format="uint", with_scaling=False, with_zeros=False, group_size=-1, fast_decoding=False, with_bias=False, layout="nt", zeros_mode="original", propagate_b=False)
    
# fmt: on
if __name__ == "__main__":
    bitblas.testing.main()
