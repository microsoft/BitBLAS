# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import bitblas
from bitblas import tvm
from bitblas.ops.general_matmul.tirscript import (
    matmul_select_implementation,
)
import logging
from bitblas import set_log_level

set_log_level(logging.DEBUG)

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
    
    
# fmt: on
if __name__ == "__main__":
    # bitblas.testing.main()
    assert_correctness_with_block_reduce(M=256, N=256, K=256, in_dtype="float16", out_dtype="float16", accum_dtype="float16", propagate_a=0, propagate_b=0)

