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
import numpy as np
import tvm

np.random.seed(0)

set_log_level(logging.DEBUG)


# fmt: off
def assert_correctness_with_block_reduce(
    M=256,
    N=256,
    K=256,
    in_dtype="float16",
    out_dtype="float32",
    accum_dtype="float32",
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
        propagate_b=propagate_b,
        layout="nt")["main"]
    target = tvm.target.Target("hip")
    intrin_info = bitblas.base.hint.IntrinInfo(
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        trans_b=True,
        input_transform_kind=propagate_a,
        weight_transform_kind=propagate_b,
    )
    arch = bitblas.base.CDNA(target=target)
    ref_sch = bitblas.gpu.MatmulTensorizationMFMA().apply_config(
        matmul_func,
        config=bitblas.base.Hint.from_dict({
            "arch": arch,
            "block": [128, 128],
            "warp": [64, 64],
            "rstep": [32],
            "chunk": [2],
            "block_reduction_depth": 16,
            "pipeline_stage": 2,
            "use_async": True,
            "intrin_info": intrin_info,
            "shared_scope": "shared",
            "vectorize": {
                "b": 8,
                "a": 8
            },
        }),
    )

    with tvm.transform.PassContext():
        ref_rt_mod = tvm.build(ref_sch.mod, target=target)

    ctx = tvm.rocm(0)
    np.random.seed(0)
    a_np = (np.random.rand(M, K)).astype("float16")
    print(a_np)
    b_np = (np.random.rand(N, K)).astype("float16")

    rocm_a = tvm.nd.array((a_np).astype("float16"), ctx)
    rocm_b = tvm.nd.array((b_np).astype("float16"), ctx)
    rocm_c = tvm.nd.array(np.zeros((M, N)).astype("float32"), ctx)

    ref_rt_mod(rocm_a, rocm_b, rocm_c)

    c_np = rocm_c.numpy()
    np.testing.assert_allclose(
        c_np, np.matmul(a_np.astype("float32"), b_np.astype("float32").T), rtol=1e-2, atol=1e-2
    )
    print(c_np)
    print(np.matmul(a_np.astype("float32"), b_np.astype("float32").T))


def test_assert_correctness_with_block_reduce():
    assert_correctness_with_block_reduce(
        M=256,
        N=256,
        K=256,
        in_dtype="float16",
        out_dtype="float32",
        accum_dtype="float32",
        propagate_a=0,
        propagate_b=0)

# fmt: on
if __name__ == "__main__":
    bitblas.testing.main()