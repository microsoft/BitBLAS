# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import bitblas
from bitblas import MatmulConfig, Matmul
import logging
from bitblas import set_log_level

set_log_level(logging.DEBUG)


def get_codegen_result(ops):
    code = ops.get_source()
    return code


# fmt: off
def matmul_codegen_default(M, N, K, A_dtype, W_dtype, accum_dtype, out_dtype, layout, with_bias,
                           group_size, with_scaling, with_zeros, zeros_mode):

    matmul_config = MatmulConfig(
        M=M,
        N=N,
        K=K,
        A_dtype=A_dtype,
        W_dtype=W_dtype,
        accum_dtype=accum_dtype,
        out_dtype=out_dtype,
        layout=layout,
        with_bias=with_bias,
        group_size=group_size,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        zeros_mode=zeros_mode,
        propagate_a=False,
        propagate_b=False,
    )
    matmul = Matmul(config=matmul_config, enable_tuning=False, backend="tl")
    assert get_codegen_result(matmul)


def matmul_finetune(M,
                    N,
                    K,
                    A_dtype,
                    W_dtype,
                    accum_dtype,
                    out_dtype,
                    layout,
                    with_bias,
                    group_size,
                    with_scaling,
                    with_zeros,
                    zeros_mode,
                    propagate_b=False):

    matmul_config = MatmulConfig(
        M=M,
        N=N,
        K=K,
        A_dtype=A_dtype,
        W_dtype=W_dtype,
        accum_dtype=accum_dtype,
        out_dtype=out_dtype,
        layout=layout,
        with_bias=with_bias,
        group_size=group_size,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        zeros_mode=zeros_mode,
        propagate_a=False,
        propagate_b=propagate_b,
    )
    matmul = Matmul(config=matmul_config, enable_tuning=False, backend="tl")
    matmul.hardware_aware_finetune(topk=20)
    assert get_codegen_result(matmul)


def test_matmul_codegen_default():
    matmul_codegen_default(1, 768, 768, "float16", "float16", "float16", "float16", "nt", False, -1,
                           False, False, None),
    matmul_codegen_default(768, 768, 768, "float16", "float16", "float16", "float16", "nt", False,
                           -1, False, False, None),
    # FP32 Accum
    matmul_codegen_default(768, 768, 768, "float16", "float16", "float32", "float16", "nt", False,
                           -1, False, False, None),
    # INT32 Accum
    matmul_codegen_default(768, 768, 768, "int8", "int8", "int32", "int8", "nt", False, -1, False,
                           False, None),


def test_matmul_finetune():
    matmul_finetune(1024, 1024, 1024, "float16", "float16", "float16", "float16", "nt", False, -1,
                    False, False, None, False)
    matmul_finetune(1024, 1024, 1024, "float16", "float16", "float16", "float16", "nt", False, -1,
                    False, False, None, False)


# fmt: on
if __name__ == "__main__":
    bitblas.testing.main()
