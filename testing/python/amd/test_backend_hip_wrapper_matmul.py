# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import bitblas
from bitblas import MatmulConfig, Matmul
import logging
from bitblas import set_log_level
from bitblas.builder.wrapper import TIRWrapper
import tvm

set_log_level(logging.DEBUG)


def get_codegen_result(ops):
    code = ops.get_source()
    return code


def matmul_backend_code_wrap(
    M,
    N,
    K,
    A_dtype,
    W_dtype,
    accum_dtype,
    out_dtype,
    with_bias,
):
    import torch
    torch.random.manual_seed(0)

    matmul_config = MatmulConfig(
        M=M,
        N=N,
        K=K,
        A_dtype=A_dtype,
        W_dtype=W_dtype,
        accum_dtype=accum_dtype,
        out_dtype=out_dtype,
        with_bias=with_bias,
    )
    target = tvm.target.Target("hip")
    matmul = Matmul(config=matmul_config, target=target, enable_tuning=False)
    backend = TIRWrapper(arch=matmul.arch)
    backend.assign_optimized_module(matmul.scheduled_ir_module)
    is_dynamic = (
        matmul.dynamic_range is not None and len(matmul.scheduled_ir_module.functions) > 1)
    wrapped_code = backend.wrap(matmul.get_source(kenrel_only=True), is_dynamic=is_dynamic)
    assert "void call" in wrapped_code

@tvm.testing.requires_rocm
def test_matmul_transform_weight():
    matmul_backend_code_wrap(128, 128, 128, "float16", "float16", "float16", "float16", False)
    matmul_backend_code_wrap(1, 256, 256, "float16", "float16", "uint4", "float16", False)


# fmt: on
if __name__ == "__main__":
    bitblas.testing.main()
