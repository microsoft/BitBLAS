# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm
import logging
from bitblas import set_log_level

set_log_level(logging.DEBUG)


def check_eual_ref_scripts_with_emitter(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    bit,
    storage_dtype,
    source_format,
    with_scaling,
    with_zeros,
    group_size,
    fast_decoding,
    with_bias,
    zeros_mode,
    propagate_a,
    propagate_b,
):
    from bitblas.ops.impl.matmul_dequantize_impl import (
        MatMulNTDequantizeEmitter,
        matmul_nt_dequantize_b,
        matmul_nt_dequantize_b_propagate_b,
        matmul_nt_dequantize_b_propagate_a_propagate_b,
    )
    func = None
    if propagate_a and propagate_b:
        func = matmul_nt_dequantize_b_propagate_a_propagate_b
    elif propagate_b:
        func = matmul_nt_dequantize_b_propagate_b
    else:
        func = matmul_nt_dequantize_b

    assert func is not None, "No function found for the given configuration"

    ref_func = func(
        M,
        N,
        K,
        in_dtype,
        out_dtype,
        accum_dtype,
        bit,
        storage_dtype,
        source_format,
        with_scaling,
        with_zeros,
        group_size,
        fast_decoding,
        with_bias,
        zeros_mode,
    )

    emit_func = MatMulNTDequantizeEmitter(
        M,
        N,
        K,
        in_dtype,
        out_dtype,
        accum_dtype,
        bit,
        storage_dtype,
        source_format,
        with_scaling,
        with_zeros,
        group_size,
        fast_decoding,
        with_bias,
        zeros_mode,
        propagate_a=propagate_a,
        propagate_b=propagate_b,
    ).emit()

    tvm.ir.assert_structural_equal(ref_func, emit_func)


def test_check_eual_ref_scripts_with_emitter():
    check_eual_ref_scripts_with_emitter(1, 16384, 16384, "float16", "float16", "float16", 4, "int8", "nf", True, False, -1, False, False, "original", False, False)
    check_eual_ref_scripts_with_emitter(16384, 16384, 16384, "float16", "float16", "float16", 4, "int8", "nf", True, False, -1, False, False, "original", False, False)
    check_eual_ref_scripts_with_emitter(1, 16384, 16384, "float16", "float16", "float16", 4, "int8", "uint", True, False, -1, False, False, "original", False, False)
    check_eual_ref_scripts_with_emitter(1, 16384, 16384, "float16", "float16", "float16", 4, "int8", "uint", True, False, -1, False, False, "original", False, False)
    check_eual_ref_scripts_with_emitter(1, 16384, 16384, "float16", "float16", "float16", 4, "int8", "uint", True, False, -1, False, False, "original", False, True)
    check_eual_ref_scripts_with_emitter(1, 16384, 16384, "float16", "float16", "float16", 4, "int8", "uint", True, False, -1, False, False, "original", False, True)
    check_eual_ref_scripts_with_emitter(1024, 1024, 1024, "float16", "float16", "float16", 4, "int8", "uint", True, False, -1, False, False, "original", True, True)

if __name__ == "__main__":
    test_check_eual_ref_scripts_with_emitter()
