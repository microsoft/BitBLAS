# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas.ops.impl.matmul_dequantize_impl import (
    MatMulNTDequantizeEmitter,
    matmul_nt_dequantize_b,
    matmul_nt_dequantize_b_propagate_b,
    matmul_nt_dequantize_b_propagate_a_propagate_b,
)
from bitblas import tvm
import logging
from bitblas import set_log_level

set_log_level(logging.DEBUG)

def compare_tir_scripts_and_emitter(
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
):
    tir_script_func = matmul_nt_dequantize_b(
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
    
    emitter_func = MatMulNTDequantizeEmitter(
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
    ).emit()
    
    tvm.ir.assert_structural_equal(tir_script_func, emitter_func)
