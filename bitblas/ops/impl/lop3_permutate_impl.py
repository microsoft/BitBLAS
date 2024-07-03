# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Literal
from tvm import DataType
from tvm import IRModule
from tvm.ir import GlobalVar
from tvm.script import tir as T


# fmt: off
# TIR interleave weight impl-> 2D implementation
def tir_interleave_weight(
    N: int = 2,
    K: int = 16,
    bits: int = 4,
    QK: int = -1,
    target_dtype: str = "float16",
    storage_dtype: str = "int32",
):
    if QK == -1:
        QK = K * bits // 32
    bits_stride = DataType(target_dtype).bits
    mask = (1 << bits) - 1  # for 4bit the val is 0x0000000f
    num_groups = 32 // bits_stride
    elems_per_group = bits_stride // bits

    @T.prim_func
    def interleave_weight(A: T.Buffer((N, QK), storage_dtype), B: T.Buffer((N, QK), storage_dtype)):
        for ax0, ax1, ax2, ax3 in T.grid(N, QK, num_groups, elems_per_group):
            with T.block("B"):
                v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                offset = v2 * elems_per_group + v3
                shift = (offset % num_groups) * bits_stride + (offset // num_groups) * bits
                B[v0, v1] = B[v0, v1] | (((A[v0, v1] >> (bits * offset)) & mask) << shift)

    @T.prim_func
    def interleave_weight_f16_2b(A: T.Buffer((N, QK), storage_dtype), B: T.Buffer((N, QK),
                                                                                  storage_dtype)):
        B_tmp_1 = T.alloc_buffer((N, QK), storage_dtype, scope="local")
        B_tmp_2 = T.alloc_buffer((N, QK), storage_dtype, scope="local")
        B_tmp_3 = T.alloc_buffer((N, QK), storage_dtype, scope="local")
        for ax0, ax1, ax2, ax3 in T.grid(N, QK, num_groups, elems_per_group):
            with T.block("B_tmp"):
                v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                offset = v2 * elems_per_group + v3
                shift = (offset % num_groups) * bits_stride + (offset // num_groups) * bits
                B[v0, v1] = B[v0, v1] | (((A[v0, v1] >> (bits * offset)) & mask) << shift)

        for ax0, ax1 in T.grid(N, QK):
            with T.block("B"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                B_tmp_1[v0, v1] = B[v0, v1] & T.uint32(0xFF0000FF)
                B_tmp_2[v0, v1] = ((B[v0, v1] & T.uint32(0x00FF0000)) << 8) >> 16
                B_tmp_3[v0, v1] = ((B[v0, v1] & T.uint32(0x0000FF00)) << 16) >> 8
                B[v0, v1] = B_tmp_1[v0, v1] | B_tmp_2[v0, v1] | B_tmp_3[v0, v1]

    @T.prim_func
    def interleave_weight_f16_1b(A: T.Buffer((N, QK), storage_dtype), B: T.Buffer((N, QK),
                                                                                  storage_dtype)):
        B_tmp_1 = T.alloc_buffer((N, QK), storage_dtype, scope="local")
        B_tmp_2 = T.alloc_buffer((N, QK), storage_dtype, scope="local")
        B_tmp_3 = T.alloc_buffer((N, QK), storage_dtype, scope="local")
        B_tmp_4 = T.alloc_buffer((N, QK), storage_dtype, scope="local")
        B_tmp_5 = T.alloc_buffer((N, QK), storage_dtype, scope="local")
        B_tmp_6 = T.alloc_buffer((N, QK), storage_dtype, scope="local")
        B_tmp_7 = T.alloc_buffer((N, QK), storage_dtype, scope="local")
        for ax0, ax1, ax2, ax3 in T.grid(N, QK, num_groups, elems_per_group):
            with T.block("B_tmp"):
                v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                offset = v2 * elems_per_group + v3
                shift = (offset % num_groups) * bits_stride + (offset // num_groups) * bits
                B[v0, v1] = B[v0, v1] | (((A[v0, v1] >> (bits * offset)) & mask) << shift)

        for ax0, ax1 in T.grid(N, QK):
            with T.block("B"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                B_tmp_1[v0, v1] = B[v0, v1] & T.uint32(0xF000000F)
                B_tmp_2[v0, v1] = ((B[v0, v1] & T.uint32(0x000000F0)) >> 4) << 8
                B_tmp_3[v0, v1] = ((B[v0, v1] & T.uint32(0x00000F00)) >> 8) << 16
                B_tmp_4[v0, v1] = ((B[v0, v1] & T.uint32(0x0000F000)) >> 12) << 24
                B_tmp_5[v0, v1] = ((B[v0, v1] & T.uint32(0x000F0000)) >> 16) << 8
                B_tmp_6[v0, v1] = ((B[v0, v1] & T.uint32(0x00F00000)) >> 20) << 12
                B_tmp_7[v0, v1] = ((B[v0, v1] & T.uint32(0x00F00000)) >> 24) << 20
                B[v0, v1] = (
                    B_tmp_1[v0, v1]
                    | B_tmp_2[v0, v1]
                    | B_tmp_3[v0, v1]
                    | B_tmp_4[v0, v1]
                    | B_tmp_5[v0, v1]
                    | B_tmp_6[v0, v1]
                    | B_tmp_7[v0, v1])

    @T.prim_func
    def interleave_weight_int8_1b(A: T.Buffer((N, QK), storage_dtype), B: T.Buffer((N, QK),
                                                                                   storage_dtype)):
        B_tmp_1 = T.alloc_buffer((N, QK), storage_dtype, scope="local")
        B_tmp_2 = T.alloc_buffer((N, QK), storage_dtype, scope="local")
        B_tmp_3 = T.alloc_buffer((N, QK), storage_dtype, scope="local")
        B_tmp_4 = T.alloc_buffer((N, QK), storage_dtype, scope="local")
        B_tmp_5 = T.alloc_buffer((N, QK), storage_dtype, scope="local")
        for ax0, ax1, ax2, ax3 in T.grid(N, QK, num_groups, elems_per_group):
            with T.block("B_tmp"):
                v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                offset = v2 * elems_per_group + v3
                shift = (offset % num_groups) * bits_stride + (offset // num_groups) * bits
                B[v0, v1] = B[v0, v1] | (((A[v0, v1] >> (bits * offset)) & mask) << shift)

        for ax0, ax1 in T.grid(N, QK):
            with T.block("B"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                B_tmp_1[v0, v1] = B[v0, v1] & T.uint32(0xF0F00F0F)
                B_tmp_2[v0, v1] = ((B[v0, v1] & T.uint32(0x000000F0)) >> 4) << 16
                B_tmp_3[v0, v1] = ((B[v0, v1] & T.uint32(0x0000F000)) >> 12) << 24
                B_tmp_4[v0, v1] = ((B[v0, v1] & T.uint32(0x000F0000)) >> 16) << 4
                B_tmp_5[v0, v1] = ((B[v0, v1] & T.uint32(0x0F000000)) >> 24) << 12
                B[v0, v1] = (
                    B_tmp_1[v0, v1]
                    | B_tmp_2[v0, v1]
                    | B_tmp_3[v0, v1]
                    | B_tmp_4[v0, v1]
                    | B_tmp_5[v0, v1])

    if target_dtype == "float16" and bits == 2:
        return interleave_weight_f16_2b
    elif target_dtype == "float16" and bits == 1:
        return interleave_weight_f16_1b
    elif target_dtype == "int8" and bits == 1:
        return interleave_weight_int8_1b

    return interleave_weight


# fmt: on


def select_implementation(
    M: int,
    N: int,
    datatype: Literal["float16", "int8"] = "float16",
    storage_dtype: Literal["int8", "uint8", "int32", "uint32"] = "int32",
    dequantize_bits: int = 4,
):
    func = tir_interleave_weight(
        N=M,
        K=N,
        bits=dequantize_bits,
        target_dtype=datatype,
        storage_dtype=storage_dtype,
    )
    mod = IRModule()
    mod.update_func(GlobalVar("main"), func)
    return mod
