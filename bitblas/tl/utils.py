# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm as tvm
from bitblas import tilelang as tilelang
from tvm import arith
from tvm import DataType
import tilelang.language as T
from typing import Union, Literal
from .mma_layout import (
    ldmatrix_32x8_to_shared_16x16_layout,
    ldmatrix_trans_32x8_to_shared_16x16_layout,
    ldmatrix_32x16_to_shared_16x32_layout_a,
    ldmatrix_32x16_to_shared_16x32_layout_b,
    mma_store_32x8_to_shared_16x16_layout,
)


def get_swizzle_layout(row_idx, col_idx, row_size, dtype: Union[DataType, str]):
    ana = arith.Analyzer()
    BANK_SIZE_BYTES = 128
    if isinstance(dtype, str):
        dtype = DataType(dtype)
    col_idx_outer, col_idx_inner = col_idx // (BANK_SIZE_BYTES // dtype.bits), col_idx % (
        BANK_SIZE_BYTES // dtype.bits)
    #  use transaction bits to support diverse dtype.
    #  for fp16, 64 elems * 16 bits = 1024 bits, 32 elems * 32 bits = 512 bits
    #  for int8, 128 elems * 8 bits = 1024 bits, 64 elems * 8 bits = 512 bits
    coalescent_bits = dtype.bits * row_size
    # permutation on 4 banks, each bank has 32 bits
    bank_elems = BANK_SIZE_BYTES // dtype.bits
    new_col_idx_outer = None

    if coalescent_bits % 1024 == 0:
        #   Use 8 * 8 permuted layout
        #   Every number below corresponds to 8 consecutive fp16 number in shared mem, i.e. one read
        #   Every row below corresponds to 32 banks
        #   0  1  2  3  4  5  6  7    ==>    0  1  2  3  4  5  6  7
        #   0  1  2  3  4  5  6  7    ==>    1  0  3  2  5  4  7  6
        #   0  1  2  3  4  5  6  7    ==>    2  3  0  1  6  7  4  5
        #   0  1  2  3  4  5  6  7    ==>    3  2  1  0  7  6  5  4
        #   0  1  2  3  4  5  6  7    ==>    4  5  6  7  0  1  2  3
        #   0  1  2  3  4  5  6  7    ==>    5  4  7  6  1  0  3  2
        #   0  1  2  3  4  5  6  7    ==>    6  7  4  5  2  3  0  1
        #   0  1  2  3  4  5  6  7    ==>    7  6  5  4  3  2  1  0
        row_idx_sub = row_idx % bank_elems
        new_col_idx_outer = col_idx_outer ^ row_idx_sub
    else:
        assert coalescent_bits % 512 == 0
        #  Use 8 * 4 permuted layout
        #  Every number below corresponds to 8 consecutive fp16 number in shared mem, i.e. one read
        #  Every row below corresponds to 16 banks
        #  0  1  2  3    ==>    0  1  2  3
        #  0  1  2  3    ==>    0  1  2  3
        #  0  1  2  3    ==>    1  0  3  2
        #  0  1  2  3    ==>    1  0  3  2
        #  0  1  2  3    ==>    2  3  0  1
        #  0  1  2  3    ==>    2  3  0  1
        #  0  1  2  3    ==>    3  2  1  0
        #  0  1  2  3    ==>    3  2  1  0
        #  View with 8 elements per row:
        #  0  1  2  3  4  0  1  2  3    ==>    0  1  2  3  0  1  2  3
        #  0  1  2  3  4  0  1  2  3    ==>    1  0  3  2  1  0  3  2
        #  0  1  2  3  4  0  1  2  3    ==>    2  3  0  1  2  3  0  1
        #  0  1  2  3  4  0  1  2  3    ==>    3  2  1  0  3  2  1  0
        row_idx_sub = row_idx % bank_elems
        #  Interleave elems per byte
        interleave_elems = 32 // dtype.bits
        new_col_idx_outer = col_idx_outer ^ (row_idx_sub // interleave_elems)

    assert (new_col_idx_outer is not None), f"Unsupported dtype {dtype} with {coalescent_bits} bits"
    return row_idx, ana.simplify(new_col_idx_outer * bank_elems + col_idx_inner)


def get_ldmatrix_offset(
    matrix: Literal["A", "B"],
    row_idx,
    col_idx,
    stride,
    dtype: Literal["float16", "int8"] = "float16",
    transpose: bool = False,
):
    assert matrix in ["A", "B"], "matrix should be either A or B"
    transform_func = (
        ldmatrix_32x8_to_shared_16x16_layout
        if dtype in ["float16", "bfloat16"] else ldmatrix_32x16_to_shared_16x32_layout_b)
    transform_func_trans = (
        ldmatrix_trans_32x8_to_shared_16x16_layout
        if dtype in ["float16", "bfloat16"] else ldmatrix_32x16_to_shared_16x32_layout_a)
    if matrix == "A":
        assert not transpose, "A matrix should not be transposed"
        new_row_idx, new_col_idx = transform_func(row_idx, col_idx)
        return new_row_idx * stride + new_col_idx
    else:
        new_row_idx, new_col_idx = transform_func_trans(row_idx, col_idx)
        return new_row_idx * stride + new_col_idx


def mma_store_index_map(*args, **kwargs):
    return mma_store_32x8_to_shared_16x16_layout(*args, **kwargs)


def get_mma_micro_size(dtype: Literal["float16", "int8"]):
    # TODO(lei): FP8 related precision support.
    # Basic Tensor Core Matrix Multiply operation Unit
    micro_size_x = micro_size_y = 16
    micro_size_k = 16
    if dtype == "int8":
        micro_size_k = 32
    return micro_size_x, micro_size_y, micro_size_k


def make_swizzle_layout(shared_buf):
    dtype = shared_buf.dtype
    shape = shared_buf.shape

    can_swizzle = shape[-1] * DataType(dtype).bits == 512
    if not can_swizzle:
        return T.Layout(shape, lambda *args: args)

    def transform_func(i, j):
        new_warp_i, new_warp_j = get_swizzle_layout(i, j, shape[-1], dtype)
        return [new_warp_i, new_warp_j]

    return T.Layout(shape, transform_func)
