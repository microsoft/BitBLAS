# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm
from tvm.script import tir as T
from tvm.tir import IndexMap
from tvm.tir.tensor_intrin.cuda import (
    ldmatrix_trans_32x8_to_shared_16x16_layout,
    ldmatrix_32x16_to_shared_16x32_layout_a,
    ldmatrix_32x16_to_shared_16x32_layout_b,
)

def ldmatrix_trans_permutation_16x16_32x8_16x16(kernel_i, kernel_j):
    thread_id = kernel_i * 2 + kernel_j // 8
    local_id = kernel_j % 8
    return ldmatrix_trans_32x8_to_shared_16x16_layout(thread_id, local_id)

@tvm.script.ir_module
class LDMATRIX_16x16:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [16, 16], dtype="float16")
        B = T.match_buffer(b, [16, 16], dtype="float16")
        
        for i, j in T.grid(16, 16):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(B[vi, vj])
                T.writes(A[vi, vj])
                A[vi, vj] = B[vi, vj]

ir_module = LDMATRIX_16x16
sch = tvm.tir.Schedule(ir_module)

block_b = sch.get_block("B")
sch.transform_layout(block_b, ('read', 0), ldmatrix_trans_permutation_16x16_32x8_16x16)
print("========================inject transform=============================")
print(sch.mod["main"].script())

index_map = IndexMap.from_func(ldmatrix_trans_permutation_16x16_32x8_16x16, index_dtype="int32")
inversed_index_map = index_map.inverse([16, 16])
def inverse_permutation(i, j):
    return inversed_index_map.map_indices([i, j])
sch.transform_layout(block_b, ('read', 0), inverse_permutation)
print("========================inverse inject transform=============================")
print(sch.mod["main"].script())


def ldmatrix_trans_permutation_16x32_16x32_16x32(kernel_i, kernel_j):
    thread_id = kernel_i * 2 + kernel_j // 16
    local_id = kernel_j % 16
    return ldmatrix_32x16_to_shared_16x32_layout_a(thread_id, local_id)

@tvm.script.ir_module
class LDMATRIX_16x32_A:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [16, 32], dtype="float16")
        B = T.match_buffer(b, [16, 32], dtype="float16")
        
        for i, j in T.grid(16, 32):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(B[vi, vj])
                T.writes(A[vi, vj])
                A[vi, vj] = B[vi, vj]

ir_module = LDMATRIX_16x32_A
sch = tvm.tir.Schedule(ir_module)

block_b = sch.get_block("B")
sch.transform_layout(block_b, ('read', 0), ldmatrix_trans_permutation_16x32_16x32_16x32)
print("========================inject transform=============================")
print(sch.mod["main"].script())

index_map = IndexMap.from_func(ldmatrix_trans_permutation_16x32_16x32_16x32, index_dtype="int32")
inversed_index_map = index_map.inverse([16, 32])
def inverse_permutation(i, j):
    return inversed_index_map.map_indices([i, j])
sch.transform_layout(block_b, ('read', 0), inverse_permutation)
print("========================inverse inject transform=============================")
print(sch.mod["main"].script())

def ldmatrix_trans_permutation_16x32_16x32_16x32(kernel_i, kernel_j):
    thread_id = kernel_i * 2 + kernel_j // 16
    local_id = kernel_j % 16
    return ldmatrix_32x16_to_shared_16x32_layout_b(thread_id, local_id)

@tvm.script.ir_module
class LDMATRIX_16x32_B:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [16, 32], dtype="float16")
        B = T.match_buffer(b, [16, 32], dtype="float16")
        
        for i, j in T.grid(16, 32):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(B[vi, vj])
                T.writes(A[vi, vj])
                A[vi, vj] = B[vi, vj]

ir_module = LDMATRIX_16x32_B
sch = tvm.tir.Schedule(ir_module)

block_b = sch.get_block("B")
sch.transform_layout(block_b, ('read', 0), ldmatrix_trans_permutation_16x32_16x32_16x32)
print("========================inject transform=============================")
print(sch.mod["main"].script())

index_map = IndexMap.from_func(ldmatrix_trans_permutation_16x32_16x32_16x32, index_dtype="int32")
inversed_index_map = index_map.inverse([16, 32])
def inverse_permutation(i, j):
    return inversed_index_map.map_indices([i, j])
sch.transform_layout(block_b, ('read', 0), inverse_permutation)
print("========================inverse inject transform=============================")
print(sch.mod["main"].script())
