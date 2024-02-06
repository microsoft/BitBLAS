import tvm
import torch
import numpy as np
import tvm.testing
from tvm.script import tir as T
import os
from tvm import te

import numpy as np

def compress_int4_to_int8(int4_weight):
    if int4_weight.dtype == np.float16:
        int4_weight = int4_weight.astype(dtype=np.int8)
    int8_weight = np.zeros(
        (*int4_weight.shape[:-1], int4_weight.shape[-1] // 2), dtype=np.int8
    )
    for j in range(int4_weight.shape[-1] // 2):
        for k in range(2):
            int8_weight[:, j] |= int4_weight[:, j * 2 + k] << (4 * k)
    return int8_weight


def interleave_weight_int4(qweight):
    nbits = 4
    qweight = qweight.view(np.int32)
    new_qweight = np.zeros_like(qweight)
    bits_stride = 16
    mask = (1 << nbits) - 1  # for 4bit the val is 0x0000000f
    num_groups = 32 // bits_stride
    elems_per_group = bits_stride // nbits
    for i in range(num_groups):
        for j in range(elems_per_group):
            offset = i * elems_per_group + j
            shift = (offset % num_groups) * bits_stride + (offset // num_groups) * nbits
            new_qweight |= ((qweight >> (nbits * offset)) & mask) << shift

    return new_qweight.view(np.int8)


N = 2
K = 16
torch.manual_seed(0)
raw_data = torch.randint(0, 7, (N, K), dtype=torch.int8).cpu().numpy()
compressed_b = compress_int4_to_int8(raw_data)
interleaved_weight = interleave_weight_int4(compressed_b)

print(f"raw_data: \n", raw_data)
print(f"interleaved_weight: \n", interleaved_weight)


def tir_interleave_weight_int4_f16(N=2, K=16, bits=4):
    QK = K * bits // 32
    bits_stride = 16
    mask = (1 << bits) - 1  # for 4bit the val is 0x0000000f
    num_groups = 32 // bits_stride
    elems_per_group = bits_stride // bits

    @T.prim_func
    def main(A: T.Buffer((N, QK), "int32"), B: T.Buffer((N, QK), "int32")):
        for ax0, ax1, ax2, ax3 in T.grid(N, QK, num_groups, elems_per_group):
            with T.block("B"):
                v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                offset = v2 * elems_per_group + v3
                shift = (offset % num_groups) * bits_stride + (offset // num_groups) * bits
                B[v0, v1] = B[v0, v1] | (((A[v0, v1] >> (bits * offset)) & mask) << shift)

    return main

interleave_func = tir_interleave_weight_int4_f16()

ref_func = tvm.build(interleave_func, target="llvm")
ctx = tvm.cpu(0)
compressed_b_cast_32 = compressed_b.view(np.int32)
print("compressed_b_cast_32: \n", compressed_b_cast_32)
tvm_compress_b = tvm.nd.array(compressed_b_cast_32, ctx)
tvm_interleaved_b = tvm.nd.array(np.zeros_like(compressed_b_cast_32), ctx)
ref_func(tvm_compress_b, tvm_interleaved_b)
tvm_interleaved_b_np = tvm_interleaved_b.asnumpy()
tvm_interleaved_b_np_int8 = tvm_interleaved_b_np.view(np.int8)
print("tvm_interleaved_b_np_int8: \n", tvm_interleaved_b_np_int8)
