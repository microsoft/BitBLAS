# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import bitblas
from bitblas import tvm
import torch
import numpy as np
from tvm.script import tir as T


def general_compress_to_int8(lowprecision_weight, source_bits=4):
    elems_per_byte = 8 // source_bits
    if lowprecision_weight.dtype == np.float16:
        lowprecision_weight = lowprecision_weight.astype(dtype=np.int8)
    int8_weight = np.zeros(
        (
            *lowprecision_weight.shape[:-1],
            lowprecision_weight.shape[-1] // elems_per_byte,
        ),
        dtype=np.int8,
    )
    for j in range(lowprecision_weight.shape[-1] // elems_per_byte):
        for k in range(elems_per_byte):
            int8_weight[:, j] |= lowprecision_weight[:, j * elems_per_byte + k] << (source_bits * k)
    return int8_weight


def interleave_weight(qweight, nbits=4, target_dtype="float16"):
    assert target_dtype in ["float16", "int8"]
    # reinterpret the data type of qweight to int32
    qweight = qweight.view(np.int32)
    new_qweight = np.zeros_like(qweight)
    bits_stride = 8 if target_dtype == "int8" else 16
    mask = (1 << nbits) - 1  # for 4bit the val is 0x0000000f
    num_groups = 32 // bits_stride
    elems_per_group = bits_stride // nbits
    for i in range(num_groups):
        for j in range(elems_per_group):
            offset = i * elems_per_group + j
            shift = (offset % num_groups) * bits_stride + (offset // num_groups) * nbits
            new_qweight |= ((qweight >> (nbits * offset)) & mask) << shift

    if nbits == 1 and target_dtype == "int8":
        # special handling for 1b interleave
        n16_weight = new_qweight & np.int32(np.uint32(0xF0F00F0F))
        n16_weight |= ((new_qweight & np.int32(np.uint32(0x000000F0))) >> 4) << 16
        n16_weight |= ((new_qweight & np.int32(np.uint32(0x0000F000))) >> 12) << 24
        n16_weight |= ((new_qweight & np.int32(np.uint32(0x000F0000))) >> 16) << 4
        n16_weight |= ((new_qweight & np.int32(np.uint32(0x0F000000))) >> 24) << 12
        return n16_weight.view(np.int8)
    elif nbits == 2 and target_dtype == "float16":
        n8_weight = new_qweight & np.int32(np.uint32(0xFF0000FF))
        n8_weight |= ((new_qweight & np.int32(np.uint32(0x0000FF00))) >> 8) << 16
        n8_weight |= ((new_qweight & np.int32(np.uint32(0x00FF0000))) >> 16) << 8
        return n8_weight.view(np.int8)
    elif nbits == 1 and target_dtype == "float16":
        n8_weight = new_qweight & np.int32(np.uint32(0xF000000F))
        n8_weight |= ((new_qweight & np.int32(np.uint32(0x000000F0))) >> 4) << 8
        n8_weight |= ((new_qweight & np.int32(np.uint32(0x00000F00))) >> 8) << 16
        n8_weight |= ((new_qweight & np.int32(np.uint32(0x0000F000))) >> 12) << 24
        n8_weight |= ((new_qweight & np.int32(np.uint32(0x000F0000))) >> 16) << 4
        n8_weight |= ((new_qweight & np.int32(np.uint32(0x00F00000))) >> 20) << 12
        n8_weight |= ((new_qweight & np.int32(np.uint32(0x0F000000))) >> 24) << 20

    return new_qweight.view(np.int8)


def tir_interleave_weight(N=2, K=16, bits=4, target_dtype="float16"):
    QK = K * bits // 32
    bits_stride = 16
    mask = (1 << bits) - 1  # for 4bit the val is 0x0000000f
    num_groups = 32 // bits_stride
    elems_per_group = bits_stride // bits

    @T.prim_func
    def interleave_weight(A: T.Buffer((N, QK), "int32"), B: T.Buffer((N, QK), "int32")):
        for ax0, ax1, ax2, ax3 in T.grid(N, QK, num_groups, elems_per_group):
            with T.block("B"):
                v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                offset = v2 * elems_per_group + v3
                shift = (offset % num_groups) * bits_stride + (offset // num_groups) * bits
                B[v0, v1] = B[v0, v1] | (((A[v0, v1] >> (bits * offset)) & mask) << shift)

    @T.prim_func
    def interleave_weight_f16_2b(A: T.Buffer((N, QK), "int32"), B: T.Buffer((N, QK), "int32")):
        B_tmp_1 = T.alloc_buffer((N, QK), "int32", scope="local")
        B_tmp_2 = T.alloc_buffer((N, QK), "int32", scope="local")
        B_tmp_3 = T.alloc_buffer((N, QK), "int32", scope="local")
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
    def interleave_weight_f16_1b(A: T.Buffer((N, QK), "int32"), B: T.Buffer((N, QK), "int32")):
        B_tmp_1 = T.alloc_buffer((N, QK), "int32", scope="local")
        B_tmp_2 = T.alloc_buffer((N, QK), "int32", scope="local")
        B_tmp_3 = T.alloc_buffer((N, QK), "int32", scope="local")
        B_tmp_4 = T.alloc_buffer((N, QK), "int32", scope="local")
        B_tmp_5 = T.alloc_buffer((N, QK), "int32", scope="local")
        B_tmp_6 = T.alloc_buffer((N, QK), "int32", scope="local")
        B_tmp_7 = T.alloc_buffer((N, QK), "int32", scope="local")
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
    def interleave_weight_int8_1b(A: T.Buffer((N, QK), "int32"), B: T.Buffer((N, QK), "int32")):
        B_tmp_1 = T.alloc_buffer((N, QK), "int32", scope="local")
        B_tmp_2 = T.alloc_buffer((N, QK), "int32", scope="local")
        B_tmp_3 = T.alloc_buffer((N, QK), "int32", scope="local")
        B_tmp_4 = T.alloc_buffer((N, QK), "int32", scope="local")
        B_tmp_5 = T.alloc_buffer((N, QK), "int32", scope="local")
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


def test_lop3_interleave_weight():
    source_nbits = 2
    N = 2
    K = 16
    target_dtype = "float16"
    torch.manual_seed(0)
    uint_max = 2**(source_nbits) - 1
    raw_data = torch.randint(0, uint_max, (N, K), dtype=torch.int8).cpu().numpy()
    compressed_b = general_compress_to_int8(raw_data, source_nbits)
    interleaved_weight = interleave_weight(compressed_b, source_nbits, target_dtype)
    interleave_func = tir_interleave_weight(N, K, source_nbits, target_dtype)

    ref_func = tvm.build(interleave_func, target="llvm")
    ctx = tvm.cpu(0)
    compressed_b_cast_32 = compressed_b.view(np.int32)
    tvm_compress_b = tvm.nd.array(compressed_b_cast_32, ctx)
    tvm_interleaved_b = tvm.nd.array(np.zeros_like(compressed_b_cast_32), ctx)
    ref_func(tvm_compress_b, tvm_interleaved_b)
    tvm_interleaved_b_np = tvm_interleaved_b.asnumpy()
    tvm_interleaved_b_np_int8 = tvm_interleaved_b_np.view(np.int8)
    np.testing.assert_allclose(tvm_interleaved_b_np_int8, interleaved_weight, atol=1e-5)


if __name__ == "__main__":
    bitblas.testing.main()
