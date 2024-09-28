# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas import tvm as tvm
from tvm import DataType
import tvm.tl.language as T

from bitblas.tl.utils import (
    get_mma_micro_size,
    make_swizzle_layout,
)

from bitblas.tl.macro_generator import (
    TensorCoreIntrinEmitter,
    TensorCoreIntrinEmitterWithLadderTransform,
)

from bitblas.quantization import (
    _tir_packed_int_to_int_convert,
    _tir_packed_to_signed_convert,
    _tir_packed_to_unsigned_convert,
    _tir_u32_to_f4_to_f16,
    _tir_u8_to_f8_e4m3_to_f16,
    _tir_packed_to_unsigned_convert_with_zeros,
)

from bitblas.ops.operator import TransformKind

# TODO(lei): Implement A General Matmul Emitter for Dequantize

def matmul_blocked_weight_only(
    M,
    N,
    K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    bit=4,
    storage_dtype="int8",
    source_format="uint",
    with_scaling=False,
    with_zeros=False,
    group_size=-1,
    fast_decoding=False,
    with_bias=False,
    zeros_mode="original",
    # Tile Related Params
    block_M=64,
    block_N=64,
    block_K=32,
    num_stages=2,
    threads=128,
    enable_rasterization=False,  # Enhance L2 Locality
):
    num_elems_per_byte = 8 // bit
    A_shape = (M, K)
    B_shape = (N, K // num_elems_per_byte)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K // num_elems_per_byte)
    B_dequantize_shared_shape = (block_N, block_K)

    import tvm.tl.language as T

    @T.prim_func
    def main(
            A: T.Buffer(A_shape, in_dtype),
            B: T.Buffer(B_shape, storage_dtype),
            C: T.Buffer((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
            B_local = T.alloc_fragment([8], storage_dtype, "local")
            B_dequantize_local = T.alloc_fragment([16], in_dtype, "local")
            B_dequantize_shared = T.alloc_shared(B_dequantize_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            tx = T.thread_binding(0, threads, thread="threadIdx.x")
            
            if enable_rasterization:
                # rasterization factor
                T.use_swizzle(10)
    
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)

                for i, j in T.Parallel(block_N, block_K // num_elems_per_byte):
                    B_shared[i, j] = B[bx * block_N + i, k * block_K // num_elems_per_byte + j]

                for i in T.serial(block_N * block_K // num_elems_per_byte // (threads * 4)):
                    for v in T.vectorized(0, 4):
                        vi = (i * threads * 4 + tx * 4 + v) // (block_K // num_elems_per_byte)
                        vj = (i * threads * 4 + tx * 4 + v) % (block_K // num_elems_per_byte)
                        B_local[v] = B_shared[vi, vj]
                    for v in T.serial(0, 8):
                        B_dequantize_local[v] = _tir_packed_to_unsigned_convert("int", 8)(
                            bit,
                            B_local[v // 2],
                            v % 2,
                            dtype=in_dtype,
                        )
                    for v in T.vectorized(0, 8):
                        vi = (i * threads * 8 + tx * 8 + v) // (block_K)
                        vj = (i * threads * 8 + tx * 8 + v) % (block_K)
                        B_dequantize_shared[vi, vj] = B_dequantize_local[v]

                T.gemm(A_shared, B_dequantize_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main