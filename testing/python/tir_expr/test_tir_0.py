# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tvm
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.tir.tensor_intrin.cuda import get_mma_intrin_group

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 512, 16, 32), "int8"), B: T.Buffer((1024, 512, 16, 8), "int8"), C: T.Buffer((16384, 16384), "int32")):
        T.func_attr({"dequantize_info": {"B": {"decode_block": "B_decode", "fast_decoding": T.bool(False), "source_format": {"bits": 2, "format": "int"}, "target_format": "int8"}}, "dlight.tensorcore_prenormlized": T.bool(True), "input_transform_kind": T.bool(True), "weight_transform_kind": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_reindex_reindex_shared = T.alloc_buffer((1, 1024, 512, 16, 32), "int8", scope="shared")
        B_reindex_reindex_shared = T.alloc_buffer((1, 1024, 512, 16, 32), "int8", scope="shared")
        B_reindex_reindex_local = T.alloc_buffer((1, 1024, 512, 16, 32), "int8", scope="local")
        B_local = T.alloc_buffer((1024, 512, 16, 8), "int8", scope="local")
        B_shared = T.alloc_buffer((1024, 512, 16, 8), "int8", scope="shared")
        A_reindex_reindex_shared_warp = T.alloc_buffer((1, 1024, 512, 32, 16), "int8", scope="warp")
        B_reindex_reindex_shared_warp = T.alloc_buffer((1, 1024, 512, 32, 16), "int8", scope="warp")
        C_reindex_shared = T.alloc_buffer((1, 1024, 1024, 16, 16), "int32", scope="shared")
        C_reindex_shared_warp = T.alloc_buffer((1, 1024, 1024, 32, 8), "int32", scope="warp")
        for ax0 in range(1):
            for ax1_0_0_ax2_0_0_fused in T.thread_binding(64, thread="blockIdx.y"):
                for ax1_0_1_ax2_0_1_fused in T.thread_binding(256, thread="blockIdx.x"):
                    for ax1_0_2 in T.thread_binding(2, thread="threadIdx.y"):
                        for ax2_0_2 in T.thread_binding(2, thread="threadIdx.z"):
                            for ax1_0_3_init, ax2_0_3_init in T.grid(8, 2):
                                with T.block("C_o_init"):
                                    v0_o = T.axis.spatial(1, ax0)
                                    v1_o = T.axis.spatial(1024, ax1_0_0_ax2_0_0_fused * 16 + ax1_0_2 * 8 + ax1_0_3_init)
                                    v2_o = T.axis.spatial(1024, ax1_0_1_ax2_0_1_fused * 4 + ax2_0_2 * 2 + ax2_0_3_init)
                                    T.reads()
                                    T.writes(C_reindex_shared_warp[0, v1_o, v2_o, 0:32, 0:8])
                                    with T.block("C_init_o"):
                                        v1_i_init_o = T.axis.spatial(1, 0)
                                        v2_i_init_o = T.axis.spatial(1, 0)
                                        T.reads()
                                        T.writes(C_reindex_shared_warp[0, v1_o, v2_o, 0:32, 0:8])
                                        C_warp = T.match_buffer(C_reindex_shared_warp[0, v1_o, v2_o, 0:32, 0:8], (32, 8), "int32", scope="warp", offset_factor=1)
                                        for tx in T.thread_binding(32, thread="threadIdx.x"):
                                            T.mma_fill("int32", 8, C_warp.data, C_warp.elem_offset)
                            for ax3_0_0 in T.serial(256, annotations={"software_pipeline_async_stages": [0], "software_pipeline_order": [0, 1, 2, 3], "software_pipeline_stage": [0, 0, 1, 1]}):
                                for ax0_ax1_ax2_ax3_ax4_fused_0 in T.thread_binding(2, thread="threadIdx.y"):
                                    for ax0_ax1_ax2_ax3_ax4_fused_1 in T.thread_binding(2, thread="threadIdx.z"):
                                        for ax0_ax1_ax2_ax3_ax4_fused_2 in T.unroll(8, annotations={"pragma_unroll_explicit": 0}):
                                            for ax0_ax1_ax2_ax3_ax4_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax0_ax1_ax2_ax3_ax4_fused_4 in T.vectorized(16):
                                                    with T.block("A_reindex_reindex_shared"):
                                                        v0 = T.axis.spatial(1, 0)
                                                        v1 = T.axis.spatial(1024, ax1_0_0_ax2_0_0_fused * 16 + (ax0_ax1_ax2_ax3_ax4_fused_0 * 8192 + ax0_ax1_ax2_ax3_ax4_fused_1 * 4096 + ax0_ax1_ax2_ax3_ax4_fused_2 * 512 + ax0_ax1_ax2_ax3_ax4_fused_3 * 16 + ax0_ax1_ax2_ax3_ax4_fused_4) // 1024)
                                                        v2 = T.axis.spatial(512, ax3_0_0 * 2 + (ax0_ax1_ax2_ax3_ax4_fused_0 * 8192 + ax0_ax1_ax2_ax3_ax4_fused_1 * 4096 + ax0_ax1_ax2_ax3_ax4_fused_2 * 512 + ax0_ax1_ax2_ax3_ax4_fused_3 * 16 + ax0_ax1_ax2_ax3_ax4_fused_4) % 1024 // 512)
                                                        v3 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_ax4_fused_0 * 8192 + ax0_ax1_ax2_ax3_ax4_fused_1 * 4096 + ax0_ax1_ax2_ax3_ax4_fused_2 * 512 + ax0_ax1_ax2_ax3_ax4_fused_3 * 16 + ax0_ax1_ax2_ax3_ax4_fused_4) % 512 // 32)
                                                        v4 = T.axis.spatial(32, (ax0_ax1_ax2_ax3_ax4_fused_0 * 8192 + ax0_ax1_ax2_ax3_ax4_fused_1 * 4096 + ax0_ax1_ax2_ax3_ax4_fused_2 * 512 + ax0_ax1_ax2_ax3_ax4_fused_3 * 16 + ax0_ax1_ax2_ax3_ax4_fused_4) % 32)
                                                        T.reads(A[v1, v2, v3, v4])
                                                        T.writes(A_reindex_reindex_shared[v0, v1, v2, v3, v4])
                                                        T.block_attr({"permuted_layout": 0})
                                                        A_reindex_reindex_shared[v0, v1, v2, v3, v4] = A[v1, v2, v3, v4]
                                for ax0_ax1_ax2_ax3_fused_0 in T.unroll(1, annotations={"pragma_unroll_explicit": 0}):
                                    for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(2, thread="threadIdx.z"):
                                        for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(2, thread="threadIdx.y"):
                                            for ax0_ax1_ax2_ax3_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax0_ax1_ax2_ax3_fused_4 in T.vectorized(16):
                                                    with T.block("B_shared"):
                                                        v0 = T.axis.spatial(1024, ax1_0_1_ax2_0_1_fused * 4 + (ax0_ax1_ax2_ax3_fused_0 * 2048 + ax0_ax1_ax2_ax3_fused_1 * 1024 + ax0_ax1_ax2_ax3_fused_2 * 512 + ax0_ax1_ax2_ax3_fused_3 * 16 + ax0_ax1_ax2_ax3_fused_4) // 256)
                                                        v1 = T.axis.spatial(512, ax3_0_0 * 2 + (ax0_ax1_ax2_ax3_fused_0 * 2048 + ax0_ax1_ax2_ax3_fused_1 * 1024 + ax0_ax1_ax2_ax3_fused_2 * 512 + ax0_ax1_ax2_ax3_fused_3 * 16 + ax0_ax1_ax2_ax3_fused_4) % 256 // 128)
                                                        v2 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_fused_0 * 2048 + ax0_ax1_ax2_ax3_fused_1 * 1024 + ax0_ax1_ax2_ax3_fused_2 * 512 + ax0_ax1_ax2_ax3_fused_3 * 16 + ax0_ax1_ax2_ax3_fused_4) % 128 // 8)
                                                        v3 = T.axis.spatial(8, (ax0_ax1_ax2_ax3_fused_0 * 2048 + ax0_ax1_ax2_ax3_fused_1 * 1024 + ax0_ax1_ax2_ax3_fused_2 * 512 + ax0_ax1_ax2_ax3_fused_3 * 16 + ax0_ax1_ax2_ax3_fused_4) % 8)
                                                        T.where((((ax0_ax1_ax2_ax3_fused_0 * 2 + ax0_ax1_ax2_ax3_fused_1) * 2 + ax0_ax1_ax2_ax3_fused_2) * 32 + ax0_ax1_ax2_ax3_fused_3) * 16 + ax0_ax1_ax2_ax3_fused_4 < 1024)
                                                        T.reads(B[v0, v1, v2, v3])
                                                        T.writes(B_shared[v0, v1, v2, v3])
                                                        B_shared[v0, v1, v2, v3] = B[v0, v1, v2, v3]
                                for ax0_1, ax1_ax2_ax3_ax4_0_fused_0 in T.grid(1, 2):
                                    for ax1_ax2_ax3_ax4_0_fused_1 in T.thread_binding(2, thread="threadIdx.y"):
                                        for ax1_ax2_ax3_ax4_0_fused_2 in T.thread_binding(2, thread="threadIdx.z"):
                                            for ax1_ax2_ax3_ax4_0_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax4_1 in range(1):
                                                    for ax0_2, ax1, ax2 in T.grid(1, 1, 1):
                                                        for ax3 in T.vectorized(4):
                                                            with T.block("B_local"):
                                                                v0 = T.axis.spatial(1024, ax1_0_1_ax2_0_1_fused * 4 + (ax1_ax2_ax3_ax4_0_fused_0 * 128 + ax1_ax2_ax3_ax4_0_fused_1 * 64 + ax1_ax2_ax3_ax4_0_fused_2 * 32 + ax1_ax2_ax3_ax4_0_fused_3) // 64 + ax0_2)
                                                                v1 = T.axis.spatial(512, ax3_0_0 * 2 + (ax1_ax2_ax3_ax4_0_fused_0 * 128 + ax1_ax2_ax3_ax4_0_fused_1 * 64 + ax1_ax2_ax3_ax4_0_fused_2 * 32 + ax1_ax2_ax3_ax4_0_fused_3) % 64 // 32 + ax1)
                                                                v2 = T.axis.spatial(16, (ax1_ax2_ax3_ax4_0_fused_0 * 128 + ax1_ax2_ax3_ax4_0_fused_1 * 64 + ax1_ax2_ax3_ax4_0_fused_2 * 32 + ax1_ax2_ax3_ax4_0_fused_3) % 32 // 2 + ax2)
                                                                v3 = T.axis.spatial(8, (ax1_ax2_ax3_ax4_0_fused_0 * 128 + ax1_ax2_ax3_ax4_0_fused_1 * 64 + ax1_ax2_ax3_ax4_0_fused_2 * 32 + ax1_ax2_ax3_ax4_0_fused_3) % 2 * 4 + ax3)
                                                                T.reads(B_shared[v0, v1, v2, v3])
                                                                T.writes(B_local[v0, v1, v2, v3])
                                                                B_local[v0, v1, v2, v3] = B_shared[v0, v1, v2, v3]
                                                    for ax0_2, ax1, ax2, ax3, ax4 in T.grid(1, 1, 1, 1, 16):
                                                        with T.block("B_reindex_reindex_local"):
                                                            v0 = T.axis.spatial(1, ax0_2)
                                                            v1 = T.axis.spatial(1024, ax1_0_1_ax2_0_1_fused * 4 + (ax1_ax2_ax3_ax4_0_fused_0 * 128 + ax1_ax2_ax3_ax4_0_fused_1 * 64 + ax1_ax2_ax3_ax4_0_fused_2 * 32 + ax1_ax2_ax3_ax4_0_fused_3) // 64 + ax1)
                                                            v2 = T.axis.spatial(512, ax3_0_0 * 2 + (ax1_ax2_ax3_ax4_0_fused_0 * 128 + ax1_ax2_ax3_ax4_0_fused_1 * 64 + ax1_ax2_ax3_ax4_0_fused_2 * 32 + ax1_ax2_ax3_ax4_0_fused_3) % 64 // 32 + ax2)
                                                            v3 = T.axis.spatial(16, (ax1_ax2_ax3_ax4_0_fused_0 * 128 + ax1_ax2_ax3_ax4_0_fused_1 * 64 + ax1_ax2_ax3_ax4_0_fused_2 * 32 + ax1_ax2_ax3_ax4_0_fused_3) % 32 // 2 + ax3)
                                                            v4 = T.axis.spatial(32, (ax1_ax2_ax3_ax4_0_fused_0 * 128 + ax1_ax2_ax3_ax4_0_fused_1 * 64 + ax1_ax2_ax3_ax4_0_fused_2 * 32 + ax1_ax2_ax3_ax4_0_fused_3) % 2 * 16 + ax4)
                                                            T.reads(B_local[v1, v2, v3, v4 // 4])
                                                            T.writes(B_reindex_reindex_local[v0, v1, v2, v3, v4])
                                                            B_reindex_reindex_local[v0, v1, v2, v3, v4] = T.bitwise_and(T.shift_right(B_local[v1, v2, v3, v4 // 4], T.Cast("int8", v4 % 4 * 2)), T.int8(3))
                                                    for ax4_2 in T.vectorized(16):
                                                        with T.block("B_reindex_reindex_shared"):
                                                            v0 = T.axis.spatial(1, ax0_1)
                                                            v1 = T.axis.spatial(1024, ax1_0_1_ax2_0_1_fused * 4 + (ax1_ax2_ax3_ax4_0_fused_0 * 128 + ax1_ax2_ax3_ax4_0_fused_1 * 64 + ax1_ax2_ax3_ax4_0_fused_2 * 32 + ax1_ax2_ax3_ax4_0_fused_3) // 64)
                                                            v2 = T.axis.spatial(512, ax3_0_0 * 2 + (ax1_ax2_ax3_ax4_0_fused_0 * 128 + ax1_ax2_ax3_ax4_0_fused_1 * 64 + ax1_ax2_ax3_ax4_0_fused_2 * 32 + ax1_ax2_ax3_ax4_0_fused_3) % 64 // 32)
                                                            v3 = T.axis.spatial(16, (ax1_ax2_ax3_ax4_0_fused_0 * 128 + ax1_ax2_ax3_ax4_0_fused_1 * 64 + ax1_ax2_ax3_ax4_0_fused_2 * 32 + ax1_ax2_ax3_ax4_0_fused_3) % 32 // 2)
                                                            v4 = T.axis.spatial(32, (ax1_ax2_ax3_ax4_0_fused_0 * 128 + ax1_ax2_ax3_ax4_0_fused_1 * 64 + ax1_ax2_ax3_ax4_0_fused_2 * 32 + ax1_ax2_ax3_ax4_0_fused_3) % 2 * 16 + ax4_1 * 16 + ax4_2)
                                                            T.reads(B_reindex_reindex_local[v0, v1, v2, v3, v4])
                                                            T.writes(B_reindex_reindex_shared[v0, v1, v2, v3, v4])
                                                            T.block_attr({"permuted_layout": 0})
                                                            B_reindex_reindex_shared[v0, v1, v2, v3, v4] = B_reindex_reindex_local[v0, v1, v2, v3, v4]
                                for ax3_0_1 in range(2):
                                    for ax0_1, ax1, ax2, ax3_0, ax4_0 in T.grid(1, 8, 1, 1, 1):
                                        with T.block("A_reindex_reindex_shared_warp_o"):
                                            v0_o = T.axis.spatial(1, ax0_1)
                                            v1_o = T.axis.spatial(1024, ax1_0_0_ax2_0_0_fused * 16 + ax1_0_2 * 8 + ax1)
                                            v2_o = T.axis.spatial(512, ax3_0_0 * 2 + ax3_0_1 + ax2)
                                            v3_o, v4_o = T.axis.remap("SS", [ax3_0, ax4_0])
                                            T.reads(A_reindex_reindex_shared[v0_o, v1_o, v2_o, 0:16, 0:32])
                                            T.writes(A_reindex_reindex_shared_warp[v0_o, v1_o, v2_o, 0:32, 0:16])
                                            T.block_attr({"permuted_layout": 0})
                                            warp = T.match_buffer(A_reindex_reindex_shared_warp[v0_o, v1_o, v2_o, 0:32, 0:16], (32, 16), "int8", scope="warp", offset_factor=32)
                                            shared = T.match_buffer(A_reindex_reindex_shared[v0_o, v1_o, v2_o, 0:16, 0:32], (16, 32), "int8", strides=("shared_s0", "shared_s1"), scope="shared", offset_factor=32)
                                            for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                T.ptx_ldmatrix("int8", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + 16 * tx, T.tvm_access_ptr(T.type_annotation("int8"), shared.data, shared.elem_offset, shared.strides[0] * 16, 1), tx * 16)
                                    for ax0_1, ax1, ax2, ax3_0, ax4_0 in T.grid(1, 2, 1, 1, 1):
                                        with T.block("B_reindex_reindex_shared_warp_o"):
                                            v0_o = T.axis.spatial(1, ax0_1)
                                            v1_o = T.axis.spatial(1024, ax1_0_1_ax2_0_1_fused * 4 + ax2_0_2 * 2 + ax1)
                                            v2_o = T.axis.spatial(512, ax3_0_0 * 2 + ax3_0_1 + ax2)
                                            v3_o, v4_o = T.axis.remap("SS", [ax3_0, ax4_0])
                                            T.reads(B_reindex_reindex_shared[v0_o, v1_o, v2_o, 0:16, 0:32])
                                            T.writes(B_reindex_reindex_shared_warp[v0_o, v1_o, v2_o, 0:32, 0:16])
                                            T.block_attr({"permuted_layout": 0})
                                            warp = T.match_buffer(B_reindex_reindex_shared_warp[v0_o, v1_o, v2_o, 0:32, 0:16], (32, 16), "int8", scope="warp", offset_factor=32)
                                            shared = T.match_buffer(B_reindex_reindex_shared[v0_o, v1_o, v2_o, 0:16, 0:32], (16, 32), "int8", strides=("shared_s0", "shared_s1"), scope="shared", offset_factor=32)
                                            for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                T.ptx_ldmatrix("int8", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + 16 * tx, T.tvm_access_ptr(T.type_annotation("int8"), shared.data, shared.elem_offset, shared.strides[0] * 16, 1), tx * 16)
                                    for ax1_0_3, ax2_0_3 in T.grid(8, 2):
                                        with T.block("C_o_update"):
                                            v0_o = T.axis.spatial(1, ax0)
                                            v1_o = T.axis.spatial(1024, ax1_0_0_ax2_0_0_fused * 16 + ax1_0_2 * 8 + ax1_0_3)
                                            v2_o = T.axis.spatial(1024, ax1_0_1_ax2_0_1_fused * 4 + ax2_0_2 * 2 + ax2_0_3)
                                            v3_o = T.axis.reduce(512, ax3_0_0 * 2 + ax3_0_1)
                                            T.reads(C_reindex_shared_warp[0, v1_o, v2_o, 0:32, 0:8], A_reindex_reindex_shared_warp[0, v1_o, v3_o, 0:32, 0:16], B_reindex_reindex_shared_warp[0, v2_o, v3_o, 0:32, 0:16])
                                            T.writes(C_reindex_shared_warp[0, v1_o, v2_o, 0:32, 0:8])
                                            with T.block("C_o"):
                                                v1_i_o = T.axis.spatial(1, 0)
                                                v2_i_o = T.axis.spatial(1, 0)
                                                v3_i_o = T.axis.reduce(1, 0)
                                                T.reads(C_reindex_shared_warp[0, v1_o, v2_o, 0:32, 0:8], A_reindex_reindex_shared_warp[0, v1_o, v3_o, 0:32, 0:16], B_reindex_reindex_shared_warp[0, v2_o, v3_o, 0:32, 0:16])
                                                T.writes(C_reindex_shared_warp[0, v1_o, v2_o, 0:32, 0:8])
                                                A_1 = T.match_buffer(A_reindex_reindex_shared_warp[0, v1_o, v3_o, 0:32, 0:16], (32, 16), "int8", scope="warp", offset_factor=32)
                                                B_1 = T.match_buffer(B_reindex_reindex_shared_warp[0, v2_o, v3_o, 0:32, 0:16], (32, 16), "int8", scope="warp", offset_factor=32)
                                                C_1 = T.match_buffer(C_reindex_shared_warp[0, v1_o, v2_o, 0:32, 0:8], (32, 8), "int32", scope="warp", offset_factor=16)
                                                for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                    T.ptx_mma("int32", "m16n8k32", "row", "col", "int8", "int8", "int32", A_1.data, A_1.elem_offset + tx * 16, B_1.data, B_1.elem_offset + tx * 16, C_1.data, C_1.elem_offset + tx * 8, T.bool(False))
                                                    T.ptx_mma("int32", "m16n8k32", "row", "col", "int8", "int8", "int32", A_1.data, A_1.elem_offset + tx * 16, B_1.data, B_1.elem_offset + tx * 16 + 8, C_1.data, C_1.elem_offset + tx * 8 + 4, T.bool(False))
                            for ax0_1, ax1 in T.grid(8, 2):
                                for ax2_0, ax3_0 in T.grid(1, 1):
                                    with T.block("C_reindex_shared_warp_o"):
                                        v0_o = T.axis.spatial(1, 0)
                                        v1_o = T.axis.spatial(1024, ax1_0_0_ax2_0_0_fused * 16 + ax1_0_2 * 8 + ax0_1)
                                        v2_o = T.axis.spatial(1024, ax1_0_1_ax2_0_1_fused * 4 + ax2_0_2 * 2 + ax1)
                                        v3_o, v4_o = T.axis.remap("SS", [ax2_0, ax3_0])
                                        T.reads(C_reindex_shared_warp[v0_o, v1_o, v2_o, 0:32, 0:8])
                                        T.writes(C_reindex_shared[v0_o, v1_o, v2_o, 0:16, 0:16])
                                        C_warp = T.match_buffer(C_reindex_shared_warp[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "int32", scope="warp", offset_factor=1)
                                        C_1 = T.match_buffer(C_reindex_shared[v0_o, v1_o, v2_o, 0:16, 0:16], (16, 16), "int32", strides=("C_s0", "C_s1"), scope="shared", offset_factor=1)
                                        for tx in T.thread_binding(32, thread="threadIdx.x"):
                                            T.mma_store("int32", 16, 16, T.tvm_access_ptr(T.type_annotation("int32"), C_1.data, C_1.elem_offset, C_1.strides[0] * 16, 2), C_warp.data, C_warp.elem_offset, C_1.strides[0])
                                for ax0_ax1_ax2_ax3_ax4_fused_0 in T.unroll(2, annotations={"pragma_unroll_explicit": 0}):
                                    for ax0_ax1_ax2_ax3_ax4_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_ax1_ax2_ax3_ax4_fused_2 in T.vectorized(4):
                                            with T.block("C_reindex_shared"):
                                                v0 = T.axis.spatial(1, 0)
                                                v1 = T.axis.spatial(1024, ax1_0_0_ax2_0_0_fused * 16 + ax1_0_2 * 8 + ax0_1)
                                                v2 = T.axis.spatial(1024, ax1_0_1_ax2_0_1_fused * 4 + ax2_0_2 * 2 + ax1)
                                                v3 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_ax4_fused_0 * 128 + ax0_ax1_ax2_ax3_ax4_fused_1 * 4 + ax0_ax1_ax2_ax3_ax4_fused_2) // 16)
                                                v4 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_ax4_fused_0 * 128 + ax0_ax1_ax2_ax3_ax4_fused_1 * 4 + ax0_ax1_ax2_ax3_ax4_fused_2) % 16)
                                                T.reads(C_reindex_shared[v0, v1, v2, v3, v4])
                                                T.writes(C[v3 + v1 * 16, v4 + v2 * 16])
                                                C[v3 + v1 * 16, v4 + v2 * 16] = C_reindex_shared[v0, v1, v2, v3, v4]

mod = Module
sch = tvm.tir.Schedule(mod, debug_mask="all")
with tvm.transform.PassContext(
            config={"tir.use_async_copy": True}
        ):
    dense_relu_0_rt_mod = tvm.build(sch.mod, target="cuda")
with open("after_memory_rewrite.cu", "+w") as f:
    f.write(dense_relu_0_rt_mod.imported_modules[0].get_source())
