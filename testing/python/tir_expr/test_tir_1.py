import tvm
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.tir.tensor_intrin.cuda import *

# from tvm.script import tir as T
@T.prim_func
def main(input0: T.Buffer[(1024, 512, 16, 32), "int8"], input1: T.Buffer[(1024, 512, 16, 8), "int8"], output0: T.Buffer[(16384, 16384), "int8"]):
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # var definition
    tx = T.env_thread("threadIdx.x")
    C_s0 = T.var("int32")
    C_s1 = T.var("int32")
    shared_s0 = T.var("int32")
    shared_s0_1 = T.var("int32")
    shared_s1 = T.var("int32")
    shared_s1_1 = T.var("int32")
    # body
    # with T.block("root")
    input0_shared = T.alloc_buffer([1024, 512, 16, 32], dtype="int8", scope="shared")
    mediate0_shared = T.alloc_buffer([1024, 512, 16, 32], dtype="int8", scope="shared")
    mediate1_shared = T.alloc_buffer([1024, 1024, 16, 16], dtype="int32", scope="shared")
    mediate1_shared_warp = T.alloc_buffer([1024, 1024, 32, 8], dtype="int32", scope="warp")
    mediate0_local = T.alloc_buffer([1024, 512, 16, 32], dtype="int8", scope="local")
    input1_shared = T.alloc_buffer([1024, 512, 16, 8], dtype="int8", scope="shared")
    input1_shared_local = T.alloc_buffer([1024, 512, 16, 8], dtype="int8", scope="local")
    input0_shared_warp = T.alloc_buffer([1024, 512, 32, 16], dtype="int8", scope="warp")
    mediate0_shared_warp = T.alloc_buffer([1024, 512, 32, 16], dtype="int8", scope="warp")
    for i_0 in T.thread_binding(256, thread="blockIdx.y"):
        for j_0 in T.thread_binding(64, thread="blockIdx.x"):
            for i_1 in T.thread_binding(2, thread="threadIdx.y"):
                for j_1 in T.thread_binding(2, thread="threadIdx.z"):
                    for i_2_init in T.serial(2, annotations={"pragma_unroll_explicit":0, "thread_rasterization":10}):
                        for j_2_init in T.serial(8, annotations={"pragma_unroll_explicit":0}):
                            with T.block("mediate1_init_o"):
                                v_i = T.axis.spatial(1024, i_0 * 4 + i_1 * 2 + i_2_init)
                                v_j = T.axis.spatial(1024, j_0 * 16 + j_1 * 8 + j_2_init)
                                v_ii_o = T.axis.spatial(1, 0)
                                v_jj_o = T.axis.spatial(1, 0)
                                T.reads()
                                T.writes(mediate1_shared_warp[v_i, v_j, 0 : 32, 0 : 8])
                                C_warp = T.match_buffer(mediate1_shared_warp[v_i, v_j, 0 : 32, 0 : 8], [32, 8], dtype="int32", scope="warp", offset_factor=1)
                                T.launch_thread(tx, 32)
                                T.mma_fill(8, C_warp.data, C_warp.elem_offset, dtype="int32")
                    for k_0 in T.serial(256, annotations={"software_pipeline_async_stages":[0], "software_pipeline_order":[0, 1, 2, 3], "software_pipeline_stage":[0, 0, 1, 1]}):
                        for ax0_ax1_ax2_ax3_0_fused_0 in T.unroll(2, annotations={"pragma_unroll_explicit":0}):
                            for ax0_ax1_ax2_ax3_0_fused_1 in T.thread_binding(2, thread="threadIdx.y"):
                                for ax0_ax1_ax2_ax3_0_fused_2 in T.thread_binding(2, thread="threadIdx.z"):
                                    for ax0_ax1_ax2_ax3_0_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax3_1 in T.vectorized(16):
                                            with T.block("input0_shared"):
                                                v0 = T.axis.spatial(1024, i_0 * 4 + (ax0_ax1_ax2_ax3_0_fused_0 * 128 + ax0_ax1_ax2_ax3_0_fused_1 * 64 + ax0_ax1_ax2_ax3_0_fused_2 * 32 + ax0_ax1_ax2_ax3_0_fused_3) // 64)
                                                v1 = T.axis.spatial(512, k_0 * 2 + (ax0_ax1_ax2_ax3_0_fused_0 * 128 + ax0_ax1_ax2_ax3_0_fused_1 * 64 + ax0_ax1_ax2_ax3_0_fused_2 * 32 + ax0_ax1_ax2_ax3_0_fused_3) % 64 // 32)
                                                v2 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_0_fused_0 * 128 + ax0_ax1_ax2_ax3_0_fused_1 * 64 + ax0_ax1_ax2_ax3_0_fused_2 * 32 + ax0_ax1_ax2_ax3_0_fused_3) % 32 // 2)
                                                v3 = T.axis.spatial(32, (ax0_ax1_ax2_ax3_0_fused_0 * 128 + ax0_ax1_ax2_ax3_0_fused_1 * 64 + ax0_ax1_ax2_ax3_0_fused_2 * 32 + ax0_ax1_ax2_ax3_0_fused_3) % 2 * 16 + ax3_1)
                                                T.reads(input0[v0, v1, v2, v3])
                                                T.writes(input0_shared[v0, v1, v2, v3])
                                                input0_shared[v0, v1, v2, v3] = input0[v0, v1, v2, v3]
                        for ax0_ax1_ax2_ax3_fused_0_0_0_0 in T.serial(2):
                            for ax0_ax1_ax2_ax3_fused_0_0_0_1 in T.thread_binding(2, thread="threadIdx.z"):
                                for ax0_ax1_ax2_ax3_fused_0_0_1 in T.thread_binding(2, thread="threadIdx.y"):
                                    for ax0_ax1_ax2_ax3_fused_0_1 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_ax1_ax2_ax3_fused_1 in T.vectorized(16):
                                            with T.block("input1_shared"):
                                                v0 = T.axis.spatial(1024, j_0 * 16 + (ax0_ax1_ax2_ax3_fused_0_0_0_0 * 2048 + ax0_ax1_ax2_ax3_fused_0_0_0_1 * 1024 + ax0_ax1_ax2_ax3_fused_0_0_1 * 512 + ax0_ax1_ax2_ax3_fused_0_1 * 16 + ax0_ax1_ax2_ax3_fused_1) // 256)
                                                v1 = T.axis.spatial(512, k_0 * 2 + (ax0_ax1_ax2_ax3_fused_0_0_0_0 * 2048 + ax0_ax1_ax2_ax3_fused_0_0_0_1 * 1024 + ax0_ax1_ax2_ax3_fused_0_0_1 * 512 + ax0_ax1_ax2_ax3_fused_0_1 * 16 + ax0_ax1_ax2_ax3_fused_1) % 256 // 128)
                                                v2 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_fused_0_0_0_0 * 2048 + ax0_ax1_ax2_ax3_fused_0_0_0_1 * 1024 + ax0_ax1_ax2_ax3_fused_0_0_1 * 512 + ax0_ax1_ax2_ax3_fused_0_1 * 16 + ax0_ax1_ax2_ax3_fused_1) % 128 // 8)
                                                v3 = T.axis.spatial(8, (ax0_ax1_ax2_ax3_fused_0_0_0_0 * 2048 + ax0_ax1_ax2_ax3_fused_0_0_0_1 * 1024 + ax0_ax1_ax2_ax3_fused_0_0_1 * 512 + ax0_ax1_ax2_ax3_fused_0_1 * 16 + ax0_ax1_ax2_ax3_fused_1) % 8)
                                                T.reads(input1[v0, v1, v2, v3])
                                                T.writes(input1_shared[v0, v1, v2, v3])
                                                input1_shared[v0, v1, v2, v3] = input1[v0, v1, v2, v3]
                        for ax0_ax1_ax2_ax3_0_fused_0 in T.serial(8):
                            for ax0_ax1_ax2_ax3_0_fused_1 in T.thread_binding(2, thread="threadIdx.y"):
                                for ax0_ax1_ax2_ax3_0_fused_2 in T.thread_binding(2, thread="threadIdx.z"):
                                    for ax0_ax1_ax2_ax3_0_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax3_1 in T.serial(1):
                                            for ax0 in T.vectorized(4):
                                                with T.block("input1_shared_local"):
                                                    v0 = T.axis.spatial(1024, j_0 * 16 + (ax0_ax1_ax2_ax3_0_fused_0 * 128 + ax0_ax1_ax2_ax3_0_fused_1 * 64 + ax0_ax1_ax2_ax3_0_fused_2 * 32 + ax0_ax1_ax2_ax3_0_fused_3) // 64)
                                                    v1 = T.axis.spatial(512, k_0 * 2 + (ax0_ax1_ax2_ax3_0_fused_0 * 128 + ax0_ax1_ax2_ax3_0_fused_1 * 64 + ax0_ax1_ax2_ax3_0_fused_2 * 32 + ax0_ax1_ax2_ax3_0_fused_3) % 64 // 32)
                                                    v2 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_0_fused_0 * 128 + ax0_ax1_ax2_ax3_0_fused_1 * 64 + ax0_ax1_ax2_ax3_0_fused_2 * 32 + ax0_ax1_ax2_ax3_0_fused_3) % 32 // 2)
                                                    v3 = T.axis.spatial(8, (ax0_ax1_ax2_ax3_0_fused_0 * 128 + ax0_ax1_ax2_ax3_0_fused_1 * 64 + ax0_ax1_ax2_ax3_0_fused_2 * 32 + ax0_ax1_ax2_ax3_0_fused_3) % 2 * 4 + ax0)
                                                    T.reads(input1_shared[v0, v1, v2, v3])
                                                    T.writes(input1_shared_local[v0, v1, v2, v3])
                                                    input1_shared_local[v0, v1, v2, v3] = input1_shared[v0, v1, v2, v3]
                                            for ax0 in T.serial(16):
                                                with T.block("mediate0_local"):
                                                    v0 = T.axis.spatial(1024, j_0 * 16 + (ax0_ax1_ax2_ax3_0_fused_0 * 128 + ax0_ax1_ax2_ax3_0_fused_1 * 64 + ax0_ax1_ax2_ax3_0_fused_2 * 32 + ax0_ax1_ax2_ax3_0_fused_3) // 64)
                                                    v1 = T.axis.spatial(512, k_0 * 2 + (ax0_ax1_ax2_ax3_0_fused_0 * 128 + ax0_ax1_ax2_ax3_0_fused_1 * 64 + ax0_ax1_ax2_ax3_0_fused_2 * 32 + ax0_ax1_ax2_ax3_0_fused_3) % 64 // 32)
                                                    v2 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_0_fused_0 * 128 + ax0_ax1_ax2_ax3_0_fused_1 * 64 + ax0_ax1_ax2_ax3_0_fused_2 * 32 + ax0_ax1_ax2_ax3_0_fused_3) % 32 // 2)
                                                    v3 = T.axis.spatial(32, (ax0_ax1_ax2_ax3_0_fused_0 * 128 + ax0_ax1_ax2_ax3_0_fused_1 * 64 + ax0_ax1_ax2_ax3_0_fused_2 * 32 + ax0_ax1_ax2_ax3_0_fused_3) % 2 * 16 + ax0)
                                                    T.reads(input1_shared_local[v0, v1, v2, v3 // 4])
                                                    T.writes(mediate0_local[v0, v1, v2, v3])
                                                    mediate0_local[v0, v1, v2, v3] = T.bitwise_and(T.shift_right(input1_shared_local[v0, v1, v2, v3 // 4], T.Cast("int8", v3 % 4), dtype="int8"), T.int8(1), dtype="int8")
                                            for ax3_2 in T.vectorized(16):
                                                with T.block("mediate0_shared"):
                                                    v0 = T.axis.spatial(1024, j_0 * 16 + (ax0_ax1_ax2_ax3_0_fused_0 * 128 + ax0_ax1_ax2_ax3_0_fused_1 * 64 + ax0_ax1_ax2_ax3_0_fused_2 * 32 + ax0_ax1_ax2_ax3_0_fused_3) // 64)
                                                    v1 = T.axis.spatial(512, k_0 * 2 + (ax0_ax1_ax2_ax3_0_fused_0 * 128 + ax0_ax1_ax2_ax3_0_fused_1 * 64 + ax0_ax1_ax2_ax3_0_fused_2 * 32 + ax0_ax1_ax2_ax3_0_fused_3) % 64 // 32)
                                                    v2 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_0_fused_0 * 128 + ax0_ax1_ax2_ax3_0_fused_1 * 64 + ax0_ax1_ax2_ax3_0_fused_2 * 32 + ax0_ax1_ax2_ax3_0_fused_3) % 32 // 2)
                                                    v3 = T.axis.spatial(32, (ax0_ax1_ax2_ax3_0_fused_0 * 128 + ax0_ax1_ax2_ax3_0_fused_1 * 64 + ax0_ax1_ax2_ax3_0_fused_2 * 32 + ax0_ax1_ax2_ax3_0_fused_3) % 2 * 16 + ax3_1 * 16 + ax3_2)
                                                    T.reads(mediate0_local[v0, v1, v2, v3])
                                                    T.writes(mediate0_shared[v0, v1, v2, v3])
                                                    mediate0_shared[v0, v1, v2, v3] = mediate0_local[v0, v1, v2, v3]
                        for k_1 in T.serial(2):
                            for ax0, ax1 in T.grid(2, 1):
                                with T.block("input0_shared_warp_o"):
                                    v0 = T.axis.spatial(1024, i_0 * 4 + i_1 * 2 + ax0)
                                    v1 = T.axis.spatial(512, ax1 * 512 + k_0 * 2 + k_1)
                                    v2_o = T.axis.spatial(1, 0)
                                    v3_o = T.axis.spatial(1, 0)
                                    T.reads(input0_shared[v0, v1, 0 : 16, 0 : 32])
                                    T.writes(input0_shared_warp[v0, v1, 0 : 32, 0 : 16])
                                    warp = T.match_buffer(input0_shared_warp[v0, v1, 0 : 32, 0 : 16], [32, 16], dtype="int8", scope="warp", offset_factor=16)
                                    shared = T.match_buffer(input0_shared[v0, v1, 0 : 16, 0 : 32], [16, 32], dtype="int8", strides=[shared_s0, shared_s1], scope="shared", offset_factor=16)
                                    T.launch_thread(tx, 32)
                                    T.ptx_ldmatrix(False, 4, ".b16", warp.data, warp.elem_offset + 16 * tx, T.tvm_access_ptr(T.type_annotation(dtype="int8"), shared.data, shared.elem_offset, shared_s0 * 16, 1, dtype="handle"), 16 * tx, dtype="int8")
                            for ax0, ax1 in T.grid(8, 1):
                                with T.block("mediate0_shared_warp_o"):
                                    v0 = T.axis.spatial(1024, j_0 * 16 + j_1 * 8 + ax0)
                                    v1 = T.axis.spatial(512, ax1 * 512 + k_0 * 2 + k_1)
                                    v2_o = T.axis.spatial(1, 0)
                                    v3_o = T.axis.spatial(1, 0)
                                    T.reads(mediate0_shared[v0, v1, 0 : 16, 0 : 32])
                                    T.writes(mediate0_shared_warp[v0, v1, 0 : 32, 0 : 16])
                                    warp_1 = T.match_buffer(mediate0_shared_warp[v0, v1, 0 : 32, 0 : 16], [32, 16], dtype="int8", scope="warp", offset_factor=16)
                                    shared_1 = T.match_buffer(mediate0_shared[v0, v1, 0 : 16, 0 : 32], [16, 32], dtype="int8", strides=[shared_s0_1, shared_s1_1], scope="shared", offset_factor=16)
                                    T.launch_thread(tx, 32)
                                    T.ptx_ldmatrix(False, 4, ".b16", warp_1.data, warp_1.elem_offset + 16 * tx, T.tvm_access_ptr(T.type_annotation(dtype="int8"), shared_1.data, shared_1.elem_offset, shared_s0_1 * 16, 1, dtype="handle"), 16 * tx, dtype="int8")
                            for i_2, j_2 in T.grid(2, 8):
                                with T.block("mediate1_update_o"):
                                    v_i = T.axis.spatial(1024, i_0 * 4 + i_1 * 2 + i_2)
                                    v_j = T.axis.spatial(1024, j_0 * 16 + j_1 * 8 + j_2)
                                    v_ii_o = T.axis.spatial(1, 0)
                                    v_jj_o = T.axis.spatial(1, 0)
                                    v_k = T.axis.reduce(512, k_0 * 2 + k_1)
                                    v_kk_o = T.axis.reduce(1, 0)
                                    T.reads(mediate1_shared_warp[v_i, v_j, 0 : 32, 0 : 8], input0_shared_warp[v_i, v_k, 0 : 32, 0 : 16], mediate0_shared_warp[v_j, v_k, 0 : 32, 0 : 16])
                                    T.writes(mediate1_shared_warp[v_i, v_j, 0 : 32, 0 : 8])
                                    A = T.match_buffer(input0_shared_warp[v_i, v_k, 0 : 32, 0 : 16], [32, 16], dtype="int8", scope="warp", offset_factor=16)
                                    B = T.match_buffer(mediate0_shared_warp[v_j, v_k, 0 : 32, 0 : 16], [32, 16], dtype="int8", scope="warp", offset_factor=16)
                                    C = T.match_buffer(mediate1_shared_warp[v_i, v_j, 0 : 32, 0 : 8], [32, 8], dtype="int32", scope="warp", offset_factor=16)
                                    T.launch_thread(tx, 32)
                                    T.ptx_mma("m16n8k32", "row", "col", "int8", "int8", "int32", A.data, A.elem_offset + tx * 16, B.data, B.elem_offset + tx * 16, C.data, C.elem_offset + tx * 8, False, dtype="int32")
                                    T.ptx_mma("m16n8k32", "row", "col", "int8", "int8", "int32", A.data, A.elem_offset + tx * 16, B.data, B.elem_offset + tx * 16 + T.FloorDiv(16, 2), C.data, C.elem_offset + tx * 8 + T.FloorDiv(8, 2), False, dtype="int32")
                    for ax0, ax1 in T.grid(2, 8):
                        with T.block("mediate1_shared_warp_o"):
                            v0 = T.axis.spatial(1024, i_0 * 4 + i_1 * 2 + ax0)
                            v1 = T.axis.spatial(1024, j_0 * 16 + j_1 * 8 + ax1)
                            v2_o = T.axis.spatial(1, 0)
                            v3_o = T.axis.spatial(1, 0)
                            T.reads(mediate1_shared_warp[v0, v1, 0 : 32, 0 : 8])
                            T.writes(mediate1_shared[v0, v1, 0 : 16, 0 : 16])
                            C_warp_1 = T.match_buffer(mediate1_shared_warp[v0, v1, 0 : 32, 0 : 8], [32, 8], dtype="int32", scope="warp", offset_factor=1)
                            C_1 = T.match_buffer(mediate1_shared[v0, v1, 0 : 16, 0 : 16], [16, 16], dtype="int32", strides=[C_s0, C_s1], scope="shared", offset_factor=1)
                            T.launch_thread(tx, 32)
                            T.mma_store(16, 16, T.tvm_access_ptr(T.type_annotation(dtype="int32"), C_1.data, C_1.elem_offset, C_s0 * 16, 2, dtype="handle"), C_warp_1.data, C_warp_1.elem_offset, C_s0, dtype="int32")
                        for ax0_ax1_ax2_ax3_fused_0 in T.unroll(2, annotations={"pragma_unroll_explicit":0}):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                                for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(4):
                                    with T.block("mediate1_shared"):
                                        v0 = T.axis.spatial(1024, i_0 * 4 + i_1 * 2 + ax0)
                                        v1 = T.axis.spatial(1024, j_0 * 16 + j_1 * 8 + ax1)
                                        v2 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_fused_0 * 128 + ax0_ax1_ax2_ax3_fused_1 * 4 + ax0_ax1_ax2_ax3_fused_2) // 16)
                                        v3 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_fused_0 * 128 + ax0_ax1_ax2_ax3_fused_1 * 4 + ax0_ax1_ax2_ax3_fused_2) % 16)
                                        T.reads(mediate1_shared[v0, v1, v2, v3])
                                        T.writes(output0[v0 * 16 + v2, v1 * 16 + v3])
                                        output0[v2 + v0 * 16, v3 + v1 * 16] = T.Cast("int8", mediate1_shared[v0, v1, v2, v3])

mod = main
sch = tvm.tir.Schedule(mod, debug_mask="all")
with tvm.transform.PassContext(
            config={"tir.use_async_copy": True}
        ):
    dense_relu_0_rt_mod = tvm.build(sch.mod, target="cuda")
with open("after_memory_rewrite.cu", "+w") as f:
    f.write(dense_relu_0_rt_mod.imported_modules[0].get_source())
