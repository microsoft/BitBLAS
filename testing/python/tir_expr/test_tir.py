# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Metadata omitted. Use show_meta=True in script() method to show it.
from tvm.script import ir as I
from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1, 16384), "float16"), B: T.Buffer((16384, 8192), "int8"), Scale: T.Buffer((16384, 512), "float16"), D: T.Buffer((1, 16384), "float16")):
        T.func_attr({"dequantize_info": {"B": {"decode_block": "B_decode", "fast_decoding": T.bool(True), "group_size": 32, "source_format": {"bits": 4, "format": "uint"}, "target_format": "float16", "with_scaling": T.bool(True)}}, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        B_decode_local = T.alloc_buffer((16384, 16384), "float16", scope="local")
        A_local = T.alloc_buffer((1, 16384), "float16", scope="local")
        B_local = T.alloc_buffer((16384, 8192), "int8", scope="local")
        C_local = T.alloc_buffer((1, 16384), "float16", scope="local")
        for ax0_0 in T.thread_binding(8192, thread="blockIdx.x"):
            for ax0_1 in T.thread_binding(2, thread="threadIdx.y"):
                for ax1_0 in range(32):
                    for ax1_1 in T.thread_binding(64, thread="threadIdx.x"):
                        for ax0 in range(1):
                            for ax1 in T.vectorized(4):
                                with T.block("B_local"):
                                    v0 = T.axis.spatial(16384, ax0_0 * 2 + ax0_1 + ax0)
                                    v1 = T.axis.spatial(8192, ax1_0 * 256 + ax1_1 * 4 + ax1)
                                    T.reads(B[v0, v1])
                                    T.writes(B_local[v0, v1])
                                    B_local[v0, v1] = B[v0, v1]
                        for ax0 in range(1):
                            with T.block("B_decode_local_o"):
                                v0_o = T.axis.spatial(16384, ax0_0 * 2 + ax0_1 + ax0)
                                v1_o = T.axis.spatial(2048, ax1_0 * 64 + ax1_1)
                                T.reads(B_local[v0_o, v1_o * 4:v1_o * 4 + 4], Scale[v0_o, v1_o // 4])
                                T.writes(B_decode_local[v0_o, v1_o * 8:v1_o * 8 + 8])
                                Compressed = T.match_buffer(B_local[v0_o, v1_o * 4:v1_o * 4 + 4], (4,), "int8", scope="local")
                                Decompressed = T.match_buffer(B_decode_local[v0_o, v1_o * 8:v1_o * 8 + 8], (8,), "float16", scope="local")
                                # Scale_1 = T.match_buffer(Scale[v0_o, v1_o // 4: v1_o // 4 + 1], (1,), "float16")
                                Scale_1 = T.match_buffer(Scale[v0_o, v1_o // 4], (1,), "float16", elem_offset=Scale.elem_offset)
                                T.call_extern("handle", "decode_i4s_to_f16_scale", Compressed.data, Decompressed.data, Scale_1.access_ptr("r"), 8)
                        for ax0 in range(1):
                            for ax1 in T.vectorized(8):
                                with T.block("A_local"):
                                    v0 = T.axis.spatial(1, ax0)
                                    v1 = T.axis.spatial(16384, ax1_0 * 512 + ax1_1 * 8 + ax1)
                                    T.reads(A[v0, v1])
                                    T.writes(A_local[v0, v1])
                                    A_local[v0, v1] = A[v0, v1]
                        for ax1_2 in range(8):
                            with T.block("C"):
                                v0 = T.axis.spatial(16384, ax0_0 * 2 + ax0_1)
                                v1 = T.axis.reduce(16384, ax1_0 * 512 + ax1_1 * 8 + ax1_2)
                                T.reads(A_local[0, v1], B_decode_local[v0, v1])
                                T.writes(C_local[0, v0])
                                with T.init():
                                    C_local[0, v0] = T.float16(0)
                                C_local[0, v0] = C_local[0, v0] + A_local[0, v1] * B_decode_local[v0, v1]
                for ax0, ax1 in T.grid(1, 1):
                    with T.block("C_local"):
                        v0 = T.axis.spatial(1, ax0)
                        v1 = T.axis.spatial(16384, ax0_0 * 2 + ax0_1 + ax1)
                        T.reads(C_local[v0, v1])
                        T.writes(D[0, v1])
                        D[0, v1] = C_local[v0, v1]


import tvm
mod = Module
sch = tvm.tir.Schedule(mod, debug_mask="all")
with tvm.transform.PassContext(
            config={"tir.use_async_copy": True}
        ):
    dense_relu_0_rt_mod = tvm.build(sch.mod, target="cuda")
with open("debug/after_memory_rewrite.cu", "+w") as f:
    f.write(dense_relu_0_rt_mod.imported_modules[0].get_source())
