import tvm
from tvm.script import ir as I
from tvm.script import tir as T
import bitblas
from bitblas.base.roller.policy import TensorCorePolicy, DefaultPolicy
from bitblas.base.roller.arch import CUDA
from bitblas.gpu.matmul_analysis import get_tensorized_func_and_tags
from bitblas.base.utils import apply_and_build
from bitblas.ops.matmul_impl import matmul_nt, matmul_nt_dequantize_b
import numpy as np


@I.ir_module
class Module:
    @T.prim_func
    def dequantize_gemv(
        lv47: T.Buffer((T.int64(256), T.int64(256), T.int64(16), T.int64(2)), "uint32"),
        lv48: T.Buffer((T.int64(4096), T.int64(128)), "float16"),
        lv41: T.Buffer((T.int64(1), 1, T.int64(4096)), "float16"),
        NT_matmul_intermediate: T.Buffer((T.int64(1), 1, T.int64(4096)), "float16"),
    ):
        T.func_attr(
            {
                "dequantize_info": {
                    "decode": {
                        "decode_block": "decode",
                        "fast_decoding": T.bool(True),
                        "group_size": 32,
                        "source_format": {"bits": 4, "format": "int"},
                        "storage_dtype": "uint32",
                        "target_format": "float16",
                        "with_scaling": T.bool(True),
                    }
                },
                "smooth_b": T.bool(True),
                "tir.noalias": T.bool(True),
                "transform_kind": 2,
            }
        )
        # with T.block("root"):
        decode_intermediate_intermediate = T.alloc_buffer(
            (T.int64(4096), T.int64(4096)), "float16"
        )
        lv47_global = T.alloc_buffer((T.int64(4096), T.int64(512)), "uint32")
        for ax0, ax1 in T.grid(T.int64(4096), T.int64(512)):
            with T.block("lv47_global"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(
                    lv47[
                        v0 // T.int64(16),
                        v1 // T.int64(2),
                        v0 % T.int64(16) // T.int64(8) * T.int64(8)
                        + v0 % T.int64(4) * T.int64(2)
                        + v1 % T.int64(2),
                        v0 % T.int64(8) // T.int64(4),
                    ]
                )
                T.writes(lv47_global[v0, v1])
                lv47_global[v0, v1] = lv47[
                    v0 // T.int64(16),
                    v1 // T.int64(2),
                    v0 % T.int64(16) // T.int64(8) * T.int64(8)
                    + v0 % T.int64(4) * T.int64(2)
                    + v1 % T.int64(2),
                    v0 % T.int64(8) // T.int64(4),
                ]
        for i, j in T.grid(T.int64(4096), T.int64(4096)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(
                    lv47_global[v_i, v_j // T.int64(8)], lv48[v_i, v_j // T.int64(32)]
                )
                T.writes(decode_intermediate_intermediate[v_i, v_j])
                decode_intermediate_intermediate[v_i, v_j] = (
                    T.Cast(
                        "float16",
                        T.bitwise_and(
                            T.shift_right(
                                lv47_global[v_i, v_j // T.int64(8)],
                                T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4),
                            ),
                            T.uint32(15),
                        ),
                    )
                    - T.float16(7)
                ) * lv48[v_i, v_j // T.int64(32)]
        for i0, i1, i2, k in T.grid(T.int64(1), 1, T.int64(4096), T.int64(4096)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(
                    lv41[v_i0, v_i1, v_k], decode_intermediate_intermediate[v_i2, v_k]
                )
                T.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                    NT_matmul_intermediate[v_i0, v_i1, v_i2]
                    + lv41[v_i0, v_i1, v_k]
                    * decode_intermediate_intermediate[v_i2, v_k]
                )

    @T.prim_func
    def dequantize_gemm(
        lv47: T.Buffer((T.int64(256), T.int64(256), T.int64(16), T.int64(2)), "uint32"),
        lv48: T.Buffer((T.int64(4096), T.int64(128)), "float16"),
        lv41: T.Buffer((T.int64(1), T.int64(4096), T.int64(4096)), "float16"),
        NT_matmul_intermediate: T.Buffer((T.int64(1), 4096, T.int64(4096)), "float16"),
    ):
        T.func_attr(
            {
                "dequantize_info": {
                    "decode": {
                        "decode_block": "decode",
                        "fast_decoding": T.bool(True),
                        "group_size": 32,
                        "source_format": {"bits": 4, "format": "int"},
                        "storage_dtype": "uint32",
                        "target_format": "float16",
                        "with_scaling": T.bool(True),
                    }
                },
                "smooth_b": T.bool(True),
                "tir.noalias": T.bool(True),
                "transform_kind": 2,
            }
        )
        # with T.block("root"):
        decode_intermediate_intermediate = T.alloc_buffer(
            (T.int64(4096), T.int64(4096)), "float16"
        )
        lv47_global = T.alloc_buffer((T.int64(4096), T.int64(512)), "uint32")
        for ax0, ax1 in T.grid(T.int64(4096), T.int64(512)):
            with T.block("lv47_global"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(
                    lv47[
                        v0 // T.int64(16),
                        v1 // T.int64(2),
                        v0 % T.int64(16) // T.int64(8) * T.int64(8)
                        + v0 % T.int64(4) * T.int64(2)
                        + v1 % T.int64(2),
                        v0 % T.int64(8) // T.int64(4),
                    ]
                )
                T.writes(lv47_global[v0, v1])
                lv47_global[v0, v1] = lv47[
                    v0 // T.int64(16),
                    v1 // T.int64(2),
                    v0 % T.int64(16) // T.int64(8) * T.int64(8)
                    + v0 % T.int64(4) * T.int64(2)
                    + v1 % T.int64(2),
                    v0 % T.int64(8) // T.int64(4),
                ]
        for i, j in T.grid(T.int64(4096), T.int64(4096)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(
                    lv47_global[v_i, v_j // T.int64(8)], lv48[v_i, v_j // T.int64(32)]
                )
                T.writes(decode_intermediate_intermediate[v_i, v_j])
                decode_intermediate_intermediate[v_i, v_j] = (
                    T.Cast(
                        "float16",
                        T.bitwise_and(
                            T.shift_right(
                                lv47_global[v_i, v_j // T.int64(8)],
                                T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4),
                            ),
                            T.uint32(15),
                        ),
                    )
                    - T.float16(7)
                ) * lv48[v_i, v_j // T.int64(32)]
        for i0, i1, i2, k in T.grid(
            T.int64(1), T.int64(4096), T.int64(4096), T.int64(4096)
        ):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(
                    lv41[v_i0, v_i1, v_k], decode_intermediate_intermediate[v_i2, v_k]
                )
                T.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                    NT_matmul_intermediate[v_i0, v_i1, v_i2]
                    + lv41[v_i0, v_i1, v_k]
                    * decode_intermediate_intermediate[v_i2, v_k]
                )


ir_module = Module
func = ir_module["dequantize_gemm"]
target = tvm.target.Target("nvidia/nvidia-a100")
arch = CUDA(target)
policy = DefaultPolicy(func=func, arch=arch)

tensorized_func, tags = get_tensorized_func_and_tags(func, arch.target)
if tags:
    policy = TensorCorePolicy(func=tensorized_func, arch=arch, tags=tags)

configs = policy.emit_config(20)
print(configs)
sch = bitblas.gpu.MatmulTensorizationMMAWithDequantizeInfo().apply_config(
    func, configs[0]
)
# cpresults, best = apply_and_build(func, configs, arch, parallel_build=True)
# print(
#     "[FastDlight] The best latency of top 1 is {:.3f} ms".format(
#         cpresults[0].latency * 1e3
#     )
# )
# print(
#     "[FastDlight] The best latency of top 20 is {:.3f} ms".format(
#         best.latency * 1e3
#     )
# )
# with open("debug/tmp.cu", "w") as f:
#     f.write(str(best.code))
