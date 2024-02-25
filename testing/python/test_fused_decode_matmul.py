# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R


@T.prim_func
def fused_fused_decode3_fused_NT_matmul8_add1(
    lv47: T.Buffer((T.int64(4096), T.int64(512)), "uint32"),
    lv48: T.Buffer((T.int64(4096), T.int64(128)), "float16"),
    p_lv41: T.handle,
    p_lv2: T.handle,
    p_output0: T.handle,
):
    T.func_attr(
        {
            "tir.noalias": T.bool(True),
            "dequantize_info": {
                "B": {
                    "decode_block": "decode",
                    "fast_decoding": True,
                    "source_format": {
                        "bits": 4,
                        "format": "int",
                    },
                    "with_scaling": True,
                    "storage_dtype": "uint32",
                    "group_size": 32,
                    "target_format": "float16",
                }
            },
        }
    )
    n = T.int64()
    lv41 = T.match_buffer(p_lv41, (T.int64(1), n, T.int64(4096)), "float16")
    lv2 = T.match_buffer(p_lv2, (T.int64(1), n, T.int64(4096)), "float16")
    T_add_intermediate_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(4096)), "float16"
    )
    # with T.block("root"):
    decode_intermediate_intermediate = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv47[v_i, v_j // T.int64(8)], lv48[v_i, v_j // T.int64(32)])
            T.writes(decode_intermediate_intermediate[v_i, v_j])
            decode_intermediate_intermediate[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv47[v_i, v_j // T.int64(8)],
                            T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv48[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv41[v_i0, v_i1, v_k], decode_intermediate_intermediate[v_i2, v_k])
            T.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                NT_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv41[v_i0, v_i1, v_k] * decode_intermediate_intermediate[v_i2, v_k]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv2[v_ax0, v_ax1, v_ax2], NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2])
            T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2] = (
                lv2[v_ax0, v_ax1, v_ax2] + NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]
            )


import tvm
from tvm import dlight as dl
import bitblas
from tvm import relax
import bitblas
from bitblas.base.roller.policy import TensorCorePolicy, DefaultPolicy
from bitblas.base.roller.arch import CUDA
from bitblas.gpu.matmul_analysis import get_tensorized_func_and_tags

dispatch_target = tvm.target.Target("cuda")
mod_deploy = tvm.IRModule.from_expr(fused_fused_decode3_fused_NT_matmul8_add1.specialize({"n": T.int64(1)}))
target = tvm.target.Target("nvidia/nvidia-a100")
arch = CUDA(target)
func = fused_fused_decode3_fused_NT_matmul8_add1.specialize({"n": T.int64(1)})
policy = DefaultPolicy(func=func, arch=arch)
try:
    tensorized_func, tags = get_tensorized_func_and_tags(func, arch.target)
except:
    tags = None
if tags:
    policy = TensorCorePolicy(func=tensorized_func, arch=arch, tags=tags)

configs = policy.emit_config(20)
print(configs[0])
sch = bitblas.gpu.gemv.GEMVWithDequantizeInfo().apply_config(func, configs[0])

# print(sch.mod)
# with dispatch_target:
#     mod_deploy = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
#         dl.gpu.Matmul(),
#         dl.gpu.GEMV(),
#         dl.gpu.Reduction(),
#         dl.gpu.GeneralReduction(),
#         dl.gpu.Fallback(),
#     )(mod_deploy)
# dynamic_range = {
#     "n": [64],
# }
# mod_deploy = bitblas.ApplyFastTuning(
#     topk=20,
#     target=dispatch_target,
#     meta_database_dir="vicuna_tune",
#     whitelist=["matmul"],
# )(mod_deploy)

# with tvm.transform.PassContext(config={"tir.use_async_copy": False}):
#     mod = tvm.build(mod_deploy, target=dispatch_target)

# with open("debug/test_dl_fused_decode_matmul.cu", "+w") as f:
#     f.write(mod.imported_modules[0].get_source())
