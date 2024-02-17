from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R

import tvm
import tvm.testing
from tvm import relax
from tvm.script import ir as I, relax as R, tir as T
from tvm import tir
from tvm.ir import IRModule
from tvm.ir.transform import PassContext, module_pass
import mlc_llm

@I.ir_module
class Before:
    @T.prim_func(private=True)
    def fused_fused_decode3_fused_NT_matmul8_add1(
        lv47: T.Buffer((T.int64(4096), T.int64(512)), "uint32"),
        lv48: T.Buffer((T.int64(4096), T.int64(128)), "float16"),
        p_lv41: T.handle,
        p_output0: T.handle,
    ):
        T.func_attr(
            {
                "tir.noalias": T.bool(True)
            }
        )
        n = T.int64()
        lv41 = T.match_buffer(p_lv41, (T.int64(1), n, T.int64(4096)), "float16")
        NT_matmul_intermediate = T.match_buffer(
            p_output0, (T.int64(1), n, T.int64(4096)), "float16"
        )
        # with T.block("root"):
        decode_intermediate_intermediate = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
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

    @R.function
    def main(
        lv47: R.Tensor((T.int64(4096), T.int64(512)), dtype="uint32"),
        lv48: R.Tensor((T.int64(4096), T.int64(128)), dtype="float16"),
        p_lv41:  R.Tensor((T.int64(1), "n", T.int64(4096)), dtype="float16"),
    ) -> R.Tensor((1, 4096), dtype="float16"):
        R.func_attr({"Primitive": 1})
        n = T.int64()
        cls = Before
        with R.dataflow():
            gv = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv47, lv48, p_lv41), out_sinfo=R.Tensor((1, n, 4096), dtype="float16"))
            R.output(gv)
        return gv


relax_mod = Before


import tvm
from tvm import dlight as dl
import bitblas
from tvm import relax
import bitblas
from bitblas.base.roller.policy import TensorCorePolicy, DefaultPolicy
from bitblas.base.roller.arch import CUDA
from bitblas.gpu.matmul_analysis import get_tensorized_func_and_tags
from bitblas.transform.annotate_decode_block import AnnotateDecodeInformation
from bitblas.transform.weight_only_propagate import WeightOnlyLayoutPropagation
dispatch_target = tvm.target.Target("nvidia/nvidia-a100")

relax_mod = AnnotateDecodeInformation()(relax_mod)

with dispatch_target:
    relax_mod = WeightOnlyLayoutPropagation(transform_level=0, faster_conversion=True)(relax_mod)

# with dispatch_target:
#     relax_mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
#         dl.gpu.Matmul(),
#         dl.gpu.GEMV(),
#         dl.gpu.Reduction(),
#         dl.gpu.GeneralReduction(),
#         dl.gpu.Fallback(),
#     )(relax_mod)
    
print(relax_mod)
# dynamic_range = {
#     "n": [1],
# }
# relax_mod = bitblas.ApplyFastTuning(
#     topk=1,
#     target=dispatch_target,
#     meta_database_dir="vicuna_tune",
#     whitelist=["matmul"],
#     dynamic_range=dynamic_range,
#     parallel_build=False
# )(relax_mod)

print(relax_mod)


# with tvm.transform.PassContext(config={"tir.use_async_copy": False}):
#     mod = tvm.build(relax_mod, target=dispatch_target)

# with open("debug/test_dl_fused_decode_matmul.cu", "+w") as f:
#     f.write(mod.imported_modules[0].get_source())
