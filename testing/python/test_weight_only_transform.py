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
from bitblas.base.utils import get_dummy_input_arrays
from copy import deepcopy
import bitblas
from bitblas.relax.transform.annotate_decode_block import AnnotateDecodeInformation
from bitblas.relax.transform.weight_only_propagate import WeightOnlyLayoutPropagation


def test_lop3_transform():
    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def fused_fused_decode3_fused_NT_matmul8_add1(
            lv47: T.Buffer((T.int64(4096), T.int64(512)), "uint32"),
            lv48: T.Buffer((T.int64(4096), T.int64(128)), "float16"),
            p_lv41: T.handle,
            p_output0: T.handle,
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            n = T.int64()
            lv41 = T.match_buffer(p_lv41, (T.int64(1), 1, T.int64(4096)), "float16")
            NT_matmul_intermediate = T.match_buffer(
                p_output0, (T.int64(1), 1, T.int64(4096)), "float16"
            )
            # with T.block("root"):
            decode_intermediate_intermediate = T.alloc_buffer(
                (T.int64(4096), T.int64(4096)), "float16"
            )
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
            for i0, i1, i2, k in T.grid(T.int64(1), 1, T.int64(4096), T.int64(4096)):
                with T.block("NT_matmul"):
                    v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                    T.reads(
                        lv41[v_i0, v_i1, v_k],
                        decode_intermediate_intermediate[v_i2, v_k],
                    )
                    T.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2])
                    with T.init():
                        NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                    NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                        NT_matmul_intermediate[v_i0, v_i1, v_i2]
                        + lv41[v_i0, v_i1, v_k]
                        * decode_intermediate_intermediate[v_i2, v_k]
                    )

        @R.function
        def main(
            lv47: R.Tensor((T.int64(4096), T.int64(512)), dtype="uint32"),
            lv48: R.Tensor((T.int64(4096), T.int64(128)), dtype="float16"),  # type: ignore
            p_lv41: R.Tensor((T.int64(1), 1, T.int64(4096)), dtype="float16"),
        ) -> R.Tensor((1, 4096), dtype="float16"):
            R.func_attr({"Primitive": 1})
            # n = T.int64()
            cls = Before
            with R.dataflow():
                gv = R.call_tir(
                    cls.fused_fused_decode3_fused_NT_matmul8_add1,
                    (lv47, lv48, p_lv41),
                    out_sinfo=R.Tensor((1, 1, 4096), dtype="float16"),
                )
                R.output(gv)
            return gv

    relax_mod = Before
    ref_mod = deepcopy(relax_mod)
    dispatch_target = tvm.target.Target("cuda")
    # input_arrays = get_dummy_input_arrays(relax_mod)

    relax_mod = AnnotateDecodeInformation()(relax_mod)
    with dispatch_target:
        relax_mod = WeightOnlyLayoutPropagation(
            transform_level=0, faster_conversion=False
        )(relax_mod)

    input_tensors = get_dummy_input_arrays(ref_mod["main"], tvm.cpu())

    ref_mod = tvm.tir.transform.MakePackedAPI()(ref_mod)
    ex = relax.build(ref_mod, "llvm")

    device = tvm.cpu(0)
    vm = relax.VirtualMachine(ex, device)
    res = vm["main"](*input_tensors)
    print("ref ", res)

    print(relax_mod)
    relax_mod = tvm.tir.transform.MakePackedAPI()(relax_mod)
    ex = relax.build(relax_mod, "llvm")

    device = tvm.cpu(0)
    vm = relax.VirtualMachine(ex, device)
    res = vm["main"](*input_tensors)
    print("relax ", res)


def test_matmul_transform():
    target = tvm.target.Target("llvm")

    # target = tvm.target.Target("nvidia/nvidia-a100")
    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def fused_fused_decode3_fused_NT_matmul8_add1(
            p_lv41: T.handle,
            lv47: T.Buffer((T.int64(4096), T.int64(4096)), "float16"),
            p_output0: T.handle,
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            n = T.int64()
            lv41 = T.match_buffer(p_lv41, (T.int64(1), 1, T.int64(4096)), "float16")
            NT_matmul_intermediate = T.match_buffer(
                p_output0, (T.int64(1), 1, T.int64(4096)), "float16"
            )

            for i0, i1, i2, k in T.grid(T.int64(1), 1, T.int64(4096), T.int64(4096)):
                with T.block("NT_matmul"):
                    v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                    T.reads(lv41[v_i0, v_i1, v_k], lv47[v_i2, v_k])
                    T.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2])
                    with T.init():
                        NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                    NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                        NT_matmul_intermediate[v_i0, v_i1, v_i2]
                        + lv41[v_i0, v_i1, v_k] * lv47[v_i2, v_k]
                    )

        @R.function
        def main(
            lv47: R.Tensor((T.int64(4096), T.int64(4096)), dtype="float16"),
            p_lv41: R.Tensor((T.int64(1), 1, T.int64(4096)), dtype="float16"),
        ) -> R.Tensor((1, 4096), dtype="float16"):
            R.func_attr({"Primitive": 1})
            # n = T.int64()
            cls = Before
            with R.dataflow():
                gv = R.call_tir(
                    cls.fused_fused_decode3_fused_NT_matmul8_add1,
                    (p_lv41, lv47),
                    out_sinfo=R.Tensor((1, 1, 4096), dtype="float16"),
                )
                R.output(gv)
            return gv

    relax_mod = Before
    ref_mod = deepcopy(relax_mod)
    dispatch_target = tvm.target.Target("cuda")
    # input_arrays = get_dummy_input_arrays(relax_mod)

    relax_mod = AnnotateDecodeInformation()(relax_mod)
    with dispatch_target:
        relax_mod = WeightOnlyLayoutPropagation(
            transform_level=1, faster_conversion=False
        )(relax_mod)

    input_tensors = get_dummy_input_arrays(ref_mod["main"], tvm.cpu())

    ref_mod = tvm.tir.transform.MakePackedAPI()(ref_mod)
    ex = relax.build(ref_mod, "llvm")

    device = tvm.cpu(0)
    vm = relax.VirtualMachine(ex, device)
    res = vm["main"](*input_tensors)
    print("ref ", res)

    print(relax_mod)
    relax_mod = tvm.tir.transform.MakePackedAPI()(relax_mod)
    ex = relax.build(relax_mod, "llvm")

    device = tvm.cpu(0)
    vm = relax.VirtualMachine(ex, device)
    res = vm["main"](*input_tensors)
    print("relax ", res)


def test_dequantize_matmul_transform():
    transform_level = 2
    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def fused_fused_decode3_fused_NT_matmul8_add1(
            lv47: T.Buffer((T.int64(4096), T.int64(512)), "uint32"),
            lv48: T.Buffer((T.int64(4096), T.int64(128)), "float16"),
            p_lv41: T.handle,
            p_output0: T.handle,
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            n = T.int64()
            lv41 = T.match_buffer(p_lv41, (T.int64(1), 1, T.int64(4096)), "float16")
            NT_matmul_intermediate = T.match_buffer(
                p_output0, (T.int64(1), 1, T.int64(4096)), "float16"
            )
            # with T.block("root"):
            decode_intermediate_intermediate = T.alloc_buffer(
                (T.int64(4096), T.int64(4096)), "float16"
            )
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
            for i0, i1, i2, k in T.grid(T.int64(1), 1, T.int64(4096), T.int64(4096)):
                with T.block("NT_matmul"):
                    v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                    T.reads(
                        lv41[v_i0, v_i1, v_k],
                        decode_intermediate_intermediate[v_i2, v_k],
                    )
                    T.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2])
                    with T.init():
                        NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                    NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                        NT_matmul_intermediate[v_i0, v_i1, v_i2]
                        + lv41[v_i0, v_i1, v_k]
                        * decode_intermediate_intermediate[v_i2, v_k]
                    )

        @R.function
        def main(
            lv47: R.Tensor((T.int64(4096), T.int64(512)), dtype="uint32"),
            lv48: R.Tensor((T.int64(4096), T.int64(128)), dtype="float16"),  # type: ignore
            p_lv41: R.Tensor((T.int64(1), 1, T.int64(4096)), dtype="float16"),
        ) -> R.Tensor((1, 4096), dtype="float16"):
            R.func_attr({"Primitive": 1})
            # n = T.int64()
            cls = Before
            with R.dataflow():
                gv = R.call_tir(
                    cls.fused_fused_decode3_fused_NT_matmul8_add1,
                    (lv47, lv48, p_lv41),
                    out_sinfo=R.Tensor((1, 1, 4096), dtype="float16"),
                )
                R.output(gv)
            return gv

    relax_mod = Before
    ref_mod = deepcopy(relax_mod)
    dispatch_target = tvm.target.Target("cuda")
    # input_arrays = get_dummy_input_arrays(relax_mod)

    relax_mod = AnnotateDecodeInformation()(relax_mod)
    with dispatch_target:
        relax_mod = WeightOnlyLayoutPropagation(
            transform_level=transform_level, faster_conversion=False
        )(relax_mod)

    input_tensors = get_dummy_input_arrays(ref_mod["main"], tvm.cpu())

    ref_mod = tvm.tir.transform.MakePackedAPI()(ref_mod)
    ex = relax.build(ref_mod, "llvm")

    device = tvm.cpu(0)
    vm = relax.VirtualMachine(ex, device)
    res = vm["main"](*input_tensors)
    print("ref ", res)

    print(relax_mod)
    relax_mod = tvm.tir.transform.MakePackedAPI()(relax_mod)
    ex = relax.build(relax_mod, "llvm")

    device = tvm.cpu(0)
    vm = relax.VirtualMachine(ex, device)
    res = vm["main"](*input_tensors)
    print("relax ", res)


# test_lop3_transform()
# test_matmul_transform()
test_dequantize_matmul_transform()
