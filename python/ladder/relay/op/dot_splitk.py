# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from tvm import relay, target, te, topi
from tvm.relay.op.strategy import wrap_topi_schedule
from tvm.relay import reg
from .utils import compute_matmul_shape

def rel_dotsplitk(arg_types, attrs):
    assert len(arg_types) == 2, "type relation arg number mismatch!"
    a_shape = arg_types[0].shape
    b_shape = arg_types[1].shape
    out_shape = compute_matmul_shape(a_shape, b_shape, attrs["transpose_a"], attrs["transpose_b"])
    out_shape = [attrs["splitk_factor"]] + out_shape
    return relay.TensorType(out_shape, attrs["out_dtype"])

def compute_dotsplitk(attrs, inputs, output_type):
    assert len(inputs) == 2, "input arg number mismatch!"
    out_shape = output_type.shape
    out_dtype = output_type.dtype
    transpose_a = attrs["transpose_a"]
    transpose_b = attrs["transpose_b"]
    splitk_factor = attrs["splitk_factor"]
    A, B = inputs
    k1_size = (A.shape[-2] if transpose_a else A.shape[-1]) // splitk_factor
    k1 = te.reduce_axis((0, k1_size), name="k")
    def fcompute(*args):
        k0 = args[0]
        BS = args[1:-2]
        m, n = args[-2], args[-1]
        k = k1 + k0 * k1_size
        A_args, B_args = [], []
        for i, arg in enumerate(BS):
            if len(A.shape) >= 2 + len(BS) - i:
                A_args.append(arg)
            if len(B.shape) >= 2 + len(BS) - i:
                B_args.append(arg)
        A_args += [k, m] if transpose_a else [m, k]
        B_args += [n, k] if transpose_b else [k, n]
        return te.sum(
            A.__getitem__(tuple(A_args)).astype(out_dtype) * B.__getitem__(tuple(B_args)).astype(out_dtype),
            axis=k1
        )
    C = te.compute(out_shape, fcompute=fcompute, name="T_dotsplitk")
    return [C]

@target.override_native_generic_func("strategy_dotsplitk")
def strategy_dotsplitk(attrs, inputs, out_type, target):
    strategy = relay.op.OpStrategy()
    strategy.add_implementation(
        compute_dotsplitk,
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="dotsplitk.generic",
    )
    return strategy

def register_dot_splitk():
    op_name = "dotsplitk"
    reg.register(op_name, "Compute the matmul operation with partial k axis.")
    op = reg.get(op_name)
    op.set_num_inputs(2)
    op.set_support_level(10)
    op.add_type_rel(op_name + "_rel", rel_dotsplitk)
    op.add_argument("lhs", "Tensor", "The left hand side tensor.")
    op.add_argument("rhs", "Tensor", "The right hand side tensor.")
    op.set_attrs_type_key("DictAttrs")
    reg.register_pattern(op_name, relay.op.OpPattern.OUT_ELEMWISE_FUSABLE)
    reg.register_strategy(op_name, strategy_dotsplitk)

register_dot_splitk()

__all__ = []
