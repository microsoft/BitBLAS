# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from tvm import relay, tir, ir, target, te, topi
from tvm.relay.op.strategy import wrap_topi_schedule
from tvm.relay import reg


def op_relation(arg_types, attrs):
    assert len(arg_types) == 2, "type relation arg number mismatch!"
    a_shape = arg_types[0].shape
    b_shape = arg_types[1].shape

    transpose_a = attrs["transpose_a"]
    transpose_b = attrs["transpose_b"]
    out_dtype = attrs.out_dtype if hasattr(
        attrs, 'out_dtype') and attrs.out_dtype else arg_types[0].dtype
    if transpose_a:
        K, M, wmma_k, wmma_m = a_shape
    else:
        M, K, wmma_m, wmma_k = a_shape

    if transpose_b:
        N, _, wmma_n, _ = b_shape
    else:
        _, N, _, wmma_n = b_shape
    out_shape = [M, N, wmma_m, wmma_n]
    return relay.TensorType(out_shape, out_dtype)


def op_compute(attrs, inputs, output_type):
    assert len(inputs) == 2, "input arg number mismatch!"
    transpose_a = attrs["transpose_a"]
    transpose_b = attrs["transpose_b"]
    out_shape = output_type.shape
    out_dtype = output_type.dtype

    A, B = inputs
    wmma_k_size = A.shape[-2] if transpose_a else A.shape[-1]
    k_size = A.shape[-4] if transpose_a else A.shape[-3]
    k = te.reduce_axis((0, k_size), name="k")
    wmma_k = te.reduce_axis((0, wmma_k_size), name="kk")

    def fcompute(*args):
        m, n, wmma_m, wmma_n = args[-4:]
        A_args = [k, m, wmma_k, wmma_m] if transpose_a else [
            m, k, wmma_m, wmma_k]
        B_args = [n, k, wmma_n, wmma_k] if transpose_b else [
            k, n, wmma_k, wmma_n]
        for arg in reversed(args[:-4]):
            if len(A_args) < len(A.shape):
                if A.shape[len(A.shape) - len(A_args) - 1] == 1:
                    A_args = [0] + A_args
                else:
                    A_args = [arg] + A_args
            if len(B_args) < len(B.shape):
                if B.shape[len(B.shape) - len(B_args) - 1] == 1:
                    B_args = [0] + B_args
                else:
                    B_args = [arg] + B_args
        return te.sum(
            A.__getitem__(tuple(A_args)).astype(out_dtype) *
            B.__getitem__(tuple(B_args)).astype(out_dtype),
            axis=[k, wmma_k]
        )
    C = te.compute(out_shape, fcompute=fcompute, name="T_matmul")

    return [C]


@target.override_native_generic_func("strategy_ladder_perfect_matmul")
def op_strategy(attrs, inputs, out_type, target):
    strategy = relay.op.OpStrategy()
    strategy.add_implementation(
        op_compute,
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="ladder.perfect_matmul.generic",
    )
    return strategy


def op_register():
    op_name = "ladder.perfect_matmul"
    reg.register(op_name)
    op = reg.get(op_name)
    op.set_num_inputs(2)
    op.set_support_level(10)
    op.add_type_rel(op_name + "_rel", op_relation)
    op.add_argument("lhs", "Tensor", "The left hand side tensor.")
    op.add_argument("rhs", "Tensor", "The right hand side tensor.")
    op.set_attrs_type_key("DictAttrs")
    reg.register_pattern(op_name, relay.op.OpPattern.OUT_ELEMWISE_FUSABLE)
    reg.register_strategy(op_name, op_strategy)


op_register()

__all__ = []
