# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from tvm import relay, tir, ir, target, te, topi, arith
from tvm.relay.op.strategy import wrap_topi_schedule
from tvm.relay import reg
from tvm.tir import IndexMap

# the topi layout_transform compute does not simplify well when lowering, so we implement a new one here
def A_global_16x16_to_shared_load_16x16_layout(i, j):
    thread_id = i * 2 + j // 8
    row = thread_id % 16
    col = (j % 8) + (thread_id // 16) * 8
    return row, col

def B_global_16x16_to_shared_load_16x16_layout(i, j):
    thread_id = i * 2 + j // 8
    row = (i // 8) * 8 + (thread_id % 8)
    col = (j % 8) + 8 * ((thread_id // 8) % 2)
    return row, col


def rel_layout_transform(arg_types, attrs):
    assert len(arg_types) == 1, "type relation arg number mismatch!"
    out_shape = arg_types[0].shape
    out_dtype = arg_types[0].dtype
    return relay.TensorType(out_shape, out_dtype)

def op_compute(attrs, inputs, output_type):
    out_shape = output_type.shape
    is_b = attrs.is_b
    trans = attrs.transpose
    is_inverse = attrs.is_inverse
    instruction_sets = 'cuda'
    if not is_b:
        transform_func = A_global_16x16_to_shared_load_16x16_layout
    else:
        if trans:
            transform_func = B_global_16x16_to_shared_load_16x16_layout
        else:
            transform_func = A_global_16x16_to_shared_load_16x16_layout
    if is_inverse:
        index_map = IndexMap.from_func(A_global_16x16_to_shared_load_16x16_layout)
        inversed_index_map = index_map.inverse([16, 16])
        def fcompute(*args):
            warp_i, warp_j = args[-2:]
            spatial_args = args[:-2]
            new_index = (*spatial_args, *inversed_index_map.map_indices([warp_i, warp_j]))
            return inputs[0][new_index]
    else:
        def fcompute(*args):
            warp_i, warp_j = args[-2:]
            spatial_args = args[:-2]
            permutate_i, permutate_j = transform_func(warp_i, warp_j)
            new_index = (*spatial_args, permutate_i, permutate_j)
            return inputs[0][new_index]
    out = te.compute(out_shape, fcompute, "ladder_layout_transform")
    return [out]

@target.override_native_generic_func("strategy_ladder_layout_transform")
def op_strategy(attrs, inputs, out_type, target):
    strategy = relay.op.OpStrategy()
    strategy.add_implementation(
        op_compute,
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="ladder.layout_transform.generic",
    )
    return strategy

def register_layout_transform():
    op_name = "ladder.layout_transform"
    reg.register(op_name, "Transform the Layout.")
    op = reg.get(op_name)
    op.set_num_inputs(1)
    op.set_support_level(10)
    op.add_type_rel(op_name + "_rel", rel_layout_transform)
    op.add_argument("lhs", "Tensor", "The left hand side tensor.")
    op.set_attrs_type_key("DictAttrs")
    reg.register_pattern(op_name, relay.op.OpPattern.INJECTIVE)
    reg.register_strategy(op_name, op_strategy)

register_layout_transform()

__all__ = []
