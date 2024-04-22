# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from tvm import relay, tir, ir, target, te, topi, arith
from tvm.relay.op.strategy import wrap_topi_schedule
from tvm.relay import reg

# the topi reshape compute does not simplify well when lowering, so we implement a new one here

def RavelIndex(indices, shape):
    idx = indices[0]
    for indice, size in zip(indices[1:], shape[1:]):
        idx = idx * size + indice
    return idx

def UnRavelIndex(idx, shape):
    indices = []
    for size in reversed(shape):
        indices.append(tir.indexmod(idx, size))
        idx = tir.indexdiv(idx, size)
    indices = list(reversed(indices))
    return indices

def op_compute(attrs, inputs, output_type):
    out_shape = output_type.shape
    in_shape = inputs[0].shape
    ana = arith.Analyzer()
    def fcompute(*args):
        for i, arg in enumerate(args):
            if isinstance(out_shape[i], tir.IntImm):
                ana.update(arg, arith.ConstIntBound(0, int(out_shape[i] - 1)))
        indices = UnRavelIndex(RavelIndex(args, out_shape), in_shape)
        simplified_indices = [ana.simplify(indice) for indice in indices]
        return inputs[0].__getitem__(tuple(simplified_indices))
    out = te.compute(out_shape, fcompute, "T_reshape")
    return [out]

@target.override_native_generic_func("strategy_welder_reshape")
def op_strategy(attrs, inputs, out_type, target):
    strategy = relay.op.OpStrategy()
    strategy.add_implementation(
        op_compute,
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="welder.reshape.generic",
    )
    return strategy

reg.register_strategy("reshape", op_strategy, level=11)
