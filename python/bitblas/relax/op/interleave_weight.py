# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from tvm.relax.block_builder import BlockBuilder
from tvm.relax.expr import Call, Expr
from tvm.relax.transform.legalize_ops.common import register_legalize

from bitblas.ops.impl import tir_interleave_weight


@register_legalize("bitblas.interleave_weight")
def _interleave_weight(bb: BlockBuilder, call: Call) -> Expr:
    nbits = call.attrs.nbits
    target_dtype = call.attrs.target_dtype
    out_dtype = call.attrs.out_dtype

    return bb.call_te(
        tir_interleave_weight(nbits, target_dtype, out_dtype),
        call.args[0],
        primfunc_name_hint="interleave_weight",
    )


__all__ = ["_interleave_weight"]
