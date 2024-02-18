from typing import Union
import tvm
from tvm import te, tir
from ...block_builder import BlockBuilder
from ...expr import Call, Expr
from .common import register_legalize, _try_convert_to_scalar_const

from bitblas.gpu.intrin.lop3 import tir_interleave_weight

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
