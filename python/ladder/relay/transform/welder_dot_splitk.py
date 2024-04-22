# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from tvm import relay, ir
import numpy as np

@relay.transform.function_pass(opt_level=0, required=["InferType"])
class WelderDotSplitK(relay.ExprMutator):
    def __init__(self, split_factor=4, size_limit = 80 * 64 * 64):
        super().__init__()
        self.split_factor = split_factor
        self.size_limit = size_limit

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_call(self, call):
        if isinstance(call.op, ir.Op) and call.op.name in ["welder.matmul", "nn.matmul", "nn.dense"]:
            shape = call.checked_type.shape
            dtype = call.checked_type.dtype
            if call.op.name in ["welder.matmul", "nn.matmul"]:
                trans_a, trans_b = call.attrs.transpose_a, call.attrs.transpose_b
            else:
                trans_a, trans_b = False, True
            A_shape = call.args[0].checked_type.shape
            k_size = int(A_shape[-2] if trans_a else A_shape[-1])
            if np.prod(shape) <= self.size_limit \
                and k_size % self.split_factor == 0 and k_size // self.split_factor >= 32:

                args = [self.visit(arg) for arg in call.args]
                attrs = ir.make_node("DictAttrs",
                                    out_dtype=dtype,
                                    transpose_a=trans_a,
                                    transpose_b=trans_b,
                                    splitk_factor=self.split_factor
                )
                dotsplitk = relay.Call(relay.op.get("dotsplitk"), args, attrs)
                dot = relay.sum(dotsplitk, axis=0)
                return dot
        return super().visit_call(call)
