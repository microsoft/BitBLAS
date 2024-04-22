# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from tvm import relay

from .utils import check_tensor_core_valid_shape

@relay.transform.function_pass(opt_level=0)
class AnnotateTensorCore(relay.ExprMutator):
    def __init__(self):
        super().__init__()

    def transform_function(self, func, mod, ctx):
        return super().visit_function(func)

    def visit_function(self, func):
        visitor = OpVisitor()
        visitor.visit(func)
        if visitor.axis is not None:
            func = func.with_attr("tensorCoreConfig", visitor.axis)
        return super().visit_function(func)

class OpVisitor(relay.ExprVisitor):
    def __init__(self):
        super().__init__()
        self.axis = None

    def visit_call(self, call):
        if call.op.name in ["nn.dense", "nn.batch_matmul", "nn.matmul", "dotsplitk", "welder.matmul", "welder.C2DImplicitGemm", "welder.C1DImplicitGemm", "ladder.quant_linear"]:
            M, N = call.checked_type.shape[-2], call.checked_type.shape[-1]
            A_shape = call.args[0].checked_type.shape
            B_shape = call.args[1].checked_type.shape
            if call.op.name == "welder.C2DImplicitGemm":
                K = B_shape[1] if call.attrs.kernel_layout == "OIHW" else B_shape[0]
            elif call.op.name == "welder.C1DImplicitGemm":
                K = B_shape[1] if call.attrs.kernel_layout == "OIW" else B_shape[0]
            elif call.op.name in ["nn.batch_matmul", "nn.matmul"] and call.attrs.transpose_a:
                K = A_shape[-2]
            elif call.op.name in ["welder.matmul", "dotsplitk"] and call.attrs["transpose_a"]:
                K = A_shape[-2]
            else:
                K = A_shape[-1]

            is_type_verified = True
            for type in call.type_args:
                if type.dtype != "float16":
                    is_type_verified = False
            if call.checked_type.dtype != "float16":
                is_type_verified = False

            is_shape_verified = check_tensor_core_valid_shape(M, N, K)

            if is_type_verified and is_shape_verified:
                num_axis = int(len(call.checked_type.shape))
                assert self.axis is None
                self.axis = (num_axis - 2, num_axis - 1)

        return super().visit_call(call)
