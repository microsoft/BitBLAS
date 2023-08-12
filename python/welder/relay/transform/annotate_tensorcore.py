from tvm import relay

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
        if call.op.name in ["nn.dense", "nn.batch_matmul", "nn.matmul", "dotsplitk", "welder.matmul"]:
            M, N = call.checked_type.shape[-2], call.checked_type.shape[-1]
            A_shape = call.args[0].checked_type.shape
            if call.op.name in ["nn.batch_matmul", "nn.matmul"] and call.attrs.transpose_a:
                K = A_shape[-2]
            elif call.op.name == "dotsplitk" and call.attrs["transpose_a"]:
                K = A_shape[-2]
            else:
                K = A_shape[-1]

            is_type_verified = True
            for type in call.type_args:
                if type.dtype != "float16":
                    is_type_verified = False
            if call.checked_type.dtype != "float16":
                is_type_verified = False

            is_shape_verified = True
            if K % 16 != 0 or M % 8 != 0 or N % 8 != 0 or M * N % 256 != 0:
                is_shape_verified = False

            if is_type_verified and is_shape_verified:
                num_axis = int(len(call.checked_type.shape))
                assert self.axis is None
                self.axis = (num_axis - 2, num_axis - 1)

        return super().visit_call(call)
