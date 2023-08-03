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
        is_type_verified = True
        if call.op.name in ["nn.dense", "nn.batch_matmul", "nn.matmul"]:
            for type in call.type_args:
                if type.dtype != "float16":
                    is_type_verified = False
            if call.checked_type.dtype != "float16":
                is_type_verified = False
            if is_type_verified:
                num_axis = len(call.checked_type.shape)
                assert self.axis is None
                self.axis = (num_axis - 2, num_axis - 1)

        return super().visit_call(call)
