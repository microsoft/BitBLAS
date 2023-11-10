from tvm import relay

@relay.transform.function_pass(opt_level=0)
class AnnotateLadderTensorCore(relay.ExprMutator):
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch

    def transform_function(self, func, mod, ctx):
        return super().visit_function(func)

    def visit_function(self, func):
        visitor = OpVisitor(self.arch)
        visitor.visit(func)
        if visitor.axis is not None:
            print("ladder_config: ", visitor.ladder_config)
            func = func.with_attr("tensorCoreConfig", visitor.axis)
            func = func.with_attr("ladder_config", visitor.ladder_config)
        return super().visit_function(func)

class OpVisitor(relay.ExprVisitor):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        self.axis = None
        self.ladder_config = None

    def visit_call(self, call):
        if call.op.name in ["ladder.perfect_im2col_conv", "ladder.C2DImplicitGemm", "ladder.perfect_matmul", "ladder.perfect_quant_linear"]:
            if self.arch.compute_capability == '80':
                pipleline_stage = 2
            else:
                pipleline_stage = 1
            A_shape = call.args[0].checked_type.shape
            B_shape = call.args[1].checked_type.shape
            if call.op.name == "ladder.perfect_im2col_conv":
                if call.attrs.kernel_layout == "OIHW":
                    K = B_shape[1] * B_shape[3]  
                    N = B_shape[0] * B_shape[2]
                elif call.attrs.kernel_layout == "HWIO":
                    K = B_shape[0] * B_shape[2]
                    N = B_shape[1] * B_shape[3]
                M = A_shape[0] * A_shape[-2]
                
                can_propagate = call.attrs.can_propagate
                is_type_verified = True
                for type in call.type_args:
                    if type.dtype != "float16":
                        is_type_verified = False
                if call.checked_type.dtype != "float16":
                    is_type_verified = False
                
                is_shape_verified = True
                is_shape_verified = (M % 16 ==0 and K % 16 == 0 and N % 16 == 0)
                if is_type_verified and is_shape_verified:
                    num_axis = int(len(call.checked_type.shape))
                    print("num_axis: ", num_axis)
                    self.axis = (num_axis - 2, num_axis - 1)
                    self.ladder_config = (True, True, 2) if can_propagate else (False, False)
            elif call.op.name == "ladder.C2DImplicitGemm":
                if call.attrs.kernel_layout == "OIHW":
                    K = B_shape[1]  
                    N = B_shape[0]
                elif call.attrs.kernel_layout == "HWIO":
                    K = B_shape[0]
                    N = B_shape[1]
                M = A_shape[0]
                num_axis = int(len(call.checked_type.shape))
                self.axis = (num_axis - 2, num_axis - 1)
                self.ladder_config = (False, False)
            elif call.op.name == "ladder.perfect_matmul":
                num_axis = int(len(call.checked_type.shape))
                self.axis = (num_axis - 2, num_axis - 1)
                can_propagate = call.attrs.can_propagate
                self.ladder_config = (True, True, pipleline_stage) if can_propagate else (False, False)
            elif call.op.name == "ladder.perfect_quant_linear":
                num_axis = int(len(call.checked_type.shape))
                self.axis = (num_axis - 2, num_axis - 1)
                can_propagate = call.attrs.can_propagate
                self.ladder_config = (True, True, pipleline_stage) if can_propagate else (False, False)
            elif call.op.name == "ladder.quant_linear":
                self.ladder_config = (False, False)

        return super().visit_call(call)
