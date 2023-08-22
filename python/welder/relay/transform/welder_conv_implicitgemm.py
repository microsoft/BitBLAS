from tvm import relay, ir
import numpy as np

@relay.transform.function_pass(opt_level=0, required=["InferType"])
class WelderConvImplicitGemm(relay.ExprMutator):
    def __init__(self):
        super().__init__()

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_call(self, call):
        if isinstance(call.op, ir.Op) and call.op.name == "welder.matmul":
            if call.checked_type.dtype != 'float16':
                return super().visit_call(call)
            transpose_a = bool(call.attrs["transpose_a"])
            transpose_b = bool(call.attrs["transpose_b"])
            A_shape = call.args[0].checked_type.shape
            B_shape = call.args[1].checked_type.shape
            A, B = self.visit(call.args[0]), self.visit(call.args[1])
            if len(A_shape) > 2 and len(B_shape) == 2 and not transpose_a and np.prod(A_shape[:-2]) > 1:
                M = np.prod(A_shape[:-1])
                N = call.checked_type.shape[-1]
                K = A_shape[-1]
                if K % 16 != 0 or M % 8 != 0 or N % 8 != 0 or M * N % 256 != 0:
                    return super().visit_call(call)
                reshape_A = relay.reshape(A, [M, K])
                C = relay.nn.matmul(reshape_A, B, out_dtype=call.checked_type.dtype, transpose_a=transpose_a, transpose_b=transpose_b)
                reshape_C = relay.reshape(C, call.checked_type.shape)
                return reshape_C
            return super().visit_call(call)
        if isinstance(call.op, ir.Op) and call.op.name == "nn.conv2d":
            if call.attrs.groups > 1:
                return super().visit_call(call)
            if (call.attrs.data_layout, call.attrs.kernel_layout) not in [("NCHW", "OIHW"), ("NHWC", "HWIO")]:
                return super().visit_call(call)
            for type in call.type_args:
                if type.dtype != "float16":
                    return super().visit_call(call)
            output_shape = call.checked_type.shape
            input_shape = call.args[0].checked_type.shape
            kernel_shape = call.args[1].checked_type.shape
            N = call.attrs.channels
            if call.attrs.data_layout == "NCHW":
                M = output_shape[0] * output_shape[2] * output_shape[3]
                K = input_shape[1] * call.attrs.kernel_size[0] * call.attrs.kernel_size[1]
            elif call.attrs.data_layout == "NHWC":
                M = output_shape[0] * output_shape[1] * output_shape[2]
                K = input_shape[3] * call.attrs.kernel_size[0] * call.attrs.kernel_size[1]
            if K % 16 != 0 or M % 8 != 0 or N % 8 != 0 or M * N % 256 != 0:
                return super().visit_call(call)

            data = self.visit(call.args[0])
            kernel = self.visit(call.args[1])
            if call.attrs.kernel_layout == "OIHW":
                reshape_kernel = relay.reshape(kernel, [kernel_shape[0], -1])
            elif call.attrs.kernel_layout == "HWIO":
                reshape_kernel = relay.reshape(kernel, [-1, kernel_shape[3]])
            gemm = relay.Call(relay.op.get("welder.C2DImplicitGemm"), [data, reshape_kernel], call.attrs)
            out_shape = call.checked_type.shape
            if call.attrs.data_layout == "NCHW":
                # C, NHW -> C, N, H, W -> N, C, H, W
                reshape = relay.reshape(gemm, [out_shape[1], out_shape[0], out_shape[2], out_shape[3]])
                transpose = relay.transpose(reshape, [1, 0, 2, 3])
                return relay.reshape(transpose, out_shape)
            else:
                # NHW, C -> N, H, W, C
                return relay.reshape(gemm, out_shape)

        return super().visit_call(call)
