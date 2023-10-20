import tvm
from tvm import relay, ir
import numpy as np


@relay.transform.function_pass(opt_level=0, required=["InferType"])
class LadderFakeQuant(relay.ExprMutator):
    def __init__(self, quant_type=0):
        super().__init__()
        '''
            0: qweight
            1: qweight + scales
            2: qweight + scales + zeros
            
        '''
        self.quant_type = quant_type

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_call(self, call):
        if isinstance(call.op, ir.Op) and call.op.name in ["welder.matmul", "nn.matmul", "nn.dense"]:
            for type in call.type_args:
                if type.dtype != "float16":
                    return super().visit_call(call)

            data = self.visit(call.args[0])
            kernel = self.visit(call.args[1])
            input_shape = call.args[0].checked_type.shape
            kernel_shape = call.args[1].checked_type.shape

            if len(kernel_shape) != 2:
                print("currently do not suppory kernel shape > 2")
                return super().visit_call(call)
            warp_compute_tile_m = 16
            warp_compute_tile_n = 16
            warp_compute_tile_k = 16
            
            if call.op.name in ["welder.matmul", "nn.matmul"]:
                transpose_a, transpose_b = call.attrs.transpose_a, call.attrs.transpose_b
            else:
                transpose_a, transpose_b = False, True

            if transpose_a:
                K, M = input_shape
            else:
                K = input_shape[-1]
                M = 1
                for i in range(len(input_shape) - 1):
                    M *= input_shape[i]

            if transpose_b:
                N, _ = kernel_shape
            else:
                _, N = kernel_shape
            
            out_shape = call.checked_type.shape
            out_dtype = call.checked_type.dtype
            # if the data's node has only one output, we can propagate the layout
            if K % warp_compute_tile_n != 0 or N % warp_compute_tile_k != 0:
                return super().visit_call(call)
            
            # convert kernel to quant format shape -> (N, K // 2), dtype -> int8
            
            quant_kernel_data = tvm.nd.array(
                np.random.randint(low=np.iinfo(np.int8).min,
                                               high=np.iinfo(np.int8).max+1,
                                               size=(int(N), int(K) // 2),
                                               dtype=np.int8
                )
            )
            
            quant_kernel = relay.const(quant_kernel_data)
            other_inputs = []
            if self.quant_type == 1:
                quant_scale_data = tvm.nd.array(
                    np.random.rand(1, int(N)).astype(np.float16)
                )
                quant_scale = relay.const(quant_scale_data)
                other_inputs.append(quant_scale)
            elif self.quant_type == 2:
                quant_scale_data = tvm.nd.array(
                    np.random.rand(1, int(N)).astype(np.float16)
                )
                quant_scale = relay.const(quant_scale_data)
                other_inputs.append(quant_scale)
                quant_zero_data = tvm.nd.array(
                    np.random.rand(1, int(N)).astype(np.float16)
                )
                quant_zero = relay.const(quant_zero_data)
                other_inputs.append(quant_zero)

            attrs = ir.make_node(
                "DictAttrs", out_dtype=out_dtype,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
            )
            q_matmul = relay.Call(
                relay.op.get("ladder.quant_linear"),
                [data, quant_kernel, *other_inputs],
                attrs,
            )
            return q_matmul
       
        return super().visit_call(call)
