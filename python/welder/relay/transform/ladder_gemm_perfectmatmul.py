from tvm import relay, ir
import numpy as np


class UsageTracer(relay.ExprVisitor):
    def __init__(self):
        super().__init__()
        self.node_output_map = {}

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_call(self, call):
        if isinstance(call.op, ir.Op):
            for arg in call.args:
                if arg not in self.node_output_map:
                    self.node_output_map[arg] = []
                self.node_output_map[arg].append(call)
        return super().visit_call(call)


class PreviousOutputFusibleTracer(relay.ExprVisitor):
    def __init__(self):
        super().__init__()
        self.output_fusible_list = ["nn.conv2d", "nn.dense", "nn.max_pool2d"]
        self.node_previous_fusible_node = {}
        
    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_call(self, call):
        if isinstance(call.op, ir.Op) and call.op.name == "nn.conv2d":
            find_previous_output = False
            current_node = call
            while not find_previous_output:
                # make sure at lease one input node has op attr
                opnode_list = []
                for input_node in current_node.args:
                    if hasattr(input_node, "op"):
                        opnode_list.append(input_node)

                if len(opnode_list) == 0:
                    self.node_previous_fusible_node[call] = None
                    find_previous_output = True
                    
                for input_node in opnode_list:
                    if input_node.op.name in self.output_fusible_list:
                        self.node_previous_fusible_node[call] = input_node
                        find_previous_output = True
                        break
                    else:  
                        current_node = input_node
        return super().visit_call(call)
        

@relay.transform.function_pass(opt_level=0, required=["InferType"])
class LadderPerfectGemmTransform(relay.ExprMutator):
    def __init__(self, use_async_propagation=False):
        super().__init__()
        self.use_async_propagation = use_async_propagation
        self.node_output_map = {}

    def transform_function(self, func, mod, ctx):
        usage_tracer = UsageTracer()
        previous_output_fusible_tracer = PreviousOutputFusibleTracer()
        usage_tracer.visit(func)
        previous_output_fusible_tracer.visit(func)
        self.node_output_map = usage_tracer.node_output_map
        self.node_previous_fusible_node = previous_output_fusible_tracer.node_previous_fusible_node
        return self.visit(func)

    def visit_call(self, call):
        if isinstance(call.op, ir.Op) and call.op.name in ["welder.matmul", "nn.matmul", "nn.dense"]:
            for type in call.type_args:
                if type.dtype != "float16":
                    return super().visit_call(call)

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
            
            data = self.visit(call.args[0])
            kernel = self.visit(call.args[1])
            out_shape = call.checked_type.shape
            out_dtype = call.checked_type.dtype
            # if the data's node has only one output, we can propagate the layout
            if M % warp_compute_tile_m != 0 or K % warp_compute_tile_n != 0 or N % warp_compute_tile_k != 0:
                if M % warp_compute_tile_m != 0 or N % warp_compute_tile_n != 0:
                    print("currently do not suppory m pad or n pad")
                    return super().visit_call(call)
                gemm = relay.Call(
                    relay.op.get("ladder.C2DImplicitGemm"),
                    [data, kernel],
                    call.attrs,
                )
                return relay.reshape(gemm, out_shape)

            can_propagate = False

            if self.use_async_propagation:
                can_propagate = True
            if len(input_shape) > 2:
                data = relay.reshape(
                    data, (K, M)) if transpose_a else relay.reshape(data, (M, K))
            perfect_data = relay.layout_transform(data, "HW", "HW16h16w")
            perfect_kernel = relay.layout_transform(kernel, "HW", "HW16h16w")

            if can_propagate:
                attrs = ir.make_node(
                    "DictAttrs",
                    is_b=False,
                    transpose=transpose_a,
                    is_inverse=False,
                )
                perfect_data = relay.Call(
                    relay.op.get("ladder.layout_transform"), [perfect_data], attrs
                )
                attrs = ir.make_node(
                    "DictAttrs",
                    is_b=True,
                    transpose=transpose_b,
                    is_inverse=False,
                )
                perfect_kernel = relay.Call(
                    relay.op.get("ladder.layout_transform"), [perfect_kernel], attrs
                )

            # transform data to M, K, wmma_m, wmma_k
            # transform kernel to K, N, wmma_k, wmma_n
            attrs = ir.make_node(
                "DictAttrs", out_dtype=out_dtype,
                transpose_a=transpose_a,
                transpose_b=transpose_b, 
                can_propagate=can_propagate
            )
            gemm = relay.Call(
                relay.op.get("ladder.perfect_matmul"),
                [perfect_data, perfect_kernel],
                attrs,
            )
            layout_convert = relay.layout_transform(gemm, "HW16h16w", "HW")
            reshape = relay.reshape(layout_convert, out_shape)
            return reshape

        elif isinstance(call.op, ir.Op) and call.op.name in ["ladder.quant_linear"]:
            kernel = self.visit(call.args[1])
            out_shape = call.checked_type.shape
            out_dtype = call.checked_type.dtype
            
  
            input_shape = call.args[0].checked_type.shape
            kernel_shape = call.args[1].checked_type.shape
            bits = int(call.attrs["bits"])
            warp_compute_tile_m = 16
            warp_compute_tile_n = 16
            warp_compute_tile_k = 32 if out_dtype == "int32" else 16
            data = self.visit(call.args[0])
            transpose_a, transpose_b = call.attrs.transpose_a, call.attrs.transpose_b
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
            
            if M % warp_compute_tile_m != 0 or K % warp_compute_tile_n != 0 or N % warp_compute_tile_k != 0:
                return super().visit_call(call)
            
            if len(kernel_shape) != 2:
                print(f"find call.op {call.op.name} currently do not support kernel shape > 2")
                return super().visit_call(call)


            if len(input_shape) > 2:
                data = relay.reshape(
                    data, (K, M)) if transpose_a else relay.reshape(data, (M, K))
            print("input_shape", input_shape)
            print("kernel_shape", kernel_shape)
            perfect_data = relay.layout_transform(data, "HW", f"HW16h{warp_compute_tile_k}w")
            attrs = ir.make_node(
                "DictAttrs",
                is_b=False,
                transpose=transpose_a,
                is_inverse=False,

            )
            perfect_data = relay.Call(
                relay.op.get("ladder.layout_transform"), [
                    perfect_data], attrs
            )
            
            _compressed_rate = 0
            if out_dtype == "int32":
                _compressed_rate = 32 // (8 // bits)
            elif out_dtype == "float16" or out_dtype == "float32":
                _compressed_rate = 16 // (8 // bits)
            # TODO(v-leiwang3): fake kernel for performance
            perfect_kernel = relay.layout_transform(kernel, "HW", f"HW16h{_compressed_rate}w")

            attrs = ir.make_node(
                "DictAttrs",
                **call.attrs,
                can_propagate=True
            )
            other_inputs = []
            for _ in range(2, len(call.args)):
                other_inputs.append(self.visit(call.args[_]))
            gemm = relay.Call(
                relay.op.get("ladder.perfect_quant_linear"),
                [perfect_data, perfect_kernel, *other_inputs],
                attrs,
            )
            layout_convert = relay.layout_transform(gemm, f"HW{warp_compute_tile_m}h{warp_compute_tile_n}w", "HW")
            reshape = relay.reshape(layout_convert, out_shape)
            return reshape
            
        return super().visit_call(call)
