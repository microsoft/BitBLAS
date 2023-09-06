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


@relay.transform.function_pass(opt_level=0, required=["InferType"])
class LadderConvImplicitGemm(relay.ExprMutator):
    def __init__(self, use_async_propagation=False):
        super().__init__()
        self.use_async_propagation = use_async_propagation
        self.node_output_map = {}

    def transform_function(self, func, mod, ctx):
        tracer = UsageTracer()
        tracer.visit(func)
        self.node_output_map = tracer.node_output_map
        return self.visit(func)

    def visit_call(self, call):
        if isinstance(call.op, ir.Op) and call.op.name == "nn.conv2d":
            if call.attrs.groups > 1:
                return super().visit_call(call)
            if (call.attrs.data_layout, call.attrs.kernel_layout) not in [
                ("NHWC", "HWIO")
            ]:
                return super().visit_call(call)
            for type in call.type_args:
                if type.dtype != "float16":
                    return super().visit_call(call)
            warp_compute_tile_m = 16
            warp_compute_tile_n = 16
            warp_compute_tile_k = 16
            input_shape = call.args[0].checked_type.shape
            kernel_shape = call.args[1].checked_type.shape
            N = call.attrs.channels
            if call.attrs.data_layout == "NHWC":
                M = input_shape[0] * input_shape[1] * input_shape[2]
                K = (
                    input_shape[3]
                    * call.attrs.kernel_size[0]
                    * call.attrs.kernel_size[1]
                )
                in_channel = input_shape[3]
                batch_size = input_shape[0]
            if call.attrs.kernel_layout == "HWIO":
                out_channel = kernel_shape[3]

            data = self.visit(call.args[0])
            kernel = self.visit(call.args[1])
            out_shape = call.checked_type.shape
            print("input_shape: ", input_shape)
            print("kernel_shape: ", kernel_shape)
            print("out_shape: ", out_shape)
            print("out_channel, in_channel, batch_size: ", out_channel, in_channel, batch_size)
            # if the data's node has only one output, we can propagate the layout

            if batch_size % warp_compute_tile_m != 0 or in_channel % warp_compute_tile_n != 0 or out_channel % warp_compute_tile_k != 0:
                if batch_size % warp_compute_tile_m != 0 or out_channel % warp_compute_tile_n != 0:
                    print("currently do not suppory m pad or n pad")
                    return super().visit_call(call)
                # print("using implicit gemm")
                if call.attrs.kernel_layout == "OIHW":
                    reshape_kernel = relay.reshape(kernel, [kernel_shape[0], -1])
                elif call.attrs.kernel_layout == "HWIO":
                    reshape_kernel = relay.reshape(kernel, [-1, kernel_shape[3]])
                gemm = relay.Call(
                    relay.op.get("ladder.C2DImplicitGemm"),
                    [data, reshape_kernel],
                    call.attrs,
                )
                return relay.reshape(gemm, out_shape)

            can_propagate = False

            if self.use_async_propagation:
                data_outputs = self.node_output_map[call.args[0]]
                if len(data_outputs) == 1:
                    can_propagate = True
                else:
                    can_propagate = True
                    # print("args[0].op: ", call.args[0].op.name)
                    # for output in self.node_output_map[call.args[0]]:
                    #     print("output: ", output.op.name)
                    #     if output.op.name != "nn.conv2d":
                    #         can_propagate = False
                # if not (M < 128 or N < 128):
                #     can_propagate = False
                print(
                    "data.op.num_outputs: ",
                    len(self.node_output_map[call.args[0]]),
                    "data.name: ",
                    call.args[0].op.name,
                    "call.name: ",
                    call.op.name,
                    "can_propagate: ",
                    can_propagate,
                )

            perfect_data = relay.layout_transform(data, "NHWC", "NHWC16n16c")
            perfect_kernel = relay.layout_transform(kernel, "HWIO", "HWIO16i16o")
            if can_propagate:
                attrs = ir.make_node(
                    "DictAttrs",
                    is_b=False,
                    transpose=False,
                    is_inverse=False,
                )
                perfect_data = relay.Call(
                    relay.op.get("ladder.layout_transform"), [perfect_data], attrs
                )
                attrs = ir.make_node(
                    "DictAttrs",
                    is_b=True,
                    transpose=(call.attrs.kernel_layout == "OIHW"),
                    is_inverse=False,
                )
                perfect_kernel = relay.Call(
                    relay.op.get("ladder.layout_transform"), [perfect_kernel], attrs
                )

            reshape_kernel = relay.reshape(
                perfect_kernel,
                [
                    -1,
                    kernel_shape[3] // warp_compute_tile_k,
                    warp_compute_tile_m,
                    warp_compute_tile_k,
                ],
            )
            # transform data to M, K, wmma_m, wmma_k
            # transform kernel to K, N, wmma_k, wmma_n
            conv2d_attrs = call.attrs
            attrs = ir.make_node(
                "DictAttrs", **conv2d_attrs, can_propagate=can_propagate
            )
            gemm = relay.Call(
                relay.op.get("ladder.perfect_im2col_conv"),
                [perfect_data, reshape_kernel],
                attrs,
            )
            if call.attrs.data_layout == "NHWC":
                out_shape = [
                    out_shape[0] // 16,
                    out_shape[1],
                    out_shape[2],
                    out_shape[3] // 16,
                    16,
                    16,
                ]
                out = relay.reshape(gemm, out_shape)
                # print("out_shape: ", out_shape)
                return relay.layout_transform(out, "NHWC16n16c", "NHWC")
            print("can not cover gemm layout")

        return super().visit_call(call)
