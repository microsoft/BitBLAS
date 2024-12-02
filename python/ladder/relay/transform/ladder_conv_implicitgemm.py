import ladder
from tvm import relay, ir
import numpy as np
from tvm.tir import IndexMap


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
                    if not (hasattr(input_node, "op") and hasattr(input_node.op, "name")):
                        current_node = input_node
                        continue
                    if input_node.op.name in self.output_fusible_list:
                        self.node_previous_fusible_node[call] = input_node
                        find_previous_output = True
                        break
                    else:  
                        current_node = input_node
        return super().visit_call(call)
        

@relay.transform.function_pass(opt_level=0, required=["InferType"])
class LadderConvImplicitGemm(relay.ExprMutator):
    def __init__(self, use_async_propagation=False, arch=None):
        super().__init__()
        self.use_async_propagation = use_async_propagation
        self.node_output_map = {}
        if arch==None:
            self.arch = ladder.arch.__getattribute__('cuda')()
        else:
            self.arch=arch

    def transform_function(self, func, mod, ctx):
        usage_tracer = UsageTracer()
        previous_output_fusible_tracer = PreviousOutputFusibleTracer()
        usage_tracer.visit(func)
        previous_output_fusible_tracer.visit(func)
        self.node_output_map = usage_tracer.node_output_map
        self.node_previous_fusible_node = previous_output_fusible_tracer.node_previous_fusible_node
        return self.visit(func)

    def visit_call(self, call):
        if isinstance(call.op, ir.Op) and call.op.name == "nn.conv2d":
            print("call.op.name: ", call.op.name)
            print("self.arch.platform: ", self.arch.platform)
            print("self.use_async_propagation: ", self.use_async_propagation)
            if call.attrs.groups > 1:
                print("groups > 1, do not fuse")
                return super().visit_call(call)
            if (call.attrs.data_layout, call.attrs.kernel_layout) not in [
                ("NHWC", "HWIO")
            ]:
                print("data_layout or kernel_layout is not supported")
                return super().visit_call(call)
            for type in call.type_args:
                if type.dtype != "float16":
                    print("dtype is not float16")
                    return super().visit_call(call)
            print("the op can be processed")
            # should pass if previous compute node is a conv node with 3 channels (this sort conv has performance issue when we use layout propagate)
            # Vulnerable Networks: VGG
            previous_fusible_node = self.node_previous_fusible_node[call]
                        
            def check_not_fusbile(node):
                if node is None:
                    return False
                if node.op.name != "nn.conv2d":
                    return False
                input_shape = node.args[0].checked_type.shape
                if node.attrs.data_layout == "NHWC":
                    in_channel = input_shape[-1]
                else:
                    raise NotImplementedError()
                if in_channel % 16 == 0:
                    return False
                
                return True

            if check_not_fusbile(previous_fusible_node) and self.arch.platform == "cuda":
                print("this is not a fusbile node")
                return super().visit_call(call)
            
            print("the op is fusable")

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

            print("get out_shape")
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
                can_propagate = True

            perfect_data = relay.layout_transform(data, "NHWC", "NHWC16n16c")
            perfect_kernel = relay.layout_transform(kernel, "HWIO", "HWIO16i16o")
            print("can_propagate: ", can_propagate)
            if self.arch.platform == "cuda":
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
            
            elif "ROCm" in self.arch.platform:
                print("using rocm and the can_propagate is: ", can_propagate)
                if can_propagate:
                    # todo(leiwang): this is a trick to evaluate the correctness of the layout transform
                    def thread_id_shared_access_64x4_to_16x16_layout_A(thread_id, local_id):
                        i = thread_id % 16
                        j = (thread_id // 16) * 4 + local_id
                        return i, j

                    def thread_id_shared_access_64x4_to_16x16_layout_B(thread_id, local_id):
                        i = local_id + (thread_id // 16) * 4
                        j = thread_id % 16
                        return i, j

                    def a_prmt_func(i, j):
                        _id = i * 16 + j
                        thread_id = _id // 4
                        local_id = _id % 4
                        return thread_id_shared_access_64x4_to_16x16_layout_A(thread_id, local_id)

                    def b_prmt_func(i, j):
                        _id = i * 16 + j
                        thread_id = _id // 4
                        local_id = _id % 4
                        return thread_id_shared_access_64x4_to_16x16_layout_B(thread_id, local_id)

                    
                    attrs = ir.make_node(
                        "DictAttrs",
                        is_b=False,
                        transpose=False,
                        is_inverse=False,
                        transform_func=IndexMap.from_func(a_prmt_func),
                    )
                    perfect_data = relay.Call(
                        relay.op.get("ladder.layout_transform"), [perfect_data], attrs
                    )
                    transpose_b = (call.attrs.kernel_layout == "HWIO")
                    assert transpose_b == True, "currently only support transpose_b == True"
                    attrs = ir.make_node(
                        "DictAttrs",
                        is_b=True,
                        transpose=transpose_b,
                        is_inverse=False,
                        transform_func=IndexMap.from_func(a_prmt_func)
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
                # set output data type to float32
                conv2d_attrs = call.attrs
                attrs = ir.make_node(
                    "DictAttrs",
                    strides=conv2d_attrs.strides,
                    padding=conv2d_attrs.padding,
                    dilation=conv2d_attrs.dilation,
                    data_layout=conv2d_attrs.data_layout,
                    kernel_layout=conv2d_attrs.kernel_layout,
                    kernel_size=conv2d_attrs.kernel_size,
                    out_dtype='float32',
                    can_propagate=can_propagate
                )
                gemm = relay.Call(
                    relay.op.get("ladder.perfect_im2col_conv"),
                    [perfect_data, reshape_kernel],
                    attrs,
                )
                gemm = relay.cast(gemm, "float16")
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
            

        return super().visit_call(call)
