from tvm import relay, ir
import numpy as np
import logging 

logger = logging.getLogger(__name__)

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
class LadderRewriteInceptionLayout(relay.ExprMutator):
    def __init__(self):
        super().__init__()
        self.node_output_map = {}

    def transform_function(self, func, mod, ctx):
        tracer = UsageTracer()
        tracer.visit(func)
        self.node_output_map = tracer.node_output_map
        return self.visit(func)

    
    def visit_call(self, call):
        if isinstance(call.op, ir.Op) and call.op.name == "add":
            lhs = call.args[0]
            rhs = call.args[1]
            # if lhs or rhs is constant
            lhs_is_constant = isinstance(lhs, relay.expr.Constant)
            rhs_is_constant = isinstance(rhs, relay.expr.Constant)
            if lhs_is_constant or rhs_is_constant:
                logger.debug('lhs or rhs is constant')
            else:
                # if one is relu or maxpool
                lhs_is_relu = hasattr(lhs,'op') and isinstance(lhs.op, ir.Op) and lhs.op.name == "nn.relu"
                rhs_is_relu = hasattr(rhs,'op') and isinstance(rhs.op, ir.Op) and rhs.op.name == "nn.relu"
                lhs_is_maxpool = hasattr(lhs,'op') and isinstance(lhs.op, ir.Op) and lhs.op.name == "nn.max_pool2d"
                rhs_is_maxpool = hasattr(rhs,'op') and isinstance(rhs.op, ir.Op) and rhs.op.name == "nn.max_pool2d"
                if lhs_is_relu or rhs_is_relu or lhs_is_maxpool or rhs_is_maxpool:
                    the_relu_or_maxpool = lhs if lhs_is_relu or lhs_is_maxpool else rhs
                    def detect_layout_transform(node):
                        output_nodes = self.node_output_map[node]
                        assert len(output_nodes) == 2
                        # get the output_node that is not node
                        the_other = output_nodes[0] if output_nodes[0] != call else output_nodes[1]
                        # detect if the_other is layout_transform
                        if isinstance(the_other.op, ir.Op) and the_other.op.name == "layout_transform":
                            return the_other
                        return None
                    
                    layout_transform = detect_layout_transform(the_relu_or_maxpool)
                    if layout_transform:
                        the_other = lhs if rhs_is_relu or rhs_is_maxpool else rhs
                        the_relu_or_maxpool_transform = relay.layout_transform(the_relu_or_maxpool, "NHWC", "NHWC16n16c")
                        the_other_transform = relay.layout_transform(the_other, "NHWC", "NHWC16n16c")
                        layout_transform_outputs = self.node_output_map[layout_transform]
                        assert len(layout_transform_outputs) == 1, "layout_transform should only have one output for now"
                        layout_transform_output = layout_transform_outputs[0]
                        if isinstance(layout_transform_output.op, ir.Op) and layout_transform_output.op.name == "ladder.layout_transform":
                            # insert a same layout_transform and an inversed layout_transform
                            attrs = ir.make_node(
                                "DictAttrs",
                                is_b=layout_transform_output.attrs.is_b,
                                transpose=layout_transform_output.attrs.transpose,
                                is_inverse=layout_transform_output.attrs.is_inverse,
                            )
                            transform_data = relay.Call(
                                relay.op.get("ladder.layout_transform"), [the_relu_or_maxpool_transform], attrs
                            )
                            
                            attrs = ir.make_node(
                                "DictAttrs",
                                is_b=layout_transform_output.attrs.is_b,
                                transpose=layout_transform_output.attrs.transpose,
                                is_inverse=True,
                            )
                            the_relu_or_maxpool_transform = relay.Call(
                                relay.op.get("ladder.layout_transform_inverse"), [transform_data], attrs
                            )
                            
                        add_node = relay.add(the_relu_or_maxpool_transform, the_other_transform)
                        return super().visit_call(relay.layout_transform(add_node, "NHWC16n16c", "NHWC"))
                
        return super().visit_call(call)
