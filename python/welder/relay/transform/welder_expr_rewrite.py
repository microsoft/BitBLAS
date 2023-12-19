from tvm import relay, ir
from tvm.relay.dataflow_pattern import DFPatternCallback, wildcard, is_op, is_constant
from typing import List

from tvm.relay.expr import RelayExpr as Expr

class SplitRewriter(DFPatternCallback):
    """This rewriting converts split operations into a sequence of
    strided_slice operations, because codegen is going to be based
    on strided_slices that will define the slice of the tensor that
    will be fed to the consumer.
    """

    def __init__(self):
        super().__init__(require_type=True)
        self.split_in = wildcard()
        self.pattern = is_op("split")(self.split_in)

    @staticmethod
    def get_section_begin_coords(split: relay.Expr) -> List[int]:
        """Currently, the split operator takes an array of indices or an integer
        indicating the number of splits. However, its an array of indices could
        represent both cases, therefore this function just make it an array of
        indices where each index represent the co-ordinate of beginning of each
        section -- defines as section begins.

        Parameters
        ----------
        split : tvm.relay.Expr
            The Relay Call expression for a split operator

        Returns
        -------
        section_begins : List[int]
            A list containing integers corresponding to section
            begins
        """
        indices_or_sections = split.attrs.indices_or_sections
        input_shape = split.args[0].checked_type.shape
        split_axis = split.attrs.axis

        if isinstance(indices_or_sections, ir.container.Array):
            # 0 is the beginning of the first section.
            return [0] + list(indices_or_sections)
        split_axis_len = input_shape[split_axis].value
        section_length = split_axis_len // indices_or_sections.value
        return list(range(0, split_axis_len, section_length))

    def callback(
        self, pre: relay.Expr, post: relay.Expr, node_map: ir.container.Map
    ) -> relay.Expr:
        split_input = post.args[0]
        split_begins = list()
        split_ends = list()
        section_begins_in_split_axis = self.get_section_begin_coords(post)
        for split_cord in section_begins_in_split_axis:
            # first begin is [0, 0, ... , 0]
            begin_shape = [0 for i in range(len(split_input.checked_type.shape))]
            begin_shape[post.attrs.axis] = split_cord
            split_begins.append(begin_shape)

            end_shape = list(split_input.checked_type.shape)
            # Only the split axis coordinate changes
            end_shape[post.attrs.axis] = split_cord
            split_ends.append(end_shape)

        # Coordinates needs to be shifted left because beginning
        # of the next section is the end of the previous
        split_ends = split_ends[1:]
        # Last section end is the shape of the tensor itself.
        split_ends.append(list(split_input.checked_type.shape))

        strided_slices = list()
        for sb, se in zip(split_begins, split_ends):
            strided_slices.append(relay.strided_slice(split_input, sb, se))

        return relay.Tuple(strided_slices)

class SoftmaxRewriter(DFPatternCallback):
    def __init__(self, require_type=False, rewrite_once=False):
        super().__init__(require_type, rewrite_once)
        self.pattern = is_op("nn.softmax")(wildcard())

    def callback(self, pre: relay.Expr, post: relay.Expr, node_map: ir.container.Map) -> relay.Expr:
        reduce_axis = post.attrs.axis
        ma = relay.op.max(post.args[0], reduce_axis, keepdims=True)
        exp = relay.op.exp(relay.op.subtract(post.args[0], ma))
        expsum = relay.op.sum(exp, reduce_axis, keepdims=True)
        out = relay.op.divide(exp, expsum)
        return out

class PowerRewriter(DFPatternCallback):
    def __init__(self, require_type=False, rewrite_once=False):
        super().__init__(require_type, rewrite_once)
        self.pattern = is_op("power")(wildcard(), is_constant())

    def callback(self, pre: relay.Expr, post: relay.Expr, node_map: ir.container.Map) -> relay.Expr:
        power = post.args[1]
        if power.data.asnumpy() == 2:
            x = post.args[0]
            return relay.op.multiply(x, x)
        else:
            return post

class ArgMaxRewriter(DFPatternCallback):
    '''
        Fake argmax rewriter For encoder network, just to make sure that the argmax can be passed
    '''
    def __init__(self, require_type=False, rewrite_once=False):
        super().__init__(require_type, rewrite_once)
        self.pattern = is_op("argmax")(wildcard())

    def callback(self, pre: relay.Expr, post: relay.Expr, node_map: ir.container.Map) -> relay.Expr:
        x = post.args[0]
        axis = post.attrs.axis
        keepdims = post.attrs.keepdims
        dtype = post.checked_type.dtype
        max = relay.max(x, axis=axis, keepdims=keepdims)
        return max


@relay.transform.function_pass(opt_level=0)
class WelderExprRewrite(relay.ExprMutator):
    def __init__(self, enable_softmax=True):
        super().__init__()
        self.enable_softmax = enable_softmax

    def transform_function(self, func, mod, ctx):
        func = SplitRewriter().rewrite(func)
        if self.enable_softmax:
            func = SoftmaxRewriter().rewrite(func)
        func = PowerRewriter().rewrite(func)
        func = ArgMaxRewriter().rewrite(func)
        return self.visit(func)

    def visit_tuple_getitem(self, op):
        if isinstance(op.tuple_value, relay.Tuple):
            return self.visit(op.tuple_value.fields[op.index])
        else:
            return super().visit_tuple_getitem(op)
