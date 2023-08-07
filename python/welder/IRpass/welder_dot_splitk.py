from tvm import relay, ir, target, te, topi
from tvm.relay.op.strategy import wrap_topi_schedule
import numpy as np
from tvm.relay import reg

def rel_dotsplitk(arg_types, attrs):
    assert len(arg_types) == 2, "type relation arg number mismatch!"
    return attrs["ret_type"]

def compute_dotsplitk(attrs, inputs, output_type):
    assert len(inputs) == 2, "input arg number mismatch!"
    out_shape = attrs["ret_type"].shape
    transpose_A = attrs["transpose_A"]
    transpose_B = attrs["transpose_B"]
    splitk_factor = attrs["splitk_factor"]
    A, B = inputs
    k1_size = (A.shape[-2] if transpose_A else A.shape[-1]) // splitk_factor
    k1 = te.reduce_axis((0, k1_size), name="k")
    def fcompute(*args):
        k0 = args[0]
        BS = args[1:-2]
        m, n = args[-2], args[-1]
        k = k1 + k0 * k1_size
        A_args, B_args = [], []
        for i, arg in enumerate(BS):
            if len(A.shape) >= 2 + len(BS) - i:
                A_args.append(arg)
            if len(B.shape) >= 2 + len(BS) - i:
                B_args.append(arg)
        A_args += [k, m] if transpose_A else [m, k]
        B_args += [n, k] if transpose_B else [k, n]
        return te.sum(A.__getitem__(tuple(A_args)) * B.__getitem__(tuple(B_args)), axis=[k1])
    C = te.compute(out_shape, fcompute=fcompute, name="T_dotsplitk")
    return [C]

def schedule_dotsplitk(*args):
    raise NotImplementedError()

@target.override_native_generic_func("strategy_dotsplitk")
def strategy_dotsplitk(attrs, inputs, out_type, target):
    strategy = relay.op.OpStrategy()
    strategy.add_implementation(
        compute_dotsplitk,
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="dotsplitk.generic",
    )
    return strategy

def register_dot_splitk():
    op_name = "dotsplitk"
    reg.register(op_name, "Compute the matmul operation with partial k axis.")
    op = reg.get(op_name)
    op.set_num_inputs(2)
    op.set_support_level(10)
    op.add_type_rel(op_name + "_rel", rel_dotsplitk)
    op.add_argument("lhs", "Tensor", "The left hand side tensor.")
    op.add_argument("rhs", "Tensor", "The right hand side tensor.")
    op.set_attrs_type_key("DictAttrs")
    reg.register_pattern(op_name, relay.op.OpPattern.OUT_ELEMWISE_FUSABLE)
    reg.register_strategy(op_name, strategy_dotsplitk)

register_dot_splitk()

@relay.transform.function_pass(opt_level=0, required=["InferType"])
class WelderDotSplitK(relay.ExprMutator):
    def __init__(self, split_factor=4, size_limit = 80 * 64 * 64):
        super().__init__()
        self.split_factor = split_factor
        self.size_limit = size_limit

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_call(self, call):
        if call.op.name == "nn.matmul" or call.op.name == "nn.dense":
            shape = call.checked_type.shape
            dtype = call.checked_type.dtype
            if call.op.name == "nn.matmul":
                trans_a, trans_b = call.attrs.transpose_a, call.attrs.transpose_b
            else:
                trans_a, trans_b = False, True
            A_shape = call.args[0].checked_type.shape
            k_size = int(A_shape[-2] if trans_a else A_shape[-1])
            if np.prod(shape) <= self.size_limit \
                and k_size % self.split_factor == 0 and k_size // self.split_factor >= 32:

                ret_type = relay.TensorType([self.split_factor] + list(shape), dtype)
                args = [self.visit(arg) for arg in call.args]
                attrs = ir.make_node("DictAttrs",
                                    ret_type=ret_type,
                                    transpose_A=trans_a,
                                    transpose_B=trans_b,
                                    splitk_factor=self.split_factor
                )
                dotsplitk = relay.Call(reg.get("dotsplitk"), args, attrs)
                dot = relay.sum(dotsplitk, axis=0)
                return dot
        return super().visit_call(call)
