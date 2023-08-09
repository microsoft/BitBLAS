from tvm import relay, ir, target, te, topi
from tvm.relay.op.strategy import wrap_topi_schedule
from tvm.relay import reg

def compute_matmul_shape(a_shape, b_shape, transpose_a, transpose_b):
    a_shape = [int(x) for x in a_shape]
    b_shape = [int(x) for x in b_shape]
    rankdiff = len(a_shape) - len(b_shape)
    if rankdiff > 0:
        b_shape = [1] * rankdiff + b_shape
    elif rankdiff < 0:
        a_shape = [1] * -rankdiff + a_shape
    out_shape = []
    for ax, bx in zip(a_shape[:-2], b_shape[:-2]):
        assert ax == bx or ax == 1 or bx == 1
        out_shape.append(max(ax, bx))
    m_value = a_shape[-1] if transpose_a else a_shape[-2]
    n_value = b_shape[-2] if transpose_b else b_shape[-1]
    out_shape += [m_value, n_value]
    return out_shape

def rel_welder_matmul(arg_types, attrs):
    assert len(arg_types) == 2, "type relation arg number mismatch!"
    a_shape = arg_types[0].shape
    b_shape = arg_types[1].shape
    out_shape = compute_matmul_shape(a_shape, b_shape, attrs["transpose_a"], attrs["transpose_b"])
    return relay.TensorType(out_shape, attrs["out_dtype"])

def compute_welder_matmul(attrs, inputs, output_type):
    assert len(inputs) == 2, "input arg number mismatch!"
    transpose_a = attrs["transpose_a"]
    transpose_b = attrs["transpose_b"]
    out_dtype = output_type.dtype
    A, B = inputs
    out_shape = compute_matmul_shape(A.shape, B.shape, transpose_a, transpose_b)
    K_size = A.shape[-2] if transpose_a else A.shape[-1]
    k = te.reduce_axis((0, K_size), name="k")
    def fcompute(*args):
        m, n = args[-2], args[-1]
        A_args = [k, m] if transpose_a else [m, k]
        B_args = [n, k] if transpose_b else [k, n]
        for arg in reversed(args[:-2]):
            if len(A_args) < len(A.shape):
                if A.shape[len(A.shape) - len(A_args) - 1] == 1:
                    A_args = [0] + A_args
                else:
                    A_args = [arg] + A_args
            if len(B_args) < len(B.shape):
                if B.shape[len(B.shape) - len(B_args) - 1] == 1:
                    B_args = [0] + B_args
                else:
                    B_args = [arg] + B_args
        return te.sum(
            A.__getitem__(tuple(A_args)).astype(out_dtype) * B.__getitem__(tuple(B_args)).astype(out_dtype),
            axis=[k]
        )
    C = te.compute(out_shape, fcompute=fcompute, name="T_matmul")
    return [C]

@target.override_native_generic_func("strategy_welder_matmul")
def strategy_welder_matmul(attrs, inputs, out_type, target):
    strategy = relay.op.OpStrategy()
    strategy.add_implementation(
        compute_welder_matmul,
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="welder.matmul.generic",
    )
    return strategy

def register_welder_matmul():
    op_name = "welder.matmul"
    reg.register(op_name, "Compute the matmul operation with A and B.")
    op = reg.get(op_name)
    op.set_num_inputs(2)
    op.set_support_level(10)
    op.add_type_rel(op_name + "_rel", rel_welder_matmul)
    op.add_argument("lhs", "Tensor", "The left hand side tensor.")
    op.add_argument("rhs", "Tensor", "The right hand side tensor.")
    op.set_attrs_type_key("DictAttrs")
    reg.register_pattern(op_name, relay.op.OpPattern.OUT_ELEMWISE_FUSABLE)
    reg.register_strategy(op_name, strategy_welder_matmul)

register_welder_matmul()
