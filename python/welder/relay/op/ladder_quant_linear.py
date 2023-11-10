from tvm import relay, ir, target, te, topi, tir
from tvm.relay.op.strategy import wrap_topi_schedule
from tvm.relay import reg
from .utils import compute_matmul_shape


def rel_ladder_quant_linear(arg_types, attrs):
    a_shape = arg_types[0].shape
    b_shape = arg_types[1].shape

    assert len(b_shape) == 2, "b_shape should be 2-D"

    a_type = arg_types[0].dtype
    b_type = arg_types[1].dtype
    transpose_a = attrs["transpose_a"]
    transpose_b = attrs["transpose_b"]
    b_type in ["int8", "uint8"]
    if hasattr(attrs, "out_dtype"):
        output_dtype = attrs["out_dtype"]
    else:
        output_dtype = a_type
    K_size = a_shape[-2] if transpose_a else a_shape[-1]
    # assert b dtype is int8
    if transpose_b:
        dequant_b_shape = [b_shape[0], K_size]
    else:
        dequant_b_shape = [K_size, b_shape[1]]

    out_shape = compute_matmul_shape(a_shape, dequant_b_shape, transpose_a, transpose_b)
    return relay.TensorType(out_shape, output_dtype)


def compute_ladder_quant_linear(attrs, inputs, output_type):
    transpose_a = attrs["transpose_a"]
    transpose_b = attrs["transpose_b"]
    out_shape = output_type.shape
    out_dtype = output_type.dtype
    A, B = inputs[:2]
    Scales = None
    Zeros = None
    if len(inputs) == 3:
        Scales = inputs[2]
    elif len(inputs) == 4:
        Scales = inputs[2]
        Zeros = inputs[3]

    group_size = int(attrs["group_size"])
    bits = int(attrs["bits"])
    format = str(attrs["format"])
    assert format == "int", "Only support int format currently"
    n_float_per_i8 = 8 // bits
    K_size = A.shape[-2] if transpose_a else A.shape[-1]
    k = te.reduce_axis((0, K_size), name="k")
    if transpose_b:
        dequant_b_shape = [B.shape[0], K_size]
    else:
        dequant_b_shape = [K_size, B.shape[1]]

    if group_size == -1:
        group_size = K_size

    def _tir_u8_to_int_to_float(
        nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str
    ):
        assert val.dtype == "int8"
        mask = tir.const((1 << nbit) - 1, "int8")
        return ((val >> (pos * nbit).astype("int8")) & mask).astype(dtype)

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

        def decode_func(n, k):
            if transpose_b:
                if bits <= 4:
                    w = _tir_u8_to_int_to_float(
                        bits,
                        B[n, k // n_float_per_i8],
                        k % n_float_per_i8,
                        dtype=A.dtype,
                    )
                else:
                    w = B[n, k // n_float_per_i8].astype(A.dtype)
            else:
                if bits <= 4:    
                    w = _tir_u8_to_int_to_float(
                        bits, B[n // n_float_per_i8, k], n % n_float_per_i8, dtype=A.dtype
                    )
                else:
                    w = B[n // n_float_per_i8, k].astype(A.dtype)

            if Scales is None:
                return w
            elif Zeros is None:
                return w * Scales[0, n]
            else:
                return w * Scales[0, n] + Zeros[0, n]

        B_decode = te.compute(dequant_b_shape, decode_func, name="B_decode")

        return te.sum(
            A.__getitem__(tuple(A_args)).astype(out_dtype)
            * B_decode.__getitem__(tuple(B_args)).astype(out_dtype),
            axis=[k],
        )

    C = te.compute(out_shape, fcompute=fcompute, name="T_quant_linear")
    return [C]


@target.override_native_generic_func("strategy_ladder_quant_linear")
def strategy_ladder_quant_linear(attrs, inputs, out_type, target):
    strategy = relay.op.OpStrategy()
    strategy.add_implementation(
        compute_ladder_quant_linear,
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="ladder.quant_linear.generic",
    )
    return strategy


def register_ladder_quant_linear():
    op_name = "ladder.quant_linear"
    reg.register(op_name, "Customize QuantLinear Function.")
    op = reg.get(op_name)
    op.set_support_level(10)
    op.add_type_rel(op_name + "_rel", rel_ladder_quant_linear)
    op.add_argument("lhs", "Tensor", "The left hand side tensor.")
    op.add_argument("rhs", "Tensor", "The right hand side tensor.")
    op.set_attrs_type_key("DictAttrs")
    reg.register_pattern(op_name, relay.op.OpPattern.OUT_ELEMWISE_FUSABLE)
    reg.register_strategy(op_name, strategy_ladder_quant_linear)


register_ladder_quant_linear()

__all__ = []
