# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from tvm import relay, ir, target, te, topi, tir
from tvm.relay.op.strategy import wrap_topi_schedule
from tvm.relay import reg
from .utils import compute_matmul_shape
from tvm.script import tir as T
import tvm

def rel_ladder_perfect_quant_linear(arg_types, attrs):
    a_shape = arg_types[0].shape
    b_shape = arg_types[1].shape

    a_type = arg_types[0].dtype
    b_type = arg_types[1].dtype
    transpose_a = attrs["transpose_a"]
    transpose_b = attrs["transpose_b"]
    out_dtype = attrs.out_dtype if hasattr(
        attrs, 'out_dtype') and attrs.out_dtype else a_type
    if transpose_a:
        K, M, wmma_k, wmma_m = a_shape
    else:
        M, K, wmma_m, wmma_k = a_shape

    if transpose_b:
        N, _, wmma_n, _ = b_shape
    else:
        _, N, _, wmma_n = b_shape
    out_shape = [M, N, wmma_m, wmma_n]
    return relay.TensorType(out_shape, out_dtype)

def compute_ladder_perfect_quant_linear(attrs, inputs, output_type):
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

    group_size = int(attrs['group_size'])
    bits = int(attrs['bits'])
    format = str(attrs['format'])
    if format == "int4b":
        format = "int"
    assert format=="int", "Only support int format currently"
    n_float_per_i8 = 8 // bits
    K_size = A.shape[0] if transpose_a else A.shape[1]
    wmma_k = A.shape[-2] if transpose_a else A.shape[-1]
    k = te.reduce_axis((0, K_size), name="k")
    if transpose_b:
        dequant_b_shape = [*B.shape[0:3], wmma_k]
    else:
        dequant_b_shape = [*B.shape[0:2], wmma_k, B.shape[-1]]
    
    if group_size == -1:
        group_size = K_size
    
    wmma_k_size = A.shape[-2] if transpose_a else A.shape[-1]
    k_size = A.shape[-4] if transpose_a else A.shape[-3]
    k = te.reduce_axis((0, k_size), name="k")
    wmma_k = te.reduce_axis((0, wmma_k_size), name="kk")

    def _tir_u8_to_int_to_float(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
        assert val.dtype == "int8"
        mask = tir.const((1 << nbit) - 1, "int8")
        return ((val >> (pos * nbit).astype("int8")) & mask).astype(dtype)
    
    
    def fcompute(*args):
        m, n, mm, nn = args[-4:]
        A_args = [k, m, wmma_k, mm] if transpose_a else [
            m, k, mm, wmma_k]
        B_args = [n, k, nn, wmma_k] if transpose_b else [
            k, n, wmma_k, nn]
        for arg in reversed(args[:-4]):
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
        
        
        def decode_func(n, k, nn, kk):
            if transpose_b:
                if bits <= 4:
                    w = _tir_u8_to_int_to_float(
                        bits, B[n, k, nn, kk // n_float_per_i8], kk % n_float_per_i8, dtype=A.dtype)
                else:
                    w = B[n, k, nn, kk].astype(A.dtype)
            else:
                if bits <= 4:
                    w = _tir_u8_to_int_to_float(
                        bits, B[n, k, nn // n_float_per_i8, kk], nn % n_float_per_i8, dtype=A.dtype)
                else:
                    w = B[n, k, nn, kk].astype(A.dtype)                

            wmma_m = wmma_n = wmma_k = 16
            if Scales is None:
                return w
            elif Zeros is None:
                return w * Scales[0, n * wmma_n + nn]
            else:
                return w * Scales[0, n * wmma_n + nn] + Zeros[0, n * wmma_n + nn]

        B_decode = te.compute(
            dequant_b_shape,
            decode_func,
            name='B_decode'
        )
        
        return te.sum(
            A.__getitem__(tuple(A_args)).astype(out_dtype) *
            B_decode.__getitem__(tuple(B_args)).astype(out_dtype),
            axis=[k, wmma_k]
        )
    C = te.compute(out_shape, fcompute=fcompute, name="T_perfect_quant_linear")
    return [C]

def compute_ladder_perfect_quant_linear_nf(attrs, inputs, output_type):
    transpose_a = attrs["transpose_a"]
    transpose_b = attrs["transpose_b"]
    out_shape = output_type.shape
    out_dtype = output_type.dtype
    A, B = inputs[:2]
    Scales = None
    LUT = inputs[2]
    Zeros = None
    if len(inputs) == 4:
        Scales = inputs[3]
    elif len(inputs) == 5:
        Scales = inputs[3]
        Zeros = inputs[4]

    group_size = int(attrs['group_size'])
    bits = int(attrs['bits'])
    format = str(attrs['format'])
    assert format=="nf", "Only support int format currently"
    n_float_per_i8 = 8 // bits
    K_size = A.shape[0] if transpose_a else A.shape[1]
    wmma_k = A.shape[-2] if transpose_a else A.shape[-1]
    k = te.reduce_axis((0, K_size), name="k")
    if transpose_b:
        dequant_b_shape = [*B.shape[0:3], wmma_k]
    else:
        dequant_b_shape = [*B.shape[0:2], wmma_k, B.shape[-1]]
    
    if group_size == -1:
        group_size = K_size
    
    wmma_k_size = A.shape[-2] if transpose_a else A.shape[-1]
    k_size = A.shape[-4] if transpose_a else A.shape[-3]
    k = te.reduce_axis((0, k_size), name="k")
    wmma_k = te.reduce_axis((0, wmma_k_size), name="kk")

    def _tir_u8_to_int(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr):
        assert val.dtype == "int8"
        mask = tir.const((1 << nbit) - 1, "int8")
        return (val >> (pos * nbit).astype("int8")) & mask
    
    def fcompute(*args):
        m, n, mm, nn = args[-4:]
        A_args = [k, m, wmma_k, mm] if transpose_a else [
            m, k, mm, wmma_k]
        B_args = [n, k, nn, wmma_k] if transpose_b else [
            k, n, wmma_k, nn]
        for arg in reversed(args[:-4]):
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
        
        
        def decode_func(n, k, nn, kk):
            if transpose_b:
                w = _tir_u8_to_int(
                    bits, B[n, k, nn, kk // n_float_per_i8], kk % n_float_per_i8)
            else:
                w = _tir_u8_to_int(
                    bits, B[n, k, nn // n_float_per_i8, kk], nn % n_float_per_i8)
                
            wmma_m = wmma_n = wmma_k = 16
            if Scales is None:
                return LUT(w)
            elif Zeros is None:
                return LUT(w) * Scales[0, n * wmma_n + nn]
            else:
                return LUT(w) * Scales[0, n * wmma_n + nn] + Zeros[0, n * wmma_n + nn]

        B_decode = te.compute(
            dequant_b_shape,
            decode_func,
            name='B_decode'
        )
        
        return te.sum(
            A.__getitem__(tuple(A_args)).astype(out_dtype) *
            B_decode.__getitem__(tuple(B_args)).astype(out_dtype),
            axis=[k, wmma_k]
        )
    C = te.compute(out_shape, fcompute=fcompute, name="T_perfect_quant_linear")
    return [C]

def compute_ladder_perfect_quant_linear_mxfp(attrs, inputs, output_type):
    transpose_a = attrs["transpose_a"]
    transpose_b = attrs["transpose_b"]
    out_shape = output_type.shape
    out_dtype = output_type.dtype
    A, B = inputs[:2]
    Scales = inputs[2]
    M = int(out_shape[0])
    bits = int(attrs['bits'])
    format = str(attrs['format'])
    assert format=="mxfp", "Only support int format currently"

    n_float_per_i8 = 8 // bits
    K_size = A.shape[0] if transpose_a else A.shape[1]
    wmma_k = A.shape[-2] if transpose_a else A.shape[-1]
    k = te.reduce_axis((0, K_size), name="k")
    if transpose_b:
        dequant_b_shape = [*B.shape[0:3], wmma_k]
    else:
        dequant_b_shape = [*B.shape[0:2], wmma_k, B.shape[-1]]
    
    group_size = 32
    wmma_k_size = A.shape[-2] if transpose_a else A.shape[-1]
    k_size = A.shape[-4] if transpose_a else A.shape[-3]
    k = te.reduce_axis((0, k_size), name="k")
    wmma_k = te.reduce_axis((0, wmma_k_size), name="kk")

    def _tir_u8_to_f8_to_float32(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, scale: tir.PrimExpr = None):
        assert nbit == 8
        # e_f4 == 0 -> e_f32 = 0
        mask = tir.const((1 << nbit) - 1, "uint32")
        f8 = (val >> (pos.astype("uint32") * tir.const(nbit, "uint32"))) & mask
        s = f8 >> tir.const(7, "uint32")
        # e5m2
        e_f8 = (f8 >> tir.const(2, "uint32")) & tir.const(31, "uint32")
        e_f16 = tir.max(e_f8 + scale, tir.const(63, "uint32"))
        m_f8 = e_f8 & tir.const(2, "uint32")
        return tir.reinterpret('float32', (e_f16 | (s << tir.const(8, "uint32"))) << tir.const(23, "uint32") | m_f8)

    def _tir_u8_to_f8_to_bfloat(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, scale: tir.PrimExpr = None):
        assert nbit == 8
        # e_f4 == 0 -> e_f32 = 0
        mask = tir.const((1 << nbit) - 1, "uint32")
        f8 = (val >> (pos.astype("uint32") * tir.const(nbit, "uint32"))) & mask
        s = f8 >> tir.const(7, "uint32")
        # e5m2
        e_f8 = (f8 >> tir.const(2, "uint32")) & tir.const(31, "uint32")
        e_f16 = tir.max(e_f8 + scale, tir.const(63, "uint32"))
        m_f8 = e_f8 & tir.const(2, "uint32")
        return tir.reinterpret('float16', (e_f16 | (s << tir.const(8, "uint32"))) << tir.const(7, "uint32") | m_f8)
    if M == 1:
        decode_func = _tir_u8_to_f8_to_float32
    else:
        decode_func = _tir_u8_to_f8_to_bfloat

    def fcompute(*args):
        m, n, mm, nn = args[-4:]
        A_args = [k, m, wmma_k, mm] if transpose_a else [
            m, k, mm, wmma_k]
        B_args = [n, k, nn, wmma_k] if transpose_b else [
            k, n, wmma_k, nn]
        for arg in reversed(args[:-4]):
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
        
        wmma_n_size = 16
        def B_decode_func(n, k, nn, kk):
            if transpose_b:
                w = decode_func(
                    bits, B[n, k, nn, kk // n_float_per_i8], kk % n_float_per_i8, Scales[(k * wmma_k_size + kk) // group_size, n * wmma_n_size + nn])
            else:
                w = decode_func(
                    bits, B[n, k, nn // n_float_per_i8, kk], nn % n_float_per_i8, Scales[(k * wmma_k_size + kk) // group_size, n * wmma_n_size + nn])
            return w

        B_decode = te.compute(
            dequant_b_shape,
            B_decode_func,
            name='B_decode'
        )
        
        return te.sum(
            A.__getitem__(tuple(A_args)).astype(out_dtype) *
            B_decode.__getitem__(tuple(B_args)).astype(out_dtype),
            axis=[k, wmma_k]
        )
    C = te.compute(out_shape, fcompute=fcompute, name="T_perfect_quant_linear")
    return [C]

def compute_ladder_perfect_quant_linear_mxfp_mxfp(attrs, inputs, output_type):
    transpose_a = attrs["transpose_a"]
    transpose_b = attrs["transpose_b"]
    out_shape = output_type.shape
    out_dtype = output_type.dtype
    A, B = inputs[:2]
    AScales = inputs[2]
    BScales = inputs[3]
    M = int(out_shape[0])
    bits = int(attrs['bits'])
    format = str(attrs['format'])
    assert format=="mxfp", "Only support int format currently"

    n_float_per_i8 = 8 // bits
    K_size = A.shape[0] if transpose_a else A.shape[1]
    wmma_k = A.shape[-2] if transpose_a else A.shape[-1]
    k = te.reduce_axis((0, K_size), name="k")
    if transpose_b:
        dequant_b_shape = [*B.shape[0:3], wmma_k]
    else:
        dequant_b_shape = [*B.shape[0:2], wmma_k, B.shape[-1]]
    dequant_a_shape = [*A.shape[0:2], A.shape[-2], wmma_k]
    group_size = 32
    wmma_k_size = A.shape[-2] if transpose_a else A.shape[-1]
    k_size = A.shape[-4] if transpose_a else A.shape[-3]
    k = te.reduce_axis((0, k_size), name="k")
    wmma_k = te.reduce_axis((0, wmma_k_size), name="kk")

    def _tir_u8_to_f8_to_bfloat(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, scale: tir.PrimExpr = None):
        assert nbit == 8
        # e_f4 == 0 -> e_f32 = 0
        mask = tvm.tir.const((1 << nbit) - 1, "uint32")
        f8 = (val >> (pos.astype("uint32") * tvm.tir.const(nbit, "uint32"))) & mask
        s = f8 >> tvm.tir.const(7, "uint32")
        # e5m2
        e_f8 = (f8 >> tvm.tir.const(2, "uint32")) & tvm.tir.const(31, "uint32")
        e_f16 = T.max(e_f8 + scale, tvm.tir.const(63, "uint32"))
        m_f8 = e_f8 & tvm.tir.const(2, "uint32")
        return tvm.tir.reinterpret("float16", (e_f16 | (s << tvm.tir.const(8, "uint32"))) << tvm.tir.const(7, "uint32") | m_f8)

    decode_func = _tir_u8_to_f8_to_bfloat

    def fcompute(*args):
        m, n, mm, nn = args[-4:]
        A_args =[
            m, k, mm, wmma_k]
        B_args = [n, k, nn, wmma_k] if transpose_b else [
            k, n, wmma_k, nn]
        for arg in reversed(args[:-4]):
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
        wmma_n_size = 16
        wmma_k_size = 16
        def B_decode_func(n, k, nn, kk):
            if transpose_b:
                w = decode_func(
                    bits, B[n, k, nn, kk // n_float_per_i8], kk % n_float_per_i8, BScales[(k * wmma_k_size + kk) // group_size, n * wmma_n_size + nn])
            else:
                w = decode_func(
                    bits, B[n, k, nn // n_float_per_i8, kk], nn % n_float_per_i8, BScales[(k * wmma_k_size + kk) // group_size, n * wmma_n_size + nn])
            return w

        B_decode = te.compute(
            dequant_b_shape,
            B_decode_func,
            name='B_decode'
        )
        
        def A_decode_func(m, k, mm, kk):
            a = w = decode_func(
                    bits, A[m, k, mm, kk // n_float_per_i8], kk % n_float_per_i8, AScales[(k * wmma_k_size + kk) // group_size, m * wmma_n_size + mm])
            return w

        A_decode = te.compute(
            dequant_a_shape,
            A_decode_func,
            name='A_decode'
        )
        
        return te.sum(
            A_decode.__getitem__(tuple(A_args)).astype(out_dtype) *
            B_decode.__getitem__(tuple(B_args)).astype(out_dtype),
            axis=[k, wmma_k]
        )
    C = te.compute(out_shape, fcompute=fcompute, name="T_perfect_quant_linear")
    return [C]

def compute_ladder_perfect_quant_linear_fp(attrs, inputs, output_type):
    transpose_a = attrs["transpose_a"]
    transpose_b = attrs["transpose_b"]
    out_shape = output_type.shape
    out_dtype = output_type.dtype
    A, B = inputs[:2]
    
    bits = int(attrs['bits'])
    format = str(attrs['format'])
    assert format == "fp_e5m2"

    n_float_per_i8 = 8 // bits
    K_size = A.shape[0] if transpose_a else A.shape[1]
    wmma_k = A.shape[-2] if transpose_a else A.shape[-1]
    k = te.reduce_axis((0, K_size), name="k")
    if transpose_b:
        dequant_b_shape = [*B.shape[0:3], wmma_k]
    else:
        dequant_b_shape = [*B.shape[0:2], wmma_k, B.shape[-1]]

    wmma_k_size = A.shape[-2] if transpose_a else A.shape[-1]
    k_size = A.shape[-4] if transpose_a else A.shape[-3]
    k = te.reduce_axis((0, k_size), name="k")
    wmma_k = te.reduce_axis((0, wmma_k_size), name="kk")

    def _tir_u8_to_f8_to_float(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr):
        assert val.dtype == "int8"
        assert nbit == 8
        return tir.reinterpret('float16', tir.Cast('int16', val) << 8)
    def fcompute(*args):
        m, n, mm, nn = args[-4:]
        A_args = [k, m, wmma_k, mm] if transpose_a else [
            m, k, mm, wmma_k]
        B_args = [n, k, nn, wmma_k] if transpose_b else [
            k, n, wmma_k, nn]
        for arg in reversed(args[:-4]):
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
        
        wmma_n_size = 16
        def B_decode_func(n, k, nn, kk):
            if transpose_b:
                w = _tir_u8_to_f8_to_float(
                    bits, B[n, k, nn, kk // n_float_per_i8], kk % n_float_per_i8)
            else:
                w = _tir_u8_to_f8_to_float(
                    bits, B[n, k, nn // n_float_per_i8, kk], nn % n_float_per_i8)
            return w

        B_decode = te.compute(
            dequant_b_shape,
            B_decode_func,
            name='B_decode'
        )
        
        return te.sum(
            A.__getitem__(tuple(A_args)).astype(out_dtype) *
            B_decode.__getitem__(tuple(B_args)).astype(out_dtype),
            axis=[k, wmma_k]
        )
    C = te.compute(out_shape, fcompute=fcompute, name="T_perfect_quant_linear")
    return [C]

@target.override_native_generic_func("strategy_ladder_perfect_quant_linear")
def strategy_ladder_perfect_quant_linear(attrs, inputs, out_type, target):
    strategy = relay.op.OpStrategy()
    if attrs["format"] == "nf":
        strategy.add_implementation(
            compute_ladder_perfect_quant_linear_nf,
            wrap_topi_schedule(topi.generic.schedule_extern),
            name="ladder.perfect_quant_linear.generic",
        )
    elif attrs["format"] == "mxfp":
        if len(inputs) == 3:
            strategy.add_implementation(
                compute_ladder_perfect_quant_linear_mxfp,
                wrap_topi_schedule(topi.generic.schedule_extern),
                name="ladder.perfect_quant_linear.generic",
            )
        elif len(inputs) == 4:
            strategy.add_implementation(
                compute_ladder_perfect_quant_linear_mxfp_mxfp,
                wrap_topi_schedule(topi.generic.schedule_extern),
                name="ladder.perfect_quant_linear.generic",
            )
        else:
            raise ValueError("Invalid number of inputs for mxfp format")
    elif "fp_" in attrs["format"]:
        strategy.add_implementation(
            compute_ladder_perfect_quant_linear_fp,
            wrap_topi_schedule(topi.generic.schedule_extern),
            name="ladder.perfect_quant_linear.generic",
        )
    else:
        strategy.add_implementation(
            compute_ladder_perfect_quant_linear,
            wrap_topi_schedule(topi.generic.schedule_extern),
            name="ladder.perfect_quant_linear.generic",
        )
    return strategy

def register_ladder_perfect_quant_linear():
    op_name = "ladder.perfect_quant_linear"
    reg.register(op_name, "Customize QuantLinear Function.")
    op = reg.get(op_name)
    op.set_support_level(10)
    op.add_type_rel(op_name + "_rel", rel_ladder_perfect_quant_linear)
    op.add_argument("lhs", "Tensor", "The left hand side tensor.")
    op.add_argument("rhs", "Tensor", "The right hand side tensor.")
    op.set_attrs_type_key("DictAttrs")
    reg.register_pattern(op_name, relay.op.OpPattern.OUT_ELEMWISE_FUSABLE)
    reg.register_strategy(op_name, strategy_ladder_perfect_quant_linear)

register_ladder_perfect_quant_linear()

__all__ = []
