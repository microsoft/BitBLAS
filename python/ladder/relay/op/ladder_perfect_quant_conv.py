# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from tvm import relay, tir, ir, target, te, topi
from tvm.relay.op.strategy import wrap_topi_schedule
from tvm.relay import reg
from logging import log


def get_pad(pad):
    pad = list(pad)
    if len(pad) == 4:
        return pad
    elif len(pad) == 2:
        return pad * 2
    elif len(pad) == 1:
        return pad * 4
    else:
        raise ValueError(pad)


def op_relation(arg_types, attrs):
    assert attrs.data_layout in ["NHWC"]
    assert attrs.kernel_layout in ["HWIO"]

    a_shape = arg_types[0].shape
    b_shape = arg_types[1].shape
    if attrs.data_layout == "NHWC":
        batch, in_h, in_w, _, wmma_m, wmma_k = a_shape
    # for perfect quant, we use nt layout, the kernel is (oc, ic*kh*kw, wmma_n, wmma_k)
    out_c = b_shape[-4]
    wmma_n = b_shape[-2]
    
    pad_top, pad_left, pad_bottom, pad_right = get_pad(attrs.padding)
    dilation_h, dilation_w = attrs.dilation
    stride_h, stride_w = attrs.strides
    kernel_h, kernel_w = attrs.kernel_size
    out_h = (
        in_h + pad_top + pad_bottom - 1 - (kernel_h - 1) * dilation_h
    ) // stride_h + 1
    out_w = (
        in_w + pad_left + pad_right - 1 - (kernel_w - 1) * dilation_w
    ) // stride_w + 1
    out_dtype = attrs.out_dtype if attrs.out_dtype else arg_types[0].dtype
    if attrs.data_layout == "NHWC":
        out_shape = [batch * out_h * out_w, out_c, wmma_m, wmma_n]
    return relay.TensorType(out_shape, out_dtype)


def op_compute_int(attrs, inputs, output_type):
    assert len(inputs[0].shape) == 6, "data arg number mismatch!"
    
    data, kernel = inputs[:2]
    out_dtype = output_type.dtype
    Scales = None
    Zeros = None
    if len(inputs) == 3:
        Scales = inputs[2]
    elif len(inputs) == 4:
        Scales = inputs[2]
        Zeros = inputs[3]
    
    bits = int(attrs['bits'])
    format = str(attrs['format'])
    assert format=="int", "Only support int format currently"
    n_float_per_i8 = 8 // bits
    def _tir_u8_to_u8(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
        assert val.dtype == "int8"
        mask = tir.const((1 << nbit) - 1, "int8")
        return ((val >> (pos * nbit).astype("int8")) & mask).astype(dtype)
    
    def kernel_decode_func(n, k, nn, kk):
        w = _tir_u8_to_u8(bits, kernel[n, k, nn, kk // n_float_per_i8], kk % n_float_per_i8, data.dtype)
        return w
    
    kernel_decode = te.compute(
        [kernel.shape[0], kernel.shape[1], kernel.shape[2], kernel.shape[3] * n_float_per_i8],
        kernel_decode_func,
        name='B_decode'
    )
    # padding
    pad_top, pad_left, pad_bottom, pad_right = get_pad(attrs.padding)
    pad_shape = list(data.shape)
    if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
        pad_value = tir.const(0.0, data.dtype)
        if attrs.data_layout == "NHWC":
            pad_shape[1] += pad_top + pad_bottom
            pad_shape[2] += pad_left + pad_right
            pad = te.compute(
                pad_shape,
                lambda n, h, w, c, nn, cc: te.if_then_else(
                    tir.all(
                        h >= pad_bottom,
                        w >= pad_left,
                        h < pad_shape[1] - pad_top,
                        w < pad_shape[2] - pad_right,
                    ),
                    data[n, h - pad_bottom, w - pad_left, c, nn, cc],
                    pad_value,
                ),
                name="pad",
            )
        else:
            log.error("currently do not support data_layout != NHWC")
    else:
        pad = data

    # compute GEMM
    if attrs.data_layout == "NHWC":
        batch, in_h, in_w, in_c, wmma_m, wmma_k = data.shape
    else:
        log.error("currently do not support data_layout != NHWC")

    # for perfect quant, we use nt layout, the kernel is (oc, ic*kh*kw, wmma_n, wmma_k)
    out_c = kernel.shape[-4]
    wmma_n = kernel.shape[-2]
    kernel_h, kernel_w = attrs.kernel_size
    stride_h, stride_w = attrs.strides
    dilation_h, dilation_w = attrs.dilation
    k_size = kernel_h * kernel_w * in_c
    k_axis = te.reduce_axis((0, k_size), name="k")
    wk_axis = te.reduce_axis((0, wmma_k), name="wk")
    out_h = (
        in_h + pad_top + pad_bottom - 1 - (kernel_h - 1) * dilation_h
    ) // stride_h + 1
    out_w = (
        in_w + pad_left + pad_right - 1 - (kernel_w - 1) * dilation_w
    ) // stride_w + 1
    n_size = out_h * out_w * batch

    if attrs.data_layout == "NHWC":
        if attrs.kernel_layout == "HWIO":
            data = te.compute(
                [n_size, k_size, wmma_m, wmma_k],
                lambda n, k, nn, kk: pad[
                    n // (out_h * out_w),
                    (n % (out_h * out_w) // out_w) * stride_h
                    + (k // (kernel_w * in_c)) * dilation_h,
                    (n % out_w) * stride_w + (k // in_c % kernel_w) * dilation_w,
                    k % in_c,
                    nn,
                    kk,
                ],
                name="data",
            )
        C = te.compute(
            [n_size, out_c, wmma_m, wmma_n],
            lambda i, j, ii, jj: te.sum(data[i, k_axis, ii, wk_axis].astype(out_dtype) * kernel_decode[j, k_axis, jj, wk_axis].astype(out_dtype), axis=[k_axis, wk_axis]),
            "T_conv",
        )

    return [C]

def op_compute_mxfp(attrs, inputs, output_type):
    assert len(inputs[0].shape) == 6, "data arg number mismatch!"
    
    data, kernel, Scales = inputs[:3]
    out_dtype = output_type.dtype
    
    group_size = 32
    bits = int(attrs['bits'])
    format = str(attrs['format'])
    assert format=="mxfp", "Only support mxfp format currently"
    n_float_per_i8 = 8 // bits

    def _tir_u8_to_f8_to_float(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str, scale: tir.PrimExpr = None):
        assert nbit == 8
        # e_f4 == 0 -> e_f32 = 0
        mask = tir.const((1 << nbit) - 1, "uint32")
        f8 = (val >> (pos.astype("uint32") * tir.const(nbit, "uint32"))) & mask
        s = f8 >> tir.const(7, "uint32")
        # e5m2
        e_f8 = (f8 >> tir.const(2, "uint32")) & tir.const(31, "uint32")
        e_f16 = tir.max(e_f8 + scale, tir.const(63, "uint32"))
        m_f8 = e_f8 & tir.const(2, "uint32")
        return tir.reinterpret(dtype, (((e_f16 | (s << tir.const(8, "uint32"))) << 2) | m_f8) << tir.const(25, "uint32"))

    def kernel_decode_func(n, k, nn, kk):
        w = _tir_u8_to_f8_to_float(bits, kernel[n, k, nn, kk // n_float_per_i8], kk % n_float_per_i8, 'float16', Scales[(k * 16 + kk) // group_size, n * 16 + nn])
        return w

    kernel_decode = te.compute(
        [kernel.shape[0], kernel.shape[1], kernel.shape[2], kernel.shape[3] * n_float_per_i8],
        kernel_decode_func,
        name='B_decode'
    )
    # padding
    pad_top, pad_left, pad_bottom, pad_right = get_pad(attrs.padding)
    pad_shape = list(data.shape)
    if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
        pad_value = tir.const(0.0, data.dtype)
        if attrs.data_layout == "NHWC":
            pad_shape[1] += pad_top + pad_bottom
            pad_shape[2] += pad_left + pad_right
            pad = te.compute(
                pad_shape,
                lambda n, h, w, c, nn, cc: te.if_then_else(
                    tir.all(
                        h >= pad_bottom,
                        w >= pad_left,
                        h < pad_shape[1] - pad_top,
                        w < pad_shape[2] - pad_right,
                    ),
                    data[n, h - pad_bottom, w - pad_left, c, nn, cc],
                    pad_value,
                ),
                name="pad",
            )
        else:
            log.error("currently do not support data_layout != NHWC")
    else:
        pad = data

    # compute GEMM
    if attrs.data_layout == "NHWC":
        batch, in_h, in_w, in_c, wmma_m, wmma_k = data.shape
    else:
        log.error("currently do not support data_layout != NHWC")

    # for perfect quant, we use nt layout, the kernel is (oc, ic*kh*kw, wmma_n, wmma_k)
    out_c = kernel.shape[-4]
    wmma_n = kernel.shape[-2]
    kernel_h, kernel_w = attrs.kernel_size
    stride_h, stride_w = attrs.strides
    dilation_h, dilation_w = attrs.dilation
    k_size = kernel_h * kernel_w * in_c
    k_axis = te.reduce_axis((0, k_size), name="k")
    wk_axis = te.reduce_axis((0, wmma_k), name="wk")
    out_h = (
        in_h + pad_top + pad_bottom - 1 - (kernel_h - 1) * dilation_h
    ) // stride_h + 1
    out_w = (
        in_w + pad_left + pad_right - 1 - (kernel_w - 1) * dilation_w
    ) // stride_w + 1
    n_size = out_h * out_w * batch

    if attrs.data_layout == "NHWC":
        if attrs.kernel_layout == "HWIO":
            data = te.compute(
                [n_size, k_size, wmma_m, wmma_k],
                lambda n, k, nn, kk: pad[
                    n // (out_h * out_w),
                    (n % (out_h * out_w) // out_w) * stride_h
                    + (k // (kernel_w * in_c)) * dilation_h,
                    (n % out_w) * stride_w + (k // in_c % kernel_w) * dilation_w,
                    k % in_c,
                    nn,
                    kk,
                ],
                name="data",
            )
        C = te.compute(
            [n_size, out_c, wmma_m, wmma_n],
            lambda i, j, ii, jj: te.sum(data[i, k_axis, ii, wk_axis].astype(out_dtype) * kernel_decode[j, k_axis, jj, wk_axis].astype(out_dtype), axis=[k_axis, wk_axis]),
            "T_conv",
        )

    return [C]

def op_compute_fp(attrs, inputs, output_type):
    assert len(inputs[0].shape) == 6, "data arg number mismatch!"
    
    data, kernel = inputs[:2]
    out_dtype = output_type.dtype
    Scales = None
    Zeros = None
    if len(inputs) == 3:
        Scales = inputs[2]
    elif len(inputs) == 4:
        Scales = inputs[2]
        Zeros = inputs[3]
    
    bits = int(attrs['bits'])
    format = str(attrs['format'])
    assert format=="fp_e5m2", "Only support fp_e5m2 format currently"
    n_float_per_i8 = 8 // bits

    def _tir_u8_to_f8_to_float(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr):
        assert val.dtype == "int8"
        assert nbit == 8
        return tir.reinterpret('float16', tir.Cast('int16', val) << 8)
    
    def kernel_decode_func(n, k, nn, kk):
        w = _tir_u8_to_f8_to_float(bits, kernel[n, k, nn, kk // n_float_per_i8], kk % n_float_per_i8)
        return w

    kernel_decode = te.compute(
        [kernel.shape[0], kernel.shape[1], kernel.shape[2], kernel.shape[3] * n_float_per_i8],
        kernel_decode_func,
        name='B_decode'
    )
    # padding
    pad_top, pad_left, pad_bottom, pad_right = get_pad(attrs.padding)
    pad_shape = list(data.shape)
    if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
        pad_value = tir.const(0.0, data.dtype)
        if attrs.data_layout == "NHWC":
            pad_shape[1] += pad_top + pad_bottom
            pad_shape[2] += pad_left + pad_right
            pad = te.compute(
                pad_shape,
                lambda n, h, w, c, nn, cc: te.if_then_else(
                    tir.all(
                        h >= pad_bottom,
                        w >= pad_left,
                        h < pad_shape[1] - pad_top,
                        w < pad_shape[2] - pad_right,
                    ),
                    data[n, h - pad_bottom, w - pad_left, c, nn, cc],
                    pad_value,
                ),
                name="pad",
            )
        else:
            log.error("currently do not support data_layout != NHWC")
    else:
        pad = data

    # compute GEMM
    if attrs.data_layout == "NHWC":
        batch, in_h, in_w, in_c, wmma_m, wmma_k = data.shape
    else:
        log.error("currently do not support data_layout != NHWC")

    # for perfect quant, we use nt layout, the kernel is (oc, ic*kh*kw, wmma_n, wmma_k)
    out_c = kernel.shape[-4]
    wmma_n = kernel.shape[-2]
    kernel_h, kernel_w = attrs.kernel_size
    stride_h, stride_w = attrs.strides
    dilation_h, dilation_w = attrs.dilation
    k_size = kernel_h * kernel_w * in_c
    k_axis = te.reduce_axis((0, k_size), name="k")
    wk_axis = te.reduce_axis((0, wmma_k), name="wk")
    out_h = (
        in_h + pad_top + pad_bottom - 1 - (kernel_h - 1) * dilation_h
    ) // stride_h + 1
    out_w = (
        in_w + pad_left + pad_right - 1 - (kernel_w - 1) * dilation_w
    ) // stride_w + 1
    n_size = out_h * out_w * batch

    if attrs.data_layout == "NHWC":
        if attrs.kernel_layout == "HWIO":
            data = te.compute(
                [n_size, k_size, wmma_m, wmma_k],
                lambda n, k, nn, kk: pad[
                    n // (out_h * out_w),
                    (n % (out_h * out_w) // out_w) * stride_h
                    + (k // (kernel_w * in_c)) * dilation_h,
                    (n % out_w) * stride_w + (k // in_c % kernel_w) * dilation_w,
                    k % in_c,
                    nn,
                    kk,
                ],
                name="data",
            )
        C = te.compute(
            [n_size, out_c, wmma_m, wmma_n],
            lambda i, j, ii, jj: te.sum(data[i, k_axis, ii, wk_axis].astype(out_dtype) * kernel_decode[j, k_axis, jj, wk_axis].astype(out_dtype), axis=[k_axis, wk_axis]),
            "T_conv",
        )

    return [C]


def op_compute_int4b(attrs, inputs, output_type):
    assert len(inputs[0].shape) == 6, "data arg number mismatch!"
    
    data, kernel = inputs[:2]
    out_dtype = output_type.dtype

    bits = int(attrs['bits'])
    format = str(attrs['format'])
    assert format=="int4b", "Only support int4b format currently"
    assert bits==4 or bits==1, "Only support 4 or 1 bits currently"
    n_float_per_i8 = 8 // bits
    def _tir_u8_to_2u4_u8(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
        assert val.dtype == "int8"
        mask = tir.const((1 << nbit) - 1, "int8")
        var_0 = (val >> pos.astype("int8")) & mask
        var_1 = (val >> (pos + nbit).astype("int8")) & mask
        return var_0 + (var_1 << 4)
    
    def kernel_decode_func(n, k, nn, kk):
        w = _tir_u8_to_2u4_u8(bits, kernel[n, k, nn, kk // n_float_per_i8], (kk * 2) % n_float_per_i8, data.dtype)
        return w
    if bits < 4:   
        kernel_decode = te.compute(
            [kernel.shape[0], kernel.shape[1], kernel.shape[2], kernel.shape[3] * n_float_per_i8 // 2],
            kernel_decode_func,
            name='B_decode'
        )
    else:
        kernel_decode = kernel

    # padding
    pad_top, pad_left, pad_bottom, pad_right = get_pad(attrs.padding)
    pad_shape = list(data.shape)
    if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
        pad_value = tir.const(0.0, data.dtype)
        if attrs.data_layout == "NHWC":
            pad_shape[1] += pad_top + pad_bottom
            pad_shape[2] += pad_left + pad_right
            pad = te.compute(
                pad_shape,
                lambda n, h, w, c, nn, cc: te.if_then_else(
                    tir.all(
                        h >= pad_bottom,
                        w >= pad_left,
                        h < pad_shape[1] - pad_top,
                        w < pad_shape[2] - pad_right,
                    ),
                    data[n, h - pad_bottom, w - pad_left, c, nn, cc],
                    pad_value,
                ),
                name="pad",
            )
        else:
            log.error("currently do not support data_layout != NHWC")
    else:
        pad = data

    # compute GEMM
    if attrs.data_layout == "NHWC":
        batch, in_h, in_w, in_c, wmma_m, wmma_k = data.shape
    else:
        log.error("currently do not support data_layout != NHWC")

    # for perfect quant, we use nt layout, the kernel is (oc, ic*kh*kw, wmma_n, wmma_k)
    out_c = kernel.shape[-4]
    wmma_n = kernel.shape[-2]
    kernel_h, kernel_w = attrs.kernel_size
    stride_h, stride_w = attrs.strides
    dilation_h, dilation_w = attrs.dilation
    k_size = kernel_h * kernel_w * in_c
    k_axis = te.reduce_axis((0, k_size), name="k")
    wk_axis = te.reduce_axis((0, wmma_k), name="wk")
    out_h = (
        in_h + pad_top + pad_bottom - 1 - (kernel_h - 1) * dilation_h
    ) // stride_h + 1
    out_w = (
        in_w + pad_left + pad_right - 1 - (kernel_w - 1) * dilation_w
    ) // stride_w + 1
    n_size = out_h * out_w * batch

    if attrs.data_layout == "NHWC":
        if attrs.kernel_layout == "HWIO":
            data = te.compute(
                [n_size, k_size, wmma_m, wmma_k],
                lambda n, k, nn, kk: pad[
                    n // (out_h * out_w),
                    (n % (out_h * out_w) // out_w) * stride_h
                    + (k // (kernel_w * in_c)) * dilation_h,
                    (n % out_w) * stride_w + (k // in_c % kernel_w) * dilation_w,
                    k % in_c,
                    nn,
                    kk,
                ],
                name="data",
            )
        C = te.compute(
            [n_size, out_c, wmma_m, wmma_n],
            lambda i, j, ii, jj: te.sum(data[i, k_axis, ii, wk_axis].astype(out_dtype) * kernel_decode[j, k_axis, jj, wk_axis].astype(out_dtype), axis=[k_axis, wk_axis]),
            "T_conv",
        )

    return [C]


@target.override_native_generic_func("strategy_ladder_perfect_im2col_quant_conv")
def op_strategy(attrs, inputs, out_type, target):
    strategy = relay.op.OpStrategy()
    if attrs["format"] == "mxfp":
        strategy.add_implementation(
            op_compute_mxfp,
            wrap_topi_schedule(topi.generic.schedule_extern),
            name="ladder.quant_linear.generic",
        )
    elif attrs["format"] == "int4b":
        strategy.add_implementation(
            op_compute_int4b,
            wrap_topi_schedule(topi.generic.schedule_extern),
            name="ladder.quant_linear.generic",
        )
    elif "fp_" in attrs["format"]:
        assert attrs["format"] == "fp_e5m2", "Only support fp_e5m2 currently"
        strategy.add_implementation(
            op_compute_fp,
            wrap_topi_schedule(topi.generic.schedule_extern),
            name="ladder.quant_linear.generic",
        )
    else:
        strategy.add_implementation(
            op_compute_int,
            wrap_topi_schedule(topi.generic.schedule_extern),
            name="ladder.perfect_im2col_quant_conv.generic",
        )
    return strategy


def op_register():
    op_name = "ladder.perfect_im2col_quant_conv"
    reg.register(op_name)
    op = reg.get(op_name)
    op.set_support_level(10)
    op.add_type_rel(op_name + "_rel", op_relation)
    op.add_argument("lhs", "Tensor", "The left hand side tensor.")
    op.add_argument("rhs", "Tensor", "The right hand side tensor.")
    op.set_attrs_type_key("relay.attrs.Conv2DAttrs")
    reg.register_pattern(op_name, relay.op.OpPattern.OUT_ELEMWISE_FUSABLE)
    reg.register_strategy(op_name, op_strategy)


op_register()

__all__ = []
