# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from tvm import relay, tir, ir, target, te, topi
from tvm.relay.op.strategy import wrap_topi_schedule
from tvm.relay import reg
import logging 

logger = logging.getLogger(__name__)

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
    assert len(arg_types) == 2, "type relation arg number mismatch!"
    assert attrs.data_layout in ["NCHW", "NHWC"]
    a_shape = arg_types[0].shape
    if attrs.data_layout == "NCHW":
        batch, _, in_h, in_w = a_shape
    elif attrs.data_layout == "NHWC":
        batch, in_h, in_w, _ = a_shape
    pad_top, pad_left, pad_bottom, pad_right = get_pad(attrs.padding)
    dilation_h, dilation_w = attrs.dilation
    stride_h, stride_w = attrs.strides
    kernel_h, kernel_w = attrs.kernel_size
    out_c = attrs.channels
    out_h = (in_h + pad_top + pad_bottom - 1 - (kernel_h - 1) * dilation_h) // stride_h + 1
    out_w = (in_w + pad_left + pad_right - 1 - (kernel_w - 1) * dilation_w) // stride_w + 1
    out_dtype = attrs.out_dtype if attrs.out_dtype else arg_types[0].dtype
    if attrs.data_layout == "NCHW":
        out_shape = [out_c, batch * out_h * out_w]
    elif attrs.data_layout == "NHWC":
        out_shape = [batch * out_h * out_w, out_c]
    return relay.TensorType(out_shape, out_dtype)

def op_compute(attrs, inputs, output_type):
    assert len(inputs) == 2, "input arg number mismatch!"
    data, kernel = inputs
    # padding
    pad_top, pad_left, pad_bottom, pad_right = get_pad(attrs.padding)
    pad_shape = list(data.shape)
    if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
        pad_value = tir.const(0.0, data.dtype)
        if attrs.data_layout == "NCHW":
            pad_shape[2] += pad_top + pad_bottom
            pad_shape[3] += pad_left + pad_right
            pad = te.compute(pad_shape, lambda n, c, h, w: te.if_then_else(
                tir.all(h>=pad_bottom, w>=pad_left, h<pad_shape[2]-pad_top, w<pad_shape[3]-pad_right),
                data[n, c, h-pad_bottom, w-pad_left], pad_value
            ), name="pad")
        elif attrs.data_layout == "NHWC":
            pad_shape[1] += pad_top + pad_bottom
            pad_shape[2] += pad_left + pad_right
            pad = te.compute(pad_shape, lambda n, h, w, c: te.if_then_else(
                tir.all(h>=pad_bottom, w>=pad_left, h<pad_shape[1]-pad_top, w<pad_shape[2]-pad_right),
                data[n, h-pad_bottom, w-pad_left, c], pad_value
            ), name="pad")
    else:
        pad = data

    # compute GEMM
    if attrs.data_layout == "NCHW":
        batch, in_c, in_h, in_w = data.shape
    elif attrs.data_layout == "NHWC":
        batch, in_h, in_w, in_c = data.shape
    out_c = attrs.channels
    kernel_h, kernel_w = attrs.kernel_size
    stride_h, stride_w = attrs.strides
    dilation_h, dilation_w = attrs.dilation
    k_size = kernel_h * kernel_w * in_c
    k_axis = te.reduce_axis((0, k_size), name="k")
    out_h = (in_h + pad_top + pad_bottom - 1 - (kernel_h - 1) * dilation_h) // stride_h + 1
    out_w = (in_w + pad_left + pad_right - 1 - (kernel_w - 1) * dilation_w) // stride_w + 1
    n_size = out_h * out_w * batch
    if attrs.data_layout == "NCHW":
        data = te.compute([k_size, n_size], lambda k, n: pad[
                n//(out_h*out_w),
                k//(kernel_h*kernel_w),
                (n%(out_h*out_w)//out_w)*stride_h+(k%(kernel_h*kernel_w)//kernel_w)*dilation_h,
                (n%out_w)*stride_w+(k%kernel_w)*dilation_w],
            name="data")
        C = te.compute([out_c, n_size], lambda i, j: te.sum(kernel[i, k_axis] * data[k_axis, j], axis=k_axis), "T_conv")

    elif attrs.data_layout == "NHWC":
        data = te.compute([n_size, k_size], lambda n, k: pad[
                n//(out_h*out_w),
                (n%(out_h*out_w)//out_w)*stride_h+(k//(kernel_w*in_c))*dilation_h,
                (n%out_w)*stride_w+(k//in_c%kernel_w)*dilation_w,
                k%in_c],
            name="data")
        C = te.compute([n_size, out_c], lambda i, j: te.sum(data[i, k_axis] * kernel[k_axis, j], axis=k_axis), "T_conv")
    return [C]

@target.override_native_generic_func("strategy_ladder_C2DImplicitGemm")
def op_strategy(attrs, inputs, out_type, target):
    strategy = relay.op.OpStrategy()
    strategy.add_implementation(
        op_compute,
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="ladder.C2DImplicitGemm.generic",
    )
    return strategy

def op_register():
    op_name = "ladder.C2DImplicitGemm"
    reg.register(op_name)
    op = reg.get(op_name)
    op.set_num_inputs(2)
    op.set_support_level(10)
    op.add_type_rel(op_name + "_rel", op_relation)
    op.add_argument("lhs", "Tensor", "The left hand side tensor.")
    op.add_argument("rhs", "Tensor", "The right hand side tensor.")
    op.set_attrs_type_key("relay.attrs.Conv2DAttrs")
    reg.register_pattern(op_name, relay.op.OpPattern.OUT_ELEMWISE_FUSABLE)
    reg.register_strategy(op_name, op_strategy)

op_register()

__all__ = []
