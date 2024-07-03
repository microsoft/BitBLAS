# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# pre-transformed tir expression of matmul
import tvm
from tvm import te, tir


def conv2d_nhwc_ohwi(
    n,
    f,
    h,
    w,
    c,
    kh,
    kw,
    s,
    d,
    p,
    in_dtype="float16",
    accum_dtype="float16",
    out_dtype="float16",
):

    A = te.placeholder((n, h, w, c), name="input", dtype=in_dtype)
    B = te.placeholder((f, kh, kw, c), name="weight", dtype=in_dtype)

    pad_shape = (n, h + 2 * p, w + 2 * p, c)
    pad_value = tir.const(0.0, A.dtype)
    pad = te.compute(
        pad_shape,
        lambda n, h, w, c: te.if_then_else(
            tir.all(
                h >= p,
                w >= p,
                h < pad_shape[1] - p,
                w < pad_shape[2] - p,
            ),
            A[n, h - p, w - p, c],
            pad_value,
        ),
        name="pad",
    )
    kernel_h, kernel_w = kh, kw
    stride_h, stride_w = s, s
    dilation_h, dilation_w = d, d
    out_h = (h + 2 * p - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    out_w = (w + 2 * p - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    out_shape = (n, out_h, out_w, f)
    kh = te.reduce_axis((0, kernel_h), name="kh")
    kw = te.reduce_axis((0, kernel_w), name="kw")
    c = te.reduce_axis((0, c), name="c")
    C = te.compute(
        out_shape,
        lambda n, h, w, f: te.sum(
            pad[n, h * stride_h + kh * tir.any(dilation_h), w * stride_w + kw * tir.any(dilation_w),
                c,].astype(accum_dtype) * B[f, kh - 1 - tir.any(dilation_h), kw - 1 - tir.any(
                    dilation_w), c].astype(accum_dtype),
            axis=[kh, kw, c],
        ),
        name="C",
    )
    args = [A, B]
    last_output = C
    if accum_dtype != out_dtype:
        D = te.compute(out_shape, lambda n, h, w, c: C[n, h, w, c].astype(out_dtype), name="D")
        last_output = D
    args.append(last_output)
    func = te.create_prim_func(args)

    return tvm.IRModule.from_expr(func)


def conv2d_nhwc_hwio(
    n,
    f,
    h,
    w,
    c,
    kh,
    kw,
    s,
    d,
    p,
    in_dtype="float16",
    accum_dtype="float16",
    out_dtype="float16",
):

    A = te.placeholder((n, h, w, c), name="input", dtype=in_dtype)
    B = te.placeholder((kh, kw, c, f), name="weight", dtype=in_dtype)

    pad_shape = (n, h + 2 * p, w + 2 * p, c)
    pad_value = tir.const(0.0, A.dtype)
    pad = te.compute(
        pad_shape,
        lambda n, h, w, c: te.if_then_else(
            tir.all(
                h >= p,
                w >= p,
                h < pad_shape[1] - p,
                w < pad_shape[2] - p,
            ),
            A[n, h - p, w - p, c],
            pad_value,
        ),
        name="pad",
    )
    kernel_h, kernel_w = kh, kw
    stride_h, stride_w = s, s
    dilation_h, dilation_w = d, d
    out_h = (h + 2 * p - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    out_w = (w + 2 * p - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    out_shape = (n, out_h, out_w, f)
    kh = te.reduce_axis((0, kernel_h), name="kh")
    kw = te.reduce_axis((0, kernel_w), name="kw")
    c = te.reduce_axis((0, c), name="c")
    C = te.compute(
        out_shape,
        lambda n, h, w, f: te.sum(
            pad[n, h * stride_h + kh * tir.any(dilation_h), w * stride_w + kw * tir.any(dilation_w),
                c,].astype(accum_dtype) * B[kh - 1 - tir.any(dilation_h), kw - 1 - tir.any(
                    dilation_w), c, f].astype(accum_dtype),
            axis=[kh, kw, c],
        ),
        name="C",
    )
    args = [A, B]
    last_output = C
    if accum_dtype != out_dtype:
        D = te.compute(out_shape, lambda n, h, w, c: C[n, h, w, c].astype(out_dtype), name="D")
        last_output = D
    args.append(last_output)
    func = te.create_prim_func(args)

    return tvm.IRModule.from_expr(func)


def select_implementation(
    n,
    f,
    h,
    w,
    c,
    kh,
    kw,
    s,
    d,
    p,
    in_dtype="float16",
    accum_dtype="float16",
    out_dtype="float16",
    input_layout="nhwc",
    weight_layout="ohwi",
):
    assert input_layout in ["nhwc", "nchw"]
    if input_layout == "nhwc" and weight_layout == "ohwi":
        return conv2d_nhwc_ohwi(
            n,
            f,
            h,
            w,
            c,
            kh,
            kw,
            s,
            d,
            p,
            in_dtype,
            accum_dtype,
            out_dtype,
        )
    elif input_layout == "nhwc" and weight_layout == "hwio":
        return conv2d_nhwc_hwio(
            n,
            f,
            h,
            w,
            c,
            kh,
            kw,
            s,
            d,
            p,
            in_dtype,
            accum_dtype,
            out_dtype,
        )
    else:
        raise ValueError("Unsupported input_layout: {} and weight_layout: {}".format(
            input_layout, weight_layout))
