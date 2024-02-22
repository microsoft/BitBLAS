import numpy as np
import tvm
from tvm.script import tir as T
from bitblas.base.roller.policy import TensorCorePolicy, DefaultPolicy
from bitblas.base.roller.arch import CUDA
from bitblas.gpu.matmul_analysis import get_tensorized_func_and_tags
from bitblas.gpu import Matmul
from bitblas.base.utils import apply_and_build
import time
from tvm import te, tir


def conv2d_nhwc_hwio(n, f, h, w, c, kh, kw, s, d, p, in_dtype="float16", out_dtype="float16"):
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
            pad[
                n,
                h * stride_h + kh * dilation_h,
                w * stride_w + kw * dilation_w,
                c,
            ]
            * B[kh, kw, c, f],
            axis=[kh, kw, c],
        ),
        name="C",
    )
    return tvm.ir.IRModule({"main": te.create_prim_func([A, B, C])})


benchmark_sets = [
    # (prim_func, input_args, default_dlight_schedule),
    # (conv2d_nhwc_hwio, (128, 64, 224, 224, 3, 7, 7, 2, 1, 3, "float16", "float16"), Matmul),
    # (conv2d_nhwc_hwio, (128, 64, 224, 224, 64, 1, 1, 2, 1, 3, "float16", "float16"), Matmul),
    # (conv2d_nhwc_hwio, (128, 64, 224, 224, 3, 7, 7, 2, 1, 3, "float32", "float32"), Matmul),
    (conv2d_nhwc_hwio, (128, 64, 224, 224, 3, 7, 7, 2, 1, 3, "float16", "float16"), Matmul),
]
benchmark_results = {}
for get_prim_func, input_args, d_schedule in benchmark_sets:
    ir_module = get_prim_func(*input_args)
    func = ir_module["main"]
    target = tvm.target.Target("nvidia/nvidia-a100")
    arch = CUDA(target)
    policy = DefaultPolicy(func=func, arch=arch)
    try:
        func, tags = get_tensorized_func_and_tags(func, arch.target)
    except:
        tags = None
    if tags:
        policy = TensorCorePolicy(func=func, arch=arch, tags=tags)

    configs = policy.emit_config(20)

    tune_start = time.time()
    cpresults, best = apply_and_build(func, configs, arch, parallel_build=True)
    fast_tune_time = time.time() - tune_start
    print("[BitBLAS] The best latency of top 1 is {:.3f} ms".format(cpresults[0].latency * 1e3))
    print("[BitBLAS] The best latency of top 20 is {:.3f} ms".format(best.latency * 1e3))

    # evaluate the performance of the default schedule

    rule = d_schedule()
    default_tune_start = time.time()
    sch_default = rule.apply(func, target, False)
    with tvm.transform.PassContext(config={"tir.use_async_copy": True}):
        mod_default = tvm.build(sch_default.mod["main"], target="cuda")
    default_tune_time = time.time() - default_tune_start

    args = func.buffer_map.values()

    profile_tensors = []
    for arg in args:
        profile_tensors.append(
            tvm.nd.array(
                np.random.uniform(0, 1, [int(i) for i in arg.shape]).astype(arg.dtype),
                device=arch.device,
            )
        )

    timer_cuda_mod = mod_default.time_evaluator(mod_default.entry_name, arch.device, number=5)
    t = timer_cuda_mod(*profile_tensors).mean

    print("Time cost of Dlight default schedule: {:.3f} ms".format(t * 1e3))

    profile_config = {
        f"{get_prim_func.__name__}-{'-'.join([str(i) for i in input_args])}": {
            "fast_dlight_top20_tune_time": fast_tune_time,
            "fast_dlight_top1_latency": cpresults[0].latency * 1e3,
            "fast_dlight_top20_latency": best.latency * 1e3,
            "default_dlight_tune_time": default_tune_time,
            "default_dlight_latency": t * 1e3,
        }
    }
    benchmark_results.update(profile_config)

headers = [
    "PrimFunc",
    "Input Arguments",
    "FastDLight Top20 Tune Time",
    "FastDLight Top1 Latency",
    "FastDLight Top20 Latency",
    "DefaultDLight Tune Time",
    "DefaultDLight Latency",
]

col_width = (
    max(len(word) for row in [headers] + list(profile_config.values()) for word in row) + 2
)  # padding

print("".join(word.ljust(col_width) for word in headers))

print("-" * col_width * len(headers))

for config, values in benchmark_results.items():
    args = config.split("-")
    func_name = args[0]
    input_args = "-".join(args[1:])
    row = [
        func_name,
        input_args,
        f" {str(values['fast_dlight_top20_tune_time'])} s",
        f"{values['fast_dlight_top1_latency']:.3f} ms",
        f"{values['fast_dlight_top20_latency']:.3f} ms",
        str(values["default_dlight_tune_time"]),
        f"{values['default_dlight_latency']:.3f} ms",
    ]
    print("".join(word.ljust(col_width) for word in row))
