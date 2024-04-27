import tvm
import os
from tvm import auto_tensorize as at
import argparse


def conv2d(N, C, H, W, K, R, S, stride, padding, dilation, layout, in_dtype, out_dtype):
    kH = (R - 1) * dilation + 1
    kW = (S - 1) * dilation + 1
    pH = H + 2 * padding
    pW = W + 2 * padding
    if layout == "nchw":
        A = tvm.te.placeholder([N, C, H, W], dtype=in_dtype, name="A")
        B = tvm.te.placeholder([K, C, R, S], dtype=in_dtype, name="B")

        Pad = tvm.te.compute(
            [N, C, pH, pW],
            lambda n, c, h, w: tvm.tir.if_then_else(
                tvm.tir.all(h >= padding, h - padding < H, w >= padding, w - padding < W),
                A[n, c, h - padding, w - padding],
                tvm.tir.const(0.0, A.dtype),
            ),
            name="Pad",
        )

        rc = tvm.te.reduce_axis([0, C], name="rc")
        rr = tvm.te.reduce_axis([0, kH], name="rr")
        rs = tvm.te.reduce_axis([0, kW], name="rs")

        P = (pH - kH) // stride + 1
        Q = (pW - kW) // stride + 1
        Conv = tvm.te.compute(
            [N, K, P, Q],
            lambda n, k, p, q: tvm.te.sum(
                (
                    Pad[n, rc, p * stride + rr * dilation, q * stride + rs * dilation]
                    * B[k, rc, rr, rs]
                ).astype(out_dtype),
                axis=[rc, rr, rs],
            ),
            name="Conv",
        )
    elif layout == "nhwc":
        A = tvm.te.placeholder([N, H, W, C], dtype=in_dtype, name="A")
        B = tvm.te.placeholder([R, S, C, K], dtype=in_dtype, name="B")

        Pad = tvm.te.compute(
            [N, pH, pW, C],
            lambda n, h, w, c: tvm.tir.if_then_else(
                tvm.tir.all(h >= padding, h - padding < H, w >= padding, w - padding < W),
                A[n, h - padding, w - padding, c],
                tvm.tir.const(0.0, A.dtype),
            ),
            name="Pad",
        )

        rc = tvm.te.reduce_axis([0, C], name="rc")
        rr = tvm.te.reduce_axis([0, kH], name="rr")
        rs = tvm.te.reduce_axis([0, kW], name="rs")

        P = (pH - kH) // stride + 1
        Q = (pW - kW) // stride + 1
        Conv = tvm.te.compute(
            [N, P, Q, K],
            lambda n, p, q, k: tvm.te.sum(
                (
                    Pad[n, p * stride + rr * dilation, q * stride + rs * dilation, rc]
                    * B[rr, rs, rc, k]
                ).astype(out_dtype),
                axis=[rr, rs, rc],
            ),
            name="Conv",
        )
    elif layout == "hwnc":
        A = tvm.te.placeholder([H, W, N, C], dtype=in_dtype, name="A")
        B = tvm.te.placeholder([R, S, C, K], dtype=in_dtype, name="B")

        Pad = tvm.te.compute(
            [pH, pW, N, C],
            lambda h, w, n, c: tvm.tir.if_then_else(
                tvm.tir.all(h >= padding, h - padding < H, w >= padding, w - padding < W),
                A[h - padding, w - padding, n, c],
                tvm.tir.const(0.0, A.dtype),
            ),
            name="Pad",
        )

        rc = tvm.te.reduce_axis([0, C], name="rc")
        rr = tvm.te.reduce_axis([0, kH], name="rr")
        rs = tvm.te.reduce_axis([0, kW], name="rs")

        P = (pH - kH) // stride + 1
        Q = (pW - kW) // stride + 1
        Conv = tvm.te.compute(
            [P, Q, N, K],
            lambda p, q, n, k: tvm.te.sum(
                (
                    Pad[p * stride + rr * dilation, q * stride + rs * dilation, n, rc]
                    * B[rr, rs, rc, k]
                ).astype(out_dtype),
                axis=[rr, rs, rc],
            ),
            name="Conv",
        )
    else:
        raise RuntimeError(f"Unkonwn layout for conv2d: {layout}")
    return [A, B, Conv]


def mapping_tensorcore(
    N,
    C,
    H,
    W,
    K,
    R,
    S,
    stride,
    padding,
    dilation,
    layout,
    layer,
    in_dtype,
    out_dtype,
    simple_mode=True,
    trials=-1,
    verbose=False,
    drop_output=False,
    use_perf_model=False,
    perf_model_ratio=0.6,
):
    A, B, Conv = conv2d(N, C, H, W, K, R, S, stride, padding, dilation, layout, in_dtype, out_dtype)
    target_dag = at.compute_dag_from_tensors([Conv])
    target = "cuda"

    log_dir = "conv2d-%s-%s-%s-layer-%s" % (layout, in_dtype, out_dtype, layer)
    log_file = "conv2d-%s-%s-%s-layer-%s.log" % (layout, in_dtype, out_dtype, layer)

    measure_opt = at.MeasureOptions(target=target, timeout=100, number=200, min_repeat_ms=500)

    if simple_mode:
        trials = 1000 if trials < 0 else trials
        result = at.auto_tensorize(
            target_dag,
            target,
            log_file,
            measure_opt,
            trials=trials,
            verbose=verbose,
            drop_output=drop_output,
        )
        if not result.defined():
            print("Can't do tensorize.")
            return
        schedule_gen = result.sch_gen
        schedule_app = result.sch_app

        # load from file
        schedule_gen.load_from_file(log_file, clear=True)
        entry = schedule_gen.get_best_entry()
        # we store 1/time_cost in file
        params, value = entry.record, 1 / entry.value
        print(value)
        print(params.to_json())
    else:
        trials = 4000 if trials < 0 else trials
        result = at.auto_tensorize_v4(
            target_dag,
            target,
            log_file,
            measure_opt,
            schedule_log_dir=log_dir,
            trials=trials,
            search_group_size=5,
            transform_dump=verbose,
            drop_output=drop_output,
            enable_perf_model=use_perf_model,
            perf_percentage=perf_model_ratio,
        )
        if not result.defined():
            print("Can't do tensorize.")
            return
        schedule_gen = result.sch_gen
        schedule_app = result.sch_app

        # we store 1/time_cost in file
        params, value = result.params, result.perf
        print(value)
        print(params.to_json())

    cost = at.evaluate_params(schedule_app, params, measure_opt, dump=verbose)
    print("Cost of %s is %f ms" % (log_dir, cost))
    return cost


shapes_b1 = [
    # (batch, C, H, W, K, _, R, S, _, stride, padding, dilation, groups)
    (128, 64, 56, 56, 64, -1, 3, 3, 1, 1, 1, 1, 1),
    (128, 64, 56, 56, 64, -1, 1, 1, 1, 1, 1, 1, 1),
    (128, 128, 28, 28, 128, -1, 3, 3, 1, 1, 1, 1, 1),
    (128, 512, 28, 28, 128, -1, 1, 1, 1, 1, 1, 1, 1),
    (1, 64, 56, 56, 64, -1, 3, 3, 1, 1, 1, 1, 1),
    (1, 64, 56, 56, 64, -1, 1, 1, 1, 1, 1, 1, 1),
    (1, 128, 28, 28, 128, -1, 3, 3, 1, 1, 1, 1, 1),
    (1, 512, 28, 28, 128, -1, 1, 1, 1, 1, 1, 1, 1),
]

supported_dtypes = set(
    [
        ("float16", "float16"),
        ("float16", "float32"),
        ("bfloat16", "float32"),
        ("float32", "float32"),
        ("float64", "float64"),
        ("int4", "int32"),
        ("int8", "int32"),
    ]
)

example_text = """
 example:
    python mapping_conv2d_tensorcore.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 20
    python mapping_conv2d_tensorcore.py --in_dtype float16 --out_dtype float32 --begin 0 --num 1 --trials 20
    python mapping_conv2d_tensorcore.py --in_dtype float32 --out_dtype float32 --begin 0 --num 1 --trials 20
    python mapping_conv2d_tensorcore.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 400 --simple_mode 0
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layout", type=str, choices=["nchw", "nhwc", "hwnc"], default="nchw")
    parser.add_argument(
        "--in_dtype",
        type=str,
        choices=["float16", "float32", "float64", "bfloat16", "int4", "int8"],
        default="float16",
    )
    parser.add_argument(
        "--out_dtype",
        type=str,
        choices=["float16", "float32", "float64", "int32"],
        default="float16",
    )
    parser.add_argument("--begin", type=int, choices=list(range(len(shapes_b1))), default=0)
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes_b1) + 1)), default=len(shapes_b1)
    )
    parser.add_argument("--simple_mode", type=int, default=1, choices=[0, 1])
    parser.add_argument("--trials", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--drop_output", action="store_true")
    parser.add_argument("--use_perf_model", action="store_true")
    parser.add_argument("--perf_model_ratio", type=float, default=0.6)

    args = parser.parse_args()
    assert 0 < args.perf_model_ratio <= 1.0
    if args.use_perf_model:
        assert args.simple_mode == 0, "Performance model is only supported without simple_mode"
    batches = [args.batch]
    beg = args.begin
    num = args.num
    print(args.simple_mode)
    assert (
        args.in_dtype,
        args.out_dtype,
    ) in supported_dtypes, (
        f"The desired dtype pair {(args.in_dtype, args.out_dtype)} is not supported by Tensor Core."
    )
    for batch in batches:
        costs = []
        for i, shape in enumerate(shapes_b1[beg : beg + num]):
            (N, C, H, W, K, _, R, S, _, stride, padding, dilation, _) = shape
            # N = batch
            print("\n\nProblem size:")
            print(N, C, H, W, K, R, S, stride, padding, dilation)
            layer_name = f"({N},{C},{H},{W},{K},{R},{S},{stride},{padding},{dilation})"
            try:
                cost = mapping_tensorcore(
                    N,
                    C,
                    H,
                    W,
                    K,
                    R,
                    S,
                    stride,
                    padding,
                    dilation,
                    args.layout,
                    layer_name,
                    args.in_dtype,
                    args.out_dtype,
                    simple_mode=args.simple_mode,
                    trials=args.trials,
                    verbose=args.verbose,
                    drop_output=args.drop_output,
                    use_perf_model=args.use_perf_model,
                    perf_model_ratio=args.perf_model_ratio,
                )
                costs.append(cost)
            except Exception as e:
                print("Fail to run\n", str(e))
                costs.append(float("inf"))
        print("\nBatch=", batch)
        for cost in costs:
            print(cost)
