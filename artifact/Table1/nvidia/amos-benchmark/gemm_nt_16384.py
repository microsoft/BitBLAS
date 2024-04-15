from timeit import repeat
import tvm
from tvm import auto_tensorize as at
import argparse
import numpy as np
from tvm.auto_tensorize.auto_tensorize import get_schedule

import ctypes
_cudart = ctypes.CDLL('libcudart.so')


def profile_start():
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception("cudaProfilerStart() returned %d" % ret)


def profile_stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception("cudaProfilerStop() returned %d" % ret)

_gemm_sizes = [[16384, 16384, 16384]]

def gemm(M, N, K, in_dtype, out_dtype):
    A = tvm.te.placeholder([M, K], dtype=in_dtype, name="A")
    B = tvm.te.placeholder([M, K], dtype=in_dtype, name="B")

    rk = tvm.te.reduce_axis([0, K], name="k")
    C = tvm.te.compute(
        [M, N], lambda i, j: tvm.te.sum((A[i, rk] * B[j, rk]).astype(out_dtype), axis=rk), name="C"
    )
    return [A, B, C]

def get_np_arrays(tensors):
    ret = []
    for t in tensors:
        np_ary = np.random.uniform(-1, 1, [int(x)
                                           for x in t.shape]).astype(t.dtype)
        ret.append(np_ary)
    return ret


def get_tvm_arrays(tensors, ctx):
    ret = []
    for t in tensors:
        np_ary = np.random.uniform(-1, 1, [int(x)
                                           for x in t.shape]).astype(t.dtype)
        tvm_ary = tvm.nd.array(np_ary, ctx)
        ret.append(tvm_ary)
    return ret


def get_tvm_arrays_from_np_arrays(arys, ctx):
    ret = []
    for ary in arys:
        tvm_ary = tvm.nd.array(ary, ctx)
        ret.append(tvm_ary)
    return ret


def mapping_tensorcore(
    M,
    N,
    K,
    layer,
    in_dtype,
    out_dtype,
    simple_mode=True,
    trials=-1,
    verbose=False,
    use_perf_model=False,
    perf_model_ratio=0.6,
):
    A, B, Gemm = gemm(M, N, K, in_dtype, out_dtype)
    target_dag = at.compute_dag_from_tensors([Gemm])
    target = "cuda"

    log_dir = "gemm-%s-%s-layer-%s" % (in_dtype, out_dtype, layer)
    log_file = "gemm-%s-%s-layer-%s.log" % (in_dtype, out_dtype, layer)

    measure_opt = at.MeasureOptions(target=target, timeout=100, number=200, min_repeat_ms=500)

    if simple_mode:
        trials = 1000 if trials < 0 else trials
        result = at.auto_tensorize(
            target_dag, target, log_file, measure_opt, trials=trials, verbose=verbose
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

    cost = 0.0
    if trials == 0:
        profile_start()
        ctx = tvm.gpu()
        inputs_ref = target_dag.get_inputs()
        
        inputs_np_arrays = get_np_arrays(inputs_ref)
        inputs_arrays = get_tvm_arrays_from_np_arrays(inputs_np_arrays, ctx)
        outputs_arrays_ref = get_tvm_arrays(list(target_dag.tensors), ctx)
        inputs_arrays = get_tvm_arrays_from_np_arrays(
                    inputs_np_arrays, ctx)
        # outputs_arrays = get_tvm_arrays(list(new_target_dag.tensors), ctx)
        sch, args = get_schedule(schedule_app, params)
        func = tvm.build(sch, args, target)
        timer_1 = func.time_evaluator(func.entry_name, ctx, number=1, repeat=1, min_repeat_ms=10)
        cost = timer_1(*inputs_arrays, *outputs_arrays_ref).mean
        profile_stop()
    else:
        cost = at.evaluate_params(schedule_app, params, measure_opt, dump=verbose)
        print("Cost of %s is %f ms" % (log_dir, cost))
    return cost


shapes = _gemm_sizes

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
    python mapping_gemm_tensorcore.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 20
    python mapping_gemm_tensorcore.py --in_dtype float16 --out_dtype float32 --begin 0 --num 1 --trials 20
    python mapping_gemm_tensorcore.py --in_dtype float32 --out_dtype float32 --begin 0 --num 1 --trials 20
    python mapping_gemm_tensorcore.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 400 --simple_mode 0
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    parser.add_argument("--begin", type=int, choices=list(range(len(shapes))), default=0)
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes) + 1)), default=len(shapes)
    )
    parser.add_argument("--simple_mode", type=int, default=1, choices=[0, 1])
    parser.add_argument("--trials", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use_perf_model", action="store_true")
    parser.add_argument("--perf_model_ratio", type=float, default=0.6)

    args = parser.parse_args()
    assert 0 < args.perf_model_ratio <= 1.0
    if args.use_perf_model:
        assert args.simple_mode == 0, "Performance model is only supported without simple_mode"
    beg = args.begin
    num = args.num
    print(args.simple_mode)
    assert (
        args.in_dtype,
        args.out_dtype,
    ) in supported_dtypes, (
        f"The desired dtype pair {(args.in_dtype, args.out_dtype)} is not supported by Tensor Core."
    )
    costs = []
    for i, shape in enumerate(shapes[beg : beg + num]):
        (M, K, N) = shape
        print("\n\nProblem size:")
        print(M, K, N)
        layer_name = f"({M}, {K}, {N})"
        try:
            cost = mapping_tensorcore(
                M,
                N,
                K,
                layer_name,
                args.in_dtype,
                args.out_dtype,
                simple_mode=args.simple_mode,
                trials=args.trials,
                verbose=args.verbose,
                use_perf_model=args.use_perf_model,
                perf_model_ratio=args.perf_model_ratio,
            )
            costs.append(cost)
        except Exception as e:
            print("Fail to run\n", str(e))
            costs.append(float("inf"))
    for cost in costs:
        print(cost)
