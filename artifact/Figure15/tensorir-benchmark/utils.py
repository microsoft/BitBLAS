from tvm import relay
from tvm.relay import op
from tvm.relay.dataflow_pattern import *
import argparse
from functools import partial
import logging
import os
import json

from tvm import meta_schedule as ms
from tvm.target import Target

from tvm.meta_schedule import postproc
from tvm.meta_schedule import schedule_rule as M
from tvm.meta_schedule.tune import TuneConfig


logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("tvm.meta_schedule").setLevel(logging.INFO)


def parse_args(workload_candidates, default_trials=20000):
    args = argparse.ArgumentParser()
    args.add_argument(
        "-w",
        "--workload",
        nargs="+",
        type=str,
        choices=workload_candidates,
        required=True,
    )
    args.add_argument("-bs", "--batch-size", nargs="+", type=int, default=[1])
    args.add_argument("-t", "--target", type=str)
    args.add_argument("-n", "--num-trials", type=int, default=default_trials)
    args.add_argument("--work-dir", type=str)
    use_rpc = args.add_mutually_exclusive_group()
    use_rpc.add_argument("--local", action="store_false",
                         dest="use_rpc", default=False)
    use_rpc.add_argument("--rpc", action="store_true", dest="use_rpc")
    args.add_argument("--rpc-host", type=str)
    args.add_argument("--rpc-port", type=int)
    args.add_argument("--rpc-key", type=str)
    args.add_argument("--workers", type=int)
    args.add_argument("--alloc-repeat", type=int, default=1)
    args.add_argument("--out-dtype", type=str, default="float16")

    parsed = args.parse_args()
    parsed.target = parsed.target or os.environ.get("TVM_TARGET")
    parsed.target = Target(parsed.target)
    parsed.work_dir = parsed.work_dir or f"logs/"
    if parsed.use_rpc:
        rpc_host = parsed.rpc_host or os.environ.get("TVM_RPC_HOST")
        rpc_port = parsed.rpc_port or int(os.environ.get("TVM_RPC_PORT"))
        rpc_key = parsed.rpc_key or os.environ.get("TVM_RPC_KEY")
        rpc_config = ms.runner.RPCConfig(
            tracker_host=rpc_host,
            tracker_port=rpc_port,
            tracker_key=rpc_key,
            session_timeout_sec=60,
        )
        workers = parsed.workers or rpc_config.count_num_servers(
            allow_missing=False)
        parsed.runner = partial(
            ms.runner.RPCRunner, rpc_config=rpc_config, max_workers=workers
        )
    else:
        parsed.runner = ms.runner.LocalRunner
    parsed.runner = parsed.runner(
        evaluator_config=ms.runner.EvaluatorConfig(
            number=3,
            repeat=1,
            min_repeat_ms=100,
            enable_cpu_cache_flush=False,
        )
    )
    return parsed


def load_config():
    cur_path = os.path.dirname(__file__)
    config_path = os.path.join(cur_path, "configs")
    with open(config_path) as f:
        return json.load(f)


def sch_rules_tensor_core():
    return [
        M.MultiLevelTiling(
            structure="SSSRRSRS",
            tile_binds=["blockIdx.y", "blockIdx.x", "threadIdx.y"],
            use_tensor_core=True,
            max_innermost_factor=4,
            vector_load_lens=[1, 2, 4, 8],
            reuse_read=M.ReuseType(
                req="must",
                levels=[4],
                scope="shared.dyn",
            ),
            reuse_write=M.ReuseType(
                req="no",
                levels=[3],
                scope="shared.dyn",
            ),
        ),
        M.AutoInline(
            into_producer=True,
            into_consumer=True,
            inline_const_tensor=True,
            disallow_if_then_else=False,
            require_injective=False,
            require_ordered=False,
            disallow_op=None,
        ),
        M.AutoInline(
            into_producer=True,
            into_consumer=True,
            inline_const_tensor=True,
            disallow_if_then_else=False,
            require_injective=False,
            require_ordered=False,
            disallow_op=None,
        ),
        M.CrossThreadReduction(
            thread_extents=[4, 8, 16, 32, 64, 128, 256, 512]),
        M.ParallelizeVectorizeUnroll(
            max_jobs_per_core=-1,  # disable parallelize
            max_vectorize_extent=-1,  # disable vectorize
            unroll_max_steps=[0, 16, 64, 512, 1024],
            unroll_explicit=True,
        ),
    ]


def postprocs_tensor_core():
    return [
        postproc.RewriteCooperativeFetch(),
        postproc.RewriteUnboundBlock(),
        postproc.RewriteParallelVectorizeUnroll(),
        postproc.RewriteReductionBlock(),
        postproc.RewriteTensorCore(),
        postproc.VerifyGPUCode(),
    ]


def get_search_config(n_trails, trails_per_task=2000):
    return TuneConfig(
        num_trials_per_iter=64,
        max_trials_per_task=trails_per_task,
        max_trials_global=n_trails,
        search_strategy_config={
            "population_size": 2048,
            "init_measured_ratio": 0.2,
            "init_min_unmeasured": 50,
            "genetic_num_iters": 3,
            "genetic_mutate_prob": 0.85,
            "genetic_max_fail_count": 10,
            "eps_greedy": 0.05,
        },
    )


def reshape_gelu_pattern(inp, bias, inv_sqrt):
    reshape = is_op("reshape")(inp)
    add = is_op("add")(reshape, bias) | is_op("nn.bias_add")(reshape, bias)
    mul = is_op("multiply")(add, inv_sqrt)
    cast_fp32 = is_op("cast")(mul)
    erf = is_op("erf")(cast_fp32)
    mul = is_op("multiply")(erf, is_constant())
    add_cast_fp32 = is_op("cast")(add)
    mul_add_half = is_op("add")(is_constant(), mul)
    mul_fp32 = is_op("multiply")(add_cast_fp32, mul_add_half)
    reshape = is_op("reshape")(mul_fp32)
    return is_op("cast")(reshape)


def convert_reshape_gelu(inp, bias, inv_sqrt):
    bias_out = inp + bias
    mul = bias_out * inv_sqrt
    erf = op.cast(op.erf(op.cast(mul, "float32")), "float16")
    mul_half = erf * relay.const(0.5, dtype="float16")
    return (mul_half + relay.const(0.5, dtype="float16")) * bias_out


class ReshapeGeLURewrite(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.inp = wildcard()
        self.bias = wildcard()
        self.inv_sqrt = wildcard()
        self.pattern = reshape_gelu_pattern(self.inp, self.bias, self.inv_sqrt)

    def callback(self, pre, post, node_map):
        inp = node_map[self.inp][0]
        bias = node_map[self.bias][0]
        inv_sqrt = node_map[self.inv_sqrt][0]
        return convert_reshape_gelu(inp, bias, inv_sqrt)


def rewrite_reshape_gelu(mod):
    mod["main"] = rewrite(ReshapeGeLURewrite(), mod["main"])
    return mod


def convert_conv2d_layout(mod, desired_layouts):
    with tvm.transform.PassContext(opt_level=3):
        seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
        return seq(mod)


def check_params_tensorcore_compatible(prim_func):
    params = prim_func.params
    buffer_map = prim_func.buffer_map
    buffers = [buffer_map[param] for param in params[:2]]
    for buffer in buffers:
        if buffer.shape[-1] % 8 != 0 or buffer.shape[-2] % 8 != 0:
            return False
    return True


def check_params_conv2d_tensorcore_compatible(prim_func):
    params = prim_func.params
    buffer_map = prim_func.buffer_map
    buffers = [buffer_map[param] for param in params[:2]]
    X = buffers[0]
    Weight = buffers[1]
    N, H, W, C = X.shape
    O, S, C, K = Weight.shape
    return K >= 3
    if (N * H * W) % 16 == 0 and (C * K * S) % 16 == 0 and O % 16 == 0:
        return True
    return False


def should_use_memhammer(task):
    mod = task.dispatched[0]
    global_var = mod.get_global_vars()[0]
    task_name = global_var.name_hint
    if "dense" in task_name or "batch_matmul" in task_name:
        # return True
        prim_func = mod[global_var]
        return check_params_tensorcore_compatible(prim_func)
    elif "conv" in task_name:
        prim_func = mod[global_var]
        return check_params_conv2d_tensorcore_compatible(prim_func)
    else:
        return False
