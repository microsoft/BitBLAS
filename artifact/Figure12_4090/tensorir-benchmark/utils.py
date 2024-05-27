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
from tvm.tir.tensor_intrin.cuda import *


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
    use_rpc.add_argument("--local", action="store_false", dest="use_rpc", default=False)
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
        workers = parsed.workers or rpc_config.count_num_servers(allow_missing=False)
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
        M.CrossThreadReduction(thread_extents=[4, 8, 16, 32, 64, 128, 256, 512]),
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
