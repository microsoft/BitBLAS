# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import numpy as np  # type: ignore

import tvm
from tvm import relay, runtime
from tvm import meta_schedule as ms
from tvm.contrib.graph_executor import GraphModule
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.meta_schedule import extract_task_from_relay
from tvm.meta_schedule.tune import tune_extracted_tasks
import os
# We use fp16-16-32 intrinsic for e2e workloads
from tvm.meta_schedule.testing import tir_tensor_intrin_fp16
from utils import *
import onnx
import time
import logging
logging.basicConfig(level=logging.INFO)
from configs import onnx_files, input_shape_dict


def f_measurement(rt_mod: runtime.Module, device: runtime.ndarray.Device, input_data):
    mod = GraphModule(rt_mod["default"](device))
    for input_name, input_value in input_data.items():
        mod.set_input(input_name, input_value)
    evaluator = mod.module.time_evaluator(
        "run",
        device,
        min_repeat_ms=500,
        repeat=3,
    )
    results = list(np.array(evaluator().results) * 1000.0)  # type: ignore
    print(results)
    t = np.mean(results)
    return t

def tune(workload, input_shape, num_trials=1000):
    if not os.path.exists("benchmark/caches/relay"):
        os.makedirs("benchmark/caches/relay")
    _workload = onnx_files[workload]
    # input_name = _workload['input_name']
    input_dtype = _workload['input_dtype']
    onnx_model = _workload['path']
    onnx_model = onnx.load(onnx_model)
    # from onnx_model get input_name
    input_name = onnx_model.graph.input[0].name
    relay_mod, params = relay.frontend.from_onnx(
        onnx_model)
    
    target = os.environ.get("TVM_TARGET")
    target = Target(target)
    
    # relay_mod = convert_conv2d_layout(
    #     relay_mod, {"nn.conv2d": ["NHWC", "default"]})
    relay_mod = relay.transform.ToMixedPrecision("float16")(relay_mod)
    relay_mod = rewrite_reshape_gelu(relay_mod)
    tasks = extract_task_from_relay(
        relay_mod, target=target, params=params)

    # run tuning tasks
    print("Tuning...")
    memhammer_tasks = []
    other_tasks = []
    for tsk in tasks:
        if should_use_memhammer(tsk):
            print(tsk.task_name, "memhammer")
            memhammer_tasks.append(tsk)
        else:
            print(tsk.task_name, "non-memhammer")
            other_tasks.append(tsk)

    search_config = get_search_config(num_trials, 1000)
    runner = ms.runner.LocalRunner(
        evaluator_config=ms.runner.EvaluatorConfig(
            number=3,
            repeat=1,
            min_repeat_ms=100,
            enable_cpu_cache_flush=False,
        )
    )
    workdir = f"./logs/{workload}-{input_shape}"
    start = time.time()
    # database = tune_extracted_tasks(
    #     memhammer_tasks,
    #     config=search_config,
    #     sch_rules=sch_rules_tensor_core,
    #     postprocs=postprocs_tensor_core,
    #     work_dir=workdir,
    #     runner=runner,
    # )

    database = tune_extracted_tasks(
        other_tasks,
        config=search_config,
        # use default CUDA rules
        work_dir=workdir,
        # database=database,
        runner=runner,
    )


    with ms.ApplyHistoryBest(database):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True,
                    "tir.predicate_opt": True},
        ):
            mod = tvm.relay.build(relay_mod, target=target, params=params)
    end = time.time()
    if input_dtype.startswith("float"):
        input_data = {
            input_name: np.random.uniform(size=input_shape).astype(input_dtype)
        }
    else:
        input_data = {
            input_name: np.random.randint(
                low=0, high=10000, size=input_shape, dtype=input_dtype
            )
        }

    dev = tvm.device(target.kind.name)
    input_data = {
        key: tvm.runtime.ndarray.array(value, dev)
        for key, value in input_data.items()
    }
    cost = end - start
    print("Cost time: ", cost)
    with open(f"{workdir}/cost_time.txt", "w") as f:
        f.write(str(cost))

    latency = f_measurement(mod, dev, input_data)
    print("Latency: ", latency)
    with open(f"{workdir}/latency.txt", "w") as f:
        f.write(str(latency))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32", "float64"],
        default="float16",
    )
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--workload", type=str, default="resnet")
    args = parser.parse_args()
    input_shape = input_shape_dict[args.workload]
    tune(args.workload, input_shape, args.trials)
