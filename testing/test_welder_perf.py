import argparse
import os

import numpy as np
import tvm

def get_rand_tensor(shape, dtype):
    if dtype == "float16":
        tensor = np.random.normal(size=shape).astype(np.float16)
    elif dtype == "float32":
        tensor = np.random.normal(size=shape).astype(np.float32)
    else:
        tensor = np.ones(shape).astype(dtype)
    return tensor

def perf_welder(prefix):
    if args.debug:
        from tvm.contrib.debugger.debug_runtime import debug_executor as graph_executor
    else:
        from tvm.contrib import graph_executor
    lib_path = os.path.join(prefix, "model.so")
    with open(os.path.join(prefix, "graph.json")) as f:
        graph_json = f.read()
    with open(os.path.join(prefix, "graph.params"), "rb") as f_params:
        params = f_params.read()
    lib = tvm.runtime.load_module(lib_path)
    rt_mod = graph_executor.create(graph_json, lib, tvm.cuda(0))
    rt_mod.load_params(params)
    shape_dict, dtype_dict = rt_mod.get_input_info()
    for name in shape_dict:
        rt_mod.set_input(name, get_rand_tensor(shape_dict[name], dtype_dict[name]))
    rt_mod.run()
    print(rt_mod.benchmark(tvm.cuda(0), min_repeat_ms=500, end_to_end=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    perf_welder(args.prefix)

