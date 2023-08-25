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
    lib_path = os.path.join(prefix, "model.so")
    lib = tvm.runtime.load_module(lib_path)
    if args.debug:
        from tvm.contrib.debugger.debug_runtime import debug_executor
        rt_mod = debug_executor.GraphModuleDebug(lib["debug_create"]("default", tvm.cuda(0)), [tvm.cuda(0)], lib["get_graph_json"](), None)
    else:
        from tvm.contrib import graph_executor
        rt_mod = graph_executor.GraphModule(lib["default"](tvm.cuda(0)))
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

