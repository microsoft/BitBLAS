import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort
from tvm.contrib import graph_executor
from tvm.contrib.debugger.debug_runtime import debug_executor

def get_max_diff(tensor_list_a, tensor_list_b):
    assert len(tensor_list_a) > 0
    total_diff = [0]
    for a, b in zip(tensor_list_a, tensor_list_b):
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        if np.any(np.logical_and(np.isnan(a), np.logical_not(np.isnan(b)))):
            return 1e7
        assert a.shape == b.shape
        diff = np.abs(a-b)
        diff /= np.abs(b).clip(1) # handle large floating numbers
        diff = np.max(diff)
        total_diff.append(diff)
    total_diff = max(total_diff)
    return total_diff

def ref_output(onnx_model_path):
    np.random.seed(0)
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    ort_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    ort_inputs = {}
    inputs = []
    for value in ort_session.get_inputs():
        if value.type == 'tensor(int64)':
            tensor = np.ones(value.shape).astype(np.int64)
        elif value.type == 'tensor(float16)':
            tensor = np.random.normal(size=value.shape).astype(np.float16)
        elif value.type == 'tensor(float)':
            tensor = np.random.normal(size=value.shape).astype(np.float32)
        else:
            raise NotImplementedError(value.type)
        ort_inputs[value.name] = tensor
        inputs.append(tensor)
    outputs = ort_session.get_outputs()
    outputs_name = [item.name for item in outputs]
    outputs = ort_session.run(outputs_name, ort_inputs)
    return inputs, outputs

def get_welder_outs(prefix, inputs):
    import tvm
    lib_path = os.path.join(prefix, "model.so")
    with open(os.path.join(prefix, "graph.json")) as f:
        graph_json = f.read()
    with open(os.path.join(prefix, "graph.params"), "rb") as f_params:
        params = f_params.read()
    lib = tvm.runtime.load_module(lib_path)
    rt_mod = graph_executor.create(graph_json, lib, tvm.cuda(0))
    rt_mod.load_params(params)
    for i, tensor in enumerate(inputs):
        rt_mod.set_input(i, tensor)
    rt_mod.run()
    outputs = []
    for i in range(rt_mod.get_num_outputs()):
        out = rt_mod.get_output(i).asnumpy()
        outputs.append(out)
    print(rt_mod.benchmark(tvm.cuda(0), min_repeat_ms=500, end_to_end=False))
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    args = parser.parse_args()

    prefix = args.prefix
    inputs, outputs_ref = ref_output(os.path.join(prefix, "model.onnx"))
    outputs = get_welder_outs(prefix, inputs)

    max_diff = get_max_diff(outputs, outputs_ref)
    print(outputs)
    print(outputs_ref)
    print("Output diff : ", max_diff)

