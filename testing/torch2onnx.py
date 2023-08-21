import argparse
import os
import os.path as osp
import time

import numpy as np
import torch
from model.pytorch import *

def torch2onnx(prefix, model, inputs):
    os.makedirs(args.prefix, exist_ok=True)
    outputs = model(*inputs)
    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs, )
    input_names = ["input"+str(i) for i in range(len(inputs))]
    output_names = ["output"+str(i) for i in range(len(outputs))]
    torch.onnx.export(
        model, inputs,
        osp.join(prefix, "model.onnx"),
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=False,
        opset_version=11)
    feed_dict = dict(zip(input_names, [tensor.cpu() for tensor in inputs]))
    np.savez(osp.join(prefix, "inputs.npz"), **feed_dict)

def run_torch(model, inputs):
    model.eval()
    def get_runtime():
        tic = time.time()
        _ = model(*inputs)
        torch.cuda.synchronize()
        return (time.time() - tic) * 1000
    with torch.no_grad():
        print("Warming up ...")
        st = time.time()
        while time.time() - st < 1.0:
            get_runtime() # warmup
        times = [get_runtime() for i in range(100)]
        print(f"avg: {np.mean(times)} ms")
        print(f"min: {np.min(times)} ms")
        print(f"max: {np.max(times)} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--prefix", type=str, default="temp")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--run_torch", action="store_true", default=False)
    args = parser.parse_args()
    assert (args.model in globals()), "Model {} not found.".format(args.model)

    torch.random.manual_seed(0)
    model, inputs = globals()[args.model](args.bs)
    model = model.cuda()
    inputs = tuple([tensor.cuda() for tensor in inputs])

    if args.fp16:
        model = model.half()
        inputs = tuple([x.half() if torch.is_floating_point(x) else x for x in inputs])
    if args.run_torch:
        run_torch(model, inputs)
    else:
        torch2onnx(args.prefix, model, inputs)
