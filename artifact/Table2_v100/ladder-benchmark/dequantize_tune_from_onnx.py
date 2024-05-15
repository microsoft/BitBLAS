# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import argparse
import os.path as osp
import numpy as np
import onnx
import ladder
import tvm
from tvm import relay
from tvm.contrib.debugger import debug_executor
from tvm.contrib import graph_executor
from ladder.utils import write_mod
from configs import models
import os
import torch
import time

async_propagation = True

# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "logs/e2e/" + fname
if async_propagation:
    log_path += "_async"


def run(prefix, arch, quant_type, quant_config, convert_int=False):
    if "model.onnx" not in prefix:
        onnx_model = onnx.load(osp.join(prefix, "model.onnx"))
    else:
        onnx_model = onnx.load(prefix)
    mod, params = relay.frontend.from_onnx(
        onnx_model, convert_config={"use_welder_matmul": True}
    )
    write_mod(mod, log_path, "load_from_onnx")

    if args.cublas:
        from tvm.relay.op.contrib.cublas import pattern_table

        seq = tvm.transform.Sequential(
            [
                relay.transform.InferType(),
                relay.transform.MergeComposite(pattern_table()),
                relay.transform.AnnotateTarget("cublas"),
                relay.transform.PartitionGraph(bind_constants=False),
                relay.transform.InferType(),
            ]
        )
        mod = seq(mod)

    if args.cudnn:
        from tvm.relay.op.contrib.cudnn import pattern_table

        seq = tvm.transform.Sequential(
            [
                relay.transform.InferType(),
                relay.transform.MergeComposite(pattern_table()),
                relay.transform.AnnotateTarget("cudnn"),
                relay.transform.PartitionGraph(bind_constants=False),
                relay.transform.InferType(),
            ]
        )
        mod = seq(mod)

    if args.nhwc:
        # must convert bias_add -> broadcast_add to propogate the layout
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.CanonicalizeOps()(mod)
        write_mod(mod, log_path, "CanonicalizeOps")
        mod = relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]})(mod)
        write_mod(mod, log_path, "ConvertLayout")
    mod = relay.transform.SimplifyInference()(mod)
    write_mod(mod, log_path, "SimplifyInference")
    mod = relay.transform.FoldConstant()(mod)
    write_mod(mod, log_path, "FoldConstant")
    mod = ladder.relay.transform.WelderExprRewrite()(mod)
    write_mod(mod, log_path, "expr_rewrite")
    mod = ladder.relay.transform.LadderConvImplicitGemm(
        use_async_propagation=async_propagation
    )(mod)
    write_mod(mod, log_path, "LadderConvImplicitGemm")
    mod = ladder.relay.transform.WelderConvImplicitGemm()(mod)
    write_mod(mod, log_path, "WelderConvImplicitGemm")
    mod = ladder.relay.transform.LadderFakeQuantConv(
        quant_config=quant_config, quant_type=quant_type, convert_int=convert_int
    )(mod)
    write_mod(mod, log_path, "LadderFakeQuantConv")
    mod = relay.transform.FoldConstant()(mod)
    write_mod(mod, log_path, "FoldConstant")
    mod = relay.transform.EliminateCommonSubexpr()(mod)
    write_mod(mod, log_path, "EliminateCommonSubexpr")
    mod = ladder.relay.transform.LadderRewriteInceptionLayout()(mod)
    write_mod(mod, log_path, "LadderRewriteInceptionLayout")
    mod = relay.transform.DeadCodeElimination()(mod)
    write_mod(mod, log_path, "DeadCodeElimination")
    mod = relay.transform.FoldConstant()(mod)
    write_mod(mod, log_path, "FoldConstant")
    mod = relay.transform.EliminateCommonSubexpr()(mod)
    write_mod(mod, log_path, "EliminateCommonSubexpr")
    mod = ladder.relay.transform.WelderFuseOps()(mod)
    write_mod(mod, log_path, "WelderFuseOps")
    mod = ladder.relay.transform.AnnotateLadderTensorCore(arch)(mod)
    write_mod(mod, log_path, "AnnotateLadderTensorCore")
    mod = ladder.relay.transform.AnnotateTensorCore()(mod)
    write_mod(mod, log_path, "AnnotateWelderTensorCore")
    start_time = time.time()
    mod = ladder.relay.transform.WelderTunePass(arch, topk=40)(mod)
    write_mod(mod, log_path, "WelderTunePass")
    end_time = time.time() - start_time
    factory = relay.build(mod, arch.target, params=params)
    lib = ladder.relay.update_lib(
        factory.get_lib(), arch, osp.join(log_path, "model.so")
    )
    with open(osp.join(log_path, "graph.json"), "w") as f:
        f.write(factory.get_graph_json())
    with open(osp.join(log_path, "graph.params"), "wb") as f_params:
        f_params.write(tvm.runtime.save_param_dict(factory.get_params()))

    rt_mod = graph_executor.create(factory.get_graph_json(), lib, tvm.cuda(0))
    rt_mod.set_input(**factory.get_params())
    print(rt_mod.benchmark(tvm.cuda(0), min_repeat_ms=500, end_to_end=False))
    print("Tune time: {}".format(end_time))


def run_from_prebuilt(prefix, arch):
    path_lib = osp.join(prefix, "model.so")
    graph_json_path = osp.join(prefix, "graph.json")
    
    loaded_lib = tvm.runtime.load_module(path_lib)
    graph_json = open(graph_json_path).read()
    
    module = debug_executor.create(graph_json, loaded_lib, tvm.cuda(0))

    torch.cuda.cudart().cudaProfilerStart()
    print(module.benchmark(tvm.cuda(0), min_repeat_ms=500, end_to_end=False))
    torch.cuda.cudart().cudaProfilerStop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        type=str,
        default="resnet-50-b1",
    )
    parser.add_argument("--arch", type=str, default="cuda")
    parser.add_argument("--cublas", action="store_false")
    parser.add_argument("--cudnn", action="store_true")
    parser.add_argument("--nhwc", action="store_false")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--convert_int", action="store_true")
    parser.add_argument("--format", type=str, default="int")
    parser.add_argument("--fake_quant", type=int, default=-1)
    parser.add_argument("--from_prebuilt", action="store_true")
    args = parser.parse_args()
    from_prebuilt = args.from_prebuilt
    arch = ladder.arch.__getattribute__(args.arch)()
    model = models[args.prefix]
    log_path = osp.join(log_path, args.prefix)
    quant_config = {
        "format": args.format,
        "bits": args.bits,
        "group_size": -1,
    }

    if args.fake_quant > -1 or args.convert_int:
        log_path += f'_fq_{args.fake_quant}_{quant_config["format"]}_{quant_config["bits"]}_ci_{args.convert_int}'

    if from_prebuilt:
        run_from_prebuilt(model, arch)
    else:
        if args.prefix in ["resnet-50-b1"]:
            # dummy simulate case for holistic schedule
            run(model, arch, args.fake_quant, quant_config, args.convert_int)
        run(model, arch, args.fake_quant, quant_config, args.convert_int)