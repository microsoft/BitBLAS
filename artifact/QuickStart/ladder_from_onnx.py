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
import os
import torch
import logging

ladder.set_log_level(logging.INFO)

# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/e2e/" + fname

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', type=str, default='llama2-70b')
parser.add_argument('--arch', type=str, default="cuda")
parser.add_argument('--cublas', action="store_true")
parser.add_argument('--cudnn', action="store_false")
parser.add_argument('--nhwc', action="store_false")
parser.add_argument('--async_propagation', action="store_true", help="Use async propagation and async instructions, which should be only enabled on data center GPUs with async copy instructions.", default=False)
parser.add_argument("--prebuilt_path", type=str, default=None, help="Path to the prebuilt model. If set, the script will run from the prebuilt model.")

args = parser.parse_args()

def run(prefix, arch, async_propagate):
    if ".onnx" in prefix:
        onnx_model = onnx.load(prefix)
    else:
        onnx_model = onnx.load(osp.join(prefix, "model.onnx"))
    mod, params = relay.frontend.from_onnx(
        onnx_model, convert_config={"use_welder_matmul": False})
    write_mod(mod, log_path, "load_from_onnx")

    if args.nhwc:
        # must convert bias_add -> broadcast_add to propogate the layout
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.CanonicalizeOps()(mod)
        write_mod(mod, log_path, "CanonicalizeOps")
        mod = relay.transform.ConvertLayout(
            {"nn.conv2d": ["NHWC", "default"]})(mod)
        write_mod(mod, log_path, "ConvertLayout")
    mod = relay.transform.FoldConstant()(mod)
    write_mod(mod, log_path, "FoldConstant")
    mod = ladder.relay.transform.WelderExprRewrite(enable_softmax=True)(mod)
    write_mod(mod, log_path, "expr_rewrite")
   
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

    mod = ladder.relay.transform.LadderConvImplicitGemm(
        use_async_propagation=async_propagate)(mod)
    write_mod(mod, log_path, "LadderConvImplicitGemm")
    mod = ladder.relay.transform.LadderPerfectGemmTransform(
        use_async_propagation=async_propagate)(mod)
    write_mod(mod, log_path, "LadderPerfectGemmTransform")
    mod = ladder.relay.transform.WelderConvImplicitGemm()(mod)
    write_mod(mod, log_path, "WelderConvImplicitGemm")
    mod = relay.transform.FoldConstant()(mod)
    write_mod(mod, log_path, "FoldConstant")
    mod = relay.transform.EliminateCommonSubexpr()(mod)
    write_mod(mod, log_path, "EliminateCommonSubexpr")
    mod = ladder.relay.transform.LadderRewriteInceptionLayout()(mod)
    write_mod(mod, log_path, "LadderRewriteInceptionLayout")
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
    write_mod(mod, log_path, "cublas_partition")
    mod = relay.transform.DeadCodeElimination()(mod)
    write_mod(mod, log_path, "DeadCodeElimination")
    mod = relay.transform.FoldConstant()(mod)
    write_mod(mod, log_path, "FoldConstant")
    mod = relay.transform.EliminateCommonSubexpr()(mod)
    write_mod(mod, log_path, "EliminateCommonSubexpr")
    mod = ladder.relay.transform.WelderFuseOps()(mod)
    write_mod(mod, log_path, "WelderFuseOps")
    mod = ladder.relay.transform.AnnotateLadderTensorCore(arch=arch)(mod)
    write_mod(mod, log_path, "AnnotateLadderTensorCore")
    mod = ladder.relay.transform.AnnotateTensorCore()(mod)
    write_mod(mod, log_path, "AnnotateWelderTensorCore")

    mod = ladder.relay.transform.WelderTunePass(arch, topk=40,save_perf_log="./debug_group_info")(mod)
    write_mod(mod, log_path, "WelderTunePass")

    factory = relay.build(mod, arch.target, params=params)
    lib = ladder.relay.update_lib(
        factory.get_lib(), arch, osp.join(log_path, "model.so"))
    with open(osp.join(log_path, "graph.json"), "w") as f:
        f.write(factory.get_graph_json())
    with open(osp.join(log_path, "graph.params"), "wb") as f_params:
        f_params.write(tvm.runtime.save_param_dict(factory.get_params()))

    rt_mod = graph_executor.create(factory.get_graph_json(), lib, tvm.cuda(0))
    rt_mod.set_input(**factory.get_params())
    print(rt_mod.benchmark(tvm.cuda(0), min_repeat_ms=500, end_to_end=False))


def run_from_prebuilt(prefix, arch):
    lib_path = os.path.join(prefix, "model.so")
    with open(os.path.join(prefix, "graph.json")) as f:
        graph_json = f.read()
    with open(os.path.join(prefix, "graph.params"), "rb") as f_params:
        params = f_params.read()
    loaded_lib = tvm.runtime.load_module(lib_path)
    module = debug_executor.create(graph_json, loaded_lib, tvm.cuda(0))
    module.load_params(params)
    print(module.benchmark(tvm.cuda(0), min_repeat_ms=500, end_to_end=False))
    module.run()
    # dummy input
    input_shape = (1, 1)
    dtype = 'int64'
    input_data = tvm.nd.array(np.ones(input_shape).astype(dtype))
    module.set_input("onnx::Reshape_0", input_data)
    module.run()
    outputs = []
    for i in range(module.get_num_outputs()):
        out = module.get_output(i).asnumpy()
        outputs.append(out)
    print(outputs)


if __name__ == "__main__":
    path = args.prefix
    arch = ladder.arch.__getattribute__(args.arch)()
    async_propagate = args.async_propagation
    if arch.compute_capability == "80":
        async_propagate = True
    # path = "/home/t-leiwang/ladder_workspace/Ladder/artifact/QuickStart/qmodels/opt-125m-4bit/qmodel_b1s1/qmodel_b1s1.onnx"
    prebuilt_path = args.prebuilt_path
    prebuilt_path = "/home/t-leiwang/ladder_workspace/Ladder/artifact/QuickStart/./progress/e2e/ladder_from_onnx"
    if prebuilt_path:
        run_from_prebuilt(prebuilt_path, arch)
    else:
        run(path, arch, async_propagate)
