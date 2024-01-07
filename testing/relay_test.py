import argparse
import os.path as osp

import onnx
import ladder
import tvm
from tvm import relay
from tvm.contrib.debugger import debug_executor
from tvm.contrib import graph_executor

def run(prefix, arch):
    onnx_model = onnx.load(osp.join(prefix, "model.onnx"))
    mod, params = relay.frontend.from_onnx(onnx_model, convert_config={"use_welder_matmul": not args.cublas})
    mod = relay.transform.SimplifyInference()(mod) # remove BN, dropout ...

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
        mod = relay.transform.CanonicalizeOps()(mod)
        mod = relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]})(mod)
        mod = relay.transform.FoldConstant()(mod)

    mod = ladder.relay.transform.WelderExprRewrite()(mod)
    mod = ladder.relay.transform.WelderConvImplicitGemm()(mod)
    mod = ladder.relay.transform.WelderDotSplitK()(mod)
    mod = relay.transform.FoldConstant()(mod)
    mod = ladder.relay.transform.WelderFuseOps()(mod)
    mod = ladder.relay.transform.AnnotateTensorCore()(mod)
    mod = ladder.relay.transform.WelderTunePass(arch, osp.join(prefix, "welder_tuned.json"))(mod)

    factory = relay.build(mod, arch.target, params=params)
    lib = ladder.relay.update_lib(factory, arch, osp.join(prefix, "model.so"))
    rt_mod = graph_executor.GraphModule(lib["default"](tvm.cuda()))
    print(rt_mod.benchmark(tvm.cuda(0), min_repeat_ms=500, end_to_end=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--arch', type=str, default="cuda")
    parser.add_argument('--cublas', action="store_true")
    parser.add_argument('--cudnn', action="store_true")
    parser.add_argument('--nhwc', action="store_true")
    args = parser.parse_args()
    arch = ladder.arch.__getattribute__(args.arch)()
    run(args.prefix, arch)
