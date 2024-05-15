import argparse
import os.path as osp
import numpy as np
import onnx
import ladder
import tvm
from tvm import relay
from tvm.contrib.debugger import debug_executor
from tvm.contrib import graph_executor
from ladder_utils.logger import write_mod, write_code, write_sch
import os
import logging
ladder.set_log_level(logging.DEBUG)

import time
# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/e2e/" + fname

def run(prefix, arch, async_propagate):
    global log_path
    if async_propagate:
        log_path += "_async"
    if ".onnx" in prefix:
        onnx_model = onnx.load(prefix)
    else:
        onnx_model = onnx.load(osp.join(prefix, "model.onnx"))
    mod, params = relay.frontend.from_onnx(onnx_model, convert_config={"use_ladder_matmul": True})
    write_mod(mod, log_path, "load_from_onnx")


    if args.nhwc:
        # must convert bias_add -> broadcast_add to propogate the layout
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.CanonicalizeOps()(mod)
        write_mod(mod, log_path, "CanonicalizeOps")
        mod = relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]})(mod)
        write_mod(mod, log_path, "ConvertLayout")
    mod = relay.transform.SimplifyInference()(mod)
    write_mod(mod, log_path, "SimplifyInference")
    mod = relay.transform.SimplifyExpr()(mod)
    write_mod(mod, log_path, "SimplifyExpr")
    mod = relay.transform.FoldConstant()(mod)
    write_mod(mod, log_path, "FoldConstant")
    mod = ladder.relay.transform.ladderExprRewrite(enable_softmax=False)(mod)
    mod = relay.transform.InferType()(mod)
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

    mod = ladder.relay.transform.LadderConvImplicitGemm(use_async_propagation=async_propagate)(mod)
    write_mod(mod, log_path, "LadderConvImplicitGemm")
    mod = ladder.relay.transform.LadderPerfectGemmTransform(
        use_async_propagation=async_propagate)(mod)
    write_mod(mod, log_path, "LadderPerfectGemmTransform")
    mod = ladder.relay.transform.ladderConvImplicitGemm()(mod)
    write_mod(mod, log_path, "ladderConvImplicitGemm")
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
    mod = ladder.relay.transform.ladderFuseOps()(mod)
    write_mod(mod, log_path, "ladderFuseOps")
    mod = ladder.relay.transform.AnnotateLadderTensorCore(arch)(mod)
    write_mod(mod, log_path, "AnnotateLadderTensorCore")
    mod = ladder.relay.transform.AnnotateTensorCore()(mod)
    write_mod(mod, log_path, "AnnotateladderTensorCore")

    start_time = time.time()
    mod = ladder.relay.transform.ladderTunePass(arch, topk=40)(mod)
    write_mod(mod, log_path, "ladderTunePass")
    end_time = time.time() - start_time
    
    factory = relay.build(mod, arch.target, params=params)
    lib = ladder.relay.update_lib(factory.get_lib(), arch, osp.join(log_path, "model.so"))
    with open(osp.join(log_path, "graph.json"), "w") as f:
        f.write(factory.get_graph_json())
    with open(osp.join(log_path, "graph.params"), "wb") as f_params:
        f_params.write(tvm.runtime.save_param_dict(factory.get_params()))

    rt_mod = graph_executor.create(factory.get_graph_json(), lib, tvm.cuda(0))
    rt_mod.set_input(**factory.get_params())
    print(rt_mod.benchmark(tvm.cuda(0), min_repeat_ms=500, end_to_end=False))
    print("Tune time: {}".format(end_time))

def run_from_prebuilt(prefix, arch):
    lib_path = os.path.join(prefix, "model.so")
    with open(os.path.join(prefix, "graph.json")) as f:
        graph_json = f.read()
    # with open(os.path.join(prefix, "graph.params"), "rb") as f_params:
    #     params = f_params.read()
    loaded_lib = tvm.runtime.load_module(lib_path)
    module = debug_executor.create(graph_json, loaded_lib, tvm.cuda(0))
    # module.load_params(params)
    print(module.benchmark(tvm.cuda(0), min_repeat_ms=500, end_to_end=False))
    input_shape = (8, 3, 512, 512)
    # input_shape = (8, 4, 64, 64)    
    dtype='float16'
    input_data = tvm.nd.array(np.ones(input_shape).astype(dtype))
    # module.set_input("input.1", input_data)
    # module.run()
    # outputs = []
    # for i in range(module.get_num_outputs()):
    #     out = module.get_output(i).asnumpy()
    #     outputs.append(out)
    # print(outputs)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='vae_encoder')
    parser.add_argument('--arch', type=str, default="cuda")
    parser.add_argument('--cublas', action="store_true")
    parser.add_argument('--cudnn', action="store_false")
    parser.add_argument('--nhwc', action="store_false")
    parser.add_argument('--use_async', action="store_true")
    args = parser.parse_args()
    arch = ladder.arch.__getattribute__(args.arch)()
    # name = args.prefix
    # path = models[name]
    # log_path = "progress/e2e/" + name + "/" + fname
    # print("Testing model: {}".format(name))
    # run(path, arch, async_propagate=args.use_async)
    # run_from_prebuilt(log_path + "_async", arch)
    run_from_prebuilt(args.prefix, arch)