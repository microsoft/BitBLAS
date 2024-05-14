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
import logging
ladder.schedule.enable_schedule_dump()
ladder.set_log_level(logging.DEBUG)

model_path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", "models")

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
parser.add_argument('--fake_quant', type=int, default=-1)
parser.add_argument("--fast_decoding", action="store_true", help="Use fast decoding mode.", default=False)
parser.add_argument('--batch', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=1)
parser.add_argument('--bits', type=int, default=4)
parser.add_argument('--convert_int', action="store_true")
parser.add_argument('--format', type=str, default='int')
parser.add_argument('--async_propagation', action="store_true", help="Use async propagation and async instructions, which should be only enabled on data center GPUs with async copy instructions.")
parser.add_argument("--prebuilt_path", type=str, default=None, help="Path to the prebuilt model. If set, the script will run from the prebuilt model.")

args = parser.parse_args()

def run(prefix, arch, async_propagate, fake_quant, quant_config, convert_int):
    global log_path
    if async_propagate:
        log_path += "_async"
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
    if fake_quant > -1:
        # quant_candidates = [
        #     (8192, 8192, True),
        #     (28672, 8192, True),
        #     (8192, 28672, True),
        #     (8192, 1024, True),
        # ]
        # set quant_candidates to None because it's only one decode layer, didn't have lm_head and word embedding.
        mod = ladder.relay.transform.LadderFakeQuant(quant_type=fake_quant, quant_config=quant_config, convert_int=convert_int)(mod)
        write_mod(mod, log_path, "LadderFakeQuant")
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
    if args.fast_decoding:
        mod = ladder.relay.transform.AnnotateFastDecoding()(mod)
        write_mod(mod, log_path, "AnnotateFastDecoding")

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
    
    loaded_lib = tvm.runtime.load_module(lib_path)
    module = debug_executor.create(graph_json, loaded_lib, tvm.cuda(0))
    
    print(module.benchmark(tvm.cuda(0), min_repeat_ms=500, end_to_end=False))

    module.run()



if __name__ == "__main__":
    quant_config = {
            'format':args.format,
            'bits': args.bits,
            'group_size': -1,
    }
    arch = ladder.arch.__getattribute__(args.arch)()
    name = args.prefix
    if args.prefix == "llama2-70b":
        path = f'{model_path}/llama_70b/llama2_70b_layer1_seq{args.seq_len}_bs{args.batch}/model.onnx'
    elif args.prefix == "bloom-176b":
        path = f'{model_path}/bloom_176b/bloom-176b_layer1_seq{args.seq_len}_bs{args.batch}/model.onnx'
    else:
        path = args.prefix
        name = path.split("/")[-1]
    # path = f'./models/llama_70b/llama2_70b_layer1_seq{args.seq_len}_bs{args.batch}/model.onnx'

    log_path = "progress/e2e/" + name + "/" + fname
    if args.fake_quant > -1:
        log_path += f'_fq_{args.fake_quant}_{quant_config["format"]}_{quant_config["bits"]}_{quant_config["group_size"]}_bs{args.batch}_seq{args.seq_len}_ci_{args.convert_int}'
    else:
        log_path += f'_bs{args.batch}_seq{args.seq_len}'
    prebuilt_path = args.prebuilt_path
    if prebuilt_path:
        print(f"Running from prebuilt model: {prebuilt_path}")
        run_from_prebuilt(prebuilt_path, arch)
    else:
        print("Testing model: {}".format(name))
        run(path, arch, async_propagate=args.async_propagation, fake_quant=args.fake_quant, quant_config=quant_config, convert_int=args.convert_int)

    os.system("rm -rf /tmp/tvmdbg_*")