# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tvm
import ladder
from ladder.graph import IRNode, OutputNode
from ladder.policy import *
from tvm import relay
import os.path as osp
from tvm.contrib.target.onnx import to_onnx
from tvm.relay.testing import run_infer_type
from tvm.contrib import graph_executor
import os
from tvm.script import tir as T
from tvm import te
import logging
ladder.set_log_level(logging.DEBUG)

# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/" + fname

arch = "cuda"
arch = ladder.arch.__getattribute__(arch)()

M = N = 16384
# shapes = ft_kernel
perf_map = {}
A = te.placeholder((M, N), name='A', dtype='float16')
B = te.compute((M, N), lambda i, j: A[i, j], name='B')

input_args = [A,]
output_args = [B,]
node = IRNode([None for _ in input_args], input_args+output_args, "ladder_matmul")
output_nodes = [OutputNode(node)]
policy = DefaultPolicy(output_nodes, arch)
configs = policy.emit_config(20)

compile_results = []
cgen = ladder.CodeGenerator()
for config in configs:
    cpresult = cgen.compile(output_nodes, config, "cuda", kernel_name="Fused")
    compile_results.append(cpresult)
ladder.utils.compile_and_load_parallel(compile_results, arch)
best_latency = 10000
best = None
values = []
for cpresult in compile_results:
    print(cpresult.config)
    code = cpresult.code
    if cpresult.lib is None:
        latency = 10000
    else:
        latency = cpresult.profile()
    values.append(latency)
    if latency < best_latency:
        best_latency = latency
        best = cpresult
    print(latency)

print("top1: {} \ttop10: {}".format(values[0], min(values)))
print("-" * 80, flush=True)
print("best config: {}".format(best.config))
print("best latency: {}".format(best_latency))

