# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import bitblas
from bitblas.gpu.matmul_analysis import get_tensorized_func_and_tags
from bitblas.base.roller.policy import TensorCorePolicy, DefaultPolicy
from bitblas.base.arch import RDNA
from bitblas.base.utils import apply_and_build
from bitblas.builder.lib_generator import LibraryGenerator
from bitblas.builder.wrapper import TIRWrapper
from bitblas import set_log_level
import tvm
from tvm.script import tir as T
import logging

set_log_level(logging.DEBUG)

M = N = 1024
in_dtype = "float32"
out_dtype = "float32"

@tvm.script.ir_module
class Add:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, N], dtype=in_dtype)
        B = T.match_buffer(b, [M, N], dtype=in_dtype)
        C = T.match_buffer(c, [M, N], dtype=out_dtype)

        for i, j in T.grid(M, N):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                with T.init():
                    C[vi, vj] = tvm.tir.const(0, out_dtype)
                C[vi, vj] = A[vi, vj].astype(out_dtype) + B[vi, vj].astype(out_dtype)

ir_module = Add

func = ir_module["main"]
target = tvm.target.Target("hip")
arch = RDNA(target)

policy = DefaultPolicy(func=func, arch=arch)
configs = policy.emit_config(topk=20)
cpresults, best = apply_and_build(func, configs, arch, parallel_build=True)

#print(best.sch.mod)

#test builder
backend = TIRWrapper(arch=arch)
backend.assign_optimized_module(best.sch.mod)
is_dynamic = False
wrapped_code = backend.wrap(best.code, is_dynamic=is_dynamic)
assert "void call" in wrapped_code

lib_generator = LibraryGenerator(arch=arch)
lib_generator.update_lib_code(wrapped_code)
lib_generator.compile_lib()

print("-----------------------")
#print(wrapped_code)