# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-docstring
import argparse
import tvm
from tvm.script import tir as T
from tvm import meta_schedule as ms
import os
from tvm.meta_schedule.testing.space_generation import (
    check_sketches,
    generate_design_space,
    get_rules,
)
from tvm.tir.tensor_intrin.cuda import get_wmma_intrin_group
import time

def write_code(code, path, fname):
    # if path doesn't exist, then create
    if not os.path.exists(path):
        os.makedirs(path)
    # join path and fname
    fname = os.path.join(path, fname)
    with open(fname, "w") as f:
        f.write(code)

parser = argparse.ArgumentParser()

parser.add_argument("--M", type=int, default=16384)
parser.add_argument("--N", type=int, default=16384)
parser.add_argument("--K", type=int, default=16384)
parser.add_argument("--trails", type=int, default=1000)

args = parser.parse_args()
M = args.M
N = args.N
K = args.K
trails = args.trails
# tuning config
target = tvm.target.Target("nvidia/nvidia-a100")

def cuda_build(mod, target, _params):
    from tvm.driver import build as tvm_build

    with tvm.transform.PassContext(config={"tir.predicate_opt": True}):
        return tvm_build(mod, target=target)


@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K], dtype="float16")
        B = T.match_buffer(b, [N, K], dtype="float16")
        C = T.match_buffer(c, [M, N], dtype="float16")

        for i, j, k in T.grid(M, N, K):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float16(0)
                C[vi, vj] = C[vi, vj] + \
                    A[vi, vk].astype("float16") * B[vj, vk].astype("float16")


def tune():
    mod = MyModule
    print(mod.script())
    workdir = f"./logs/gemm_M{M}N{N}K{K}_nt/"
    start = time.time()
    if M == 1:
        database = ms.tune_tir(
            mod=mod,
            target=target,
            max_trials_global=trails,
            num_trials_per_iter=16,
            work_dir=workdir,
        )
    else:
        database = ms.tune_tir(
            mod=mod,
            target=target,
            max_trials_global=trails,
            num_trials_per_iter=16,
            work_dir=workdir,
            space=ms.space_generator.PostOrderApply(
                sch_rules="cuda-tensorcore",
                postprocs="cuda-tensorcore",
                mutator_probs="cuda-tensorcore"
            )
        )
    sch = ms.tir_integration.compile_tir(
        database=database, mod=mod, target=target)
    if sch is None:
        print("No valid schedule found!")
        exit()
    end = time.time()
    print(sch.mod.script())
    print(sch.trace)
    cuda_mod = tvm.build(sch.mod, target="cuda")
    write_code(
        cuda_mod.imported_modules[0].get_source(), workdir, "cuda_mod.cu")

    print("Time cost: %f" % (end - start))
    # get seconds
    time_seconds = end - start
    with open(workdir + "time_cost.txt", "w") as f:
        f.write(str(time_seconds))

if __name__ == "__main__":
    tune()
