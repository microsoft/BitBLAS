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
from tvm import te, IRModule
from tvm.script import tir as T
from tvm import meta_schedule as ms
import os
from tvm.meta_schedule.testing import tir_tensor_intrin_fp16
import time
from utils import *
def write_code(code, path, fname):
    # if path doesn't exist, then create
    if not os.path.exists(path):
        os.makedirs(path)
    # join path and fname
    fname = os.path.join(path, fname)
    with open(fname, "w") as f:
        f.write(code)

parser = argparse.ArgumentParser()

parser.add_argument("--N", type=int, default=1)
parser.add_argument("--C", type=int, default=64)
parser.add_argument("--H", type=int, default=56)
parser.add_argument("--W", type=int, default=56)
parser.add_argument("--F", type=int, default=64)
parser.add_argument("--K", type=int, default=3)
parser.add_argument("--S", type=int, default=1)
parser.add_argument("--P", type=int, default=1)
parser.add_argument("--trails", type=int, default=1000)

args = parser.parse_args()

N = args.N
C = args.C
H = args.H
W = args.W
F = args.F
K = args.K
S = args.S
P = args.P

trails = args.trails
# tuning config
target = tvm.target.Target("nvidia/geforce-rtx-4090")

def cuda_build(mod, target, _params):
    from tvm.driver import build as tvm_build

    with tvm.transform.PassContext(config={"tir.predicate_opt": True}):
        return tvm_build(mod, target=target)


def conv2d(N, C, H, W, K, R, S, stride, padding, dilation, layout, in_dtype, out_dtype):
    kH = (R - 1) * dilation + 1
    kW = (S - 1) * dilation + 1
    pH = H + 2 * padding
    pW = W + 2 * padding
    if layout == "nchw":
        A = tvm.te.placeholder([N, C, H, W], dtype=in_dtype, name="A")
        B = tvm.te.placeholder([K, C, R, S], dtype=in_dtype, name="B")

        Pad = tvm.te.compute(
            [N, C, pH, pW],
            lambda n, c, h, w: tvm.tir.if_then_else(
                tvm.tir.all(h >= padding, h - padding < H, w >= padding, w - padding < W),
                A[n, c, h - padding, w - padding],
                tvm.tir.const(0.0, A.dtype),
            ),
            name="Pad",
        )

        rc = tvm.te.reduce_axis([0, C], name="rc")
        rr = tvm.te.reduce_axis([0, kH], name="rr")
        rs = tvm.te.reduce_axis([0, kW], name="rs")

        P = (pH - kH) // stride + 1
        Q = (pW - kW) // stride + 1
        Conv = tvm.te.compute(
            [N, K, P, Q],
            lambda n, k, p, q: tvm.te.sum(
                (
                    Pad[n, rc, p * stride + rr * dilation, q * stride + rs * dilation]
                    * B[k, rc, rr, rs]
                ).astype(out_dtype),
                axis=[rc, rr, rs],
            ),
            name="Conv",
        )
    elif layout == "nhwc":
        A = tvm.te.placeholder([N, H, W, C], dtype=in_dtype, name="A")
        B = tvm.te.placeholder([R, S, C, K], dtype=in_dtype, name="B")

        Pad = tvm.te.compute(
            [N, pH, pW, C],
            lambda n, h, w, c: tvm.tir.if_then_else(
                tvm.tir.all(h >= padding, h - padding < H, w >= padding, w - padding < W),
                A[n, h - padding, w - padding, c],
                tvm.tir.const(0.0, A.dtype),
            ),
            name="Pad",
        )

        rc = tvm.te.reduce_axis([0, C], name="rc")
        rr = tvm.te.reduce_axis([0, kH], name="rr")
        rs = tvm.te.reduce_axis([0, kW], name="rs")

        P = (pH - kH) // stride + 1
        Q = (pW - kW) // stride + 1
        Conv = tvm.te.compute(
            [N, P, Q, K],
            lambda n, p, q, k: tvm.te.sum(
                (
                    Pad[n, p * stride + rr * dilation, q * stride + rs * dilation, rc]
                    * B[rr, rs, rc, k]
                ).astype(out_dtype),
                axis=[rr, rs, rc],
            ),
            name="Conv",
        )
    elif layout == "hwnc":
        A = tvm.te.placeholder([H, W, N, C], dtype=in_dtype, name="A")
        B = tvm.te.placeholder([R, S, C, K], dtype=in_dtype, name="B")

        Pad = tvm.te.compute(
            [pH, pW, N, C],
            lambda h, w, n, c: tvm.tir.if_then_else(
                tvm.tir.all(h >= padding, h - padding < H, w >= padding, w - padding < W),
                A[h - padding, w - padding, n, c],
                tvm.tir.const(0.0, A.dtype),
            ),
            name="Pad",
        )

        rc = tvm.te.reduce_axis([0, C], name="rc")
        rr = tvm.te.reduce_axis([0, kH], name="rr")
        rs = tvm.te.reduce_axis([0, kW], name="rs")

        P = (pH - kH) // stride + 1
        Q = (pW - kW) // stride + 1
        Conv = tvm.te.compute(
            [P, Q, N, K],
            lambda p, q, n, k: tvm.te.sum(
                (
                    Pad[p * stride + rr * dilation, q * stride + rs * dilation, n, rc]
                    * B[rr, rs, rc, k]
                ).astype(out_dtype),
                axis=[rr, rs, rc],
            ),
            name="Conv",
        )
    else:
        raise RuntimeError(f"Unkonwn layout for conv2d: {layout}")
    return [A, B, Conv]


conv_args = conv2d(
    N = N,
    C = C,
    H = H,
    W = W,
    K = F,
    R = K,
    S = K,
    stride = S,
    padding = P,
    dilation = 1,
    layout = "nhwc",
    in_dtype = "float16",
    out_dtype = "float16",
)


func = te.create_prim_func(conv_args)

mod = IRModule.from_expr(func)
def tune():
    print(mod.script())
    workdir = f"./logs/conv2d_nhwc_N{N}_C{C}_H{H}_W{W}_F{F}_K{K}_S{S}_P{P}/"
    start = time.time()
    if N == 1:
        database = ms.tune_tir(
            mod=mod,
            target=target,
            config=get_search_config(trails, trails),
            work_dir=workdir,
        )
    else:
        database = ms.tune_tir(
            mod=mod,
            target=target,
            config=get_search_config(trails, trails),
            work_dir=workdir,
            sch_rules=sch_rules_tensor_core,
            postprocs=postprocs_tensor_core,
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
