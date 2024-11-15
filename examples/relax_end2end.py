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

import numpy as np
import os
from typing import Dict
import time
import bitblas
from bitblas import tvm as tvm
from tvm import relay, relax, runtime, transform
from tvm.relax.testing import relay_translator, nn
from tvm.target.target import Target
import tvm.relay.testing
from tvm.ir.module import IRModule
from bitblas.relax import ApplyDefaultSchedule, ApplyFastTuning

fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# get current file path
log_path = os.path.dirname(os.path.abspath(__file__)) + "/progress/" + fname

count = 0

bitblas.set_log_level("Debug")


def write_code(code, path, fname):
    global count
    fname = str(count) + "." + fname
    count += 1
    if not os.path.exists(path):
        os.makedirs(path)
    fname = os.path.join(path, fname)
    with open(fname, "w") as f:
        f.write(code)


def write_sch(sch, path, fname):
    py_fname = fname + ".py"
    write_code(sch.mod["main"].script(), path, py_fname)
    cu_fname = fname + ".cu"
    write_code(sch.mod.astext(), path, cu_fname)


def write_mod(mod, path, fname):
    py_fname = fname + ".py"
    write_code(mod.script(show_meta=False), path, py_fname)
    cu_fname = fname + ".cu"
    write_code(mod.astext(show_meta_data=False), path, cu_fname)


def get_network(name, batch_size, layout="NHWC", dtype="float32"):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-") or name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape)
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mlp":
        mod, params = relay.testing.mlp.get_workload(
            batch_size=batch_size, image_shape=image_shape, dtype=dtype)

    return mod, params, input_shape, output_shape


# Define the neural network and compilation target.
network = "mlp"
# network = "resnet-18"
batch_size = 128
layout = "NHWC"
# Path to cross compiler
target = tvm.target.Target("nvidia/nvidia-a100")
dtype = "float32"

relay_mod, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)


def apply_opt_before_tuning(relay_mod: IRModule, params: Dict[str, runtime.NDArray],
                            target: Target):
    with transform.PassContext(opt_level=3):
        main_func = relay_mod["main"]
        bind_main_func = relay.build_module.bind_params_by_name(main_func, params)
        relay_mod = IRModule.from_expr(bind_main_func)
        write_mod(relay_mod, log_path, "create_mod")
        relay_mod = relay.transform.SimplifyInference()(relay_mod)
        write_mod(relay_mod, log_path, "SimplifyInference")
        relay_mod = relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]})(relay_mod)
        write_mod(relay_mod, log_path, "ConvertLayout")
        relay_mod = relay.transform.FoldConstant()(relay_mod)
        write_mod(relay_mod, log_path, "FoldConstant")
        relay_mod = relay.transform.FoldScaleAxis()(relay_mod)
        write_mod(relay_mod, log_path, "FoldScaleAxis")
        relay_mod = relay.transform.CanonicalizeOps()(relay_mod)
        write_mod(relay_mod, log_path, "CanonicalizeOps")
        relay_mod = relay.transform.AlterOpLayout()(relay_mod)
        write_mod(relay_mod, log_path, "AlterOpLayout")
        relay_mod = relay.transform.FoldConstant()(relay_mod)
        write_mod(relay_mod, log_path, "FoldConstant")

        # opt_level=2 and select_impl_strategy are required for avoiding winograd lowering
        relax_mod = relay_translator.from_relay(
            relay_mod["main"],
            opt_level=2,
            target=target,
            append_op_attrs=True,
            select_impl_strategy="first")
        write_mod(relax_mod, log_path, "relay_translator_relax")
        relax_mod = relax.transform.AnnotateTIROpPattern()(relax_mod)
        write_mod(relax_mod, log_path, "AnnotateTIROpPattern")
        relax_mod = relax.transform.FuseOps()(relax_mod)
        write_mod(relax_mod, log_path, "FuseOps")
        relax_mod = relax.transform.FuseTIR()(relax_mod)
        write_mod(relax_mod, log_path, "FuseTIR")
    return relax_mod


relax_mod = apply_opt_before_tuning(relay_mod, params, target)
start_tune_time = time.time()
relax_mod = ApplyFastTuning(topk=20, target=target, parallel_build=True)(relax_mod)
end_tune_time = time.time()

write_mod(relax_mod, log_path, "ApplyFastTuning")
print("Time cost of Fast Dlight tuniing: {:.3f} s".format((end_tune_time - start_tune_time)))

with target:
    schedule_rules = [
        bitblas.gpu.Matmul(),
        bitblas.gpu.GEMV(),
        bitblas.gpu.Reduction(),
        bitblas.gpu.GeneralReduction(),
        bitblas.gpu.Fallback(),
    ]
    for rule in schedule_rules:
        relax_mod = ApplyDefaultSchedule(rule)(relax_mod)

write_mod(relax_mod, log_path, "ApplyFastTuning")

relax_mod = relax.transform.RunCodegen()(relax_mod)

write_mod(relax_mod, log_path, "run_codegen")

relax_mod = tvm.tir.transform.MakePackedAPI()(relax_mod)
write_mod(relax_mod, log_path, "make_packed_api")

ex = relax.build(relax_mod, target)
write_code(ex.mod.imported_modules[0].imported_modules[0].get_source(), log_path, "tmp.cu")

device = tvm.cuda(0)
vm = relax.VirtualMachine(ex, device)

# init parameters
params = nn.init_params(relax_mod)

input_args = []

input_args.append(tvm.nd.array(np.random.uniform(-1, 1, size=input_shape).astype(dtype), device))

res = vm["main"](*input_args)

print(res)

device.sync()

start = time.time()

for _ in range(10):
    vm["main"](*input_args)

device.sync()

end = time.time()

print("Time cost is: ", (end - start) * 100, "ms")
