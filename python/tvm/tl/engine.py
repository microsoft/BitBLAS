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
"""The compiler for TL programs."""

import os
import os.path as osp
import tvm
from tvm import tir, tl, relay
from tvm.contrib import nvcc


def is_device_call(func: tir.PrimFunc):
    return bool(func.attrs and "calling_conv" in func.attrs and func.attrs["calling_conv"] == 2)


def is_host_call(func: tir.PrimFunc):
    return not is_device_call(func)


@tvm.register_func("tvm_tl_cuda_compile", override=True)
def tvm_callback_cuda_compile(code, target):
    tvm_root = osp.join(osp.dirname(__file__), "../../..")
    tl_template_path = osp.abspath(osp.join(tvm_root, "src/tl"))
    if "TL_CUTLASS_PATH" in os.environ:
        cutlass_path = os.environ["TL_CUTLASS_PATH"]
    else:
        cutlass_path = osp.abspath(osp.join(tvm_root, "3rdparty/cutlass/include"))
    compute_version = "".join(nvcc.get_target_compute_version(target).split("."))

    # special handle for Hopper
    if compute_version == "90":
        arch = [f"-arch=sm_90a"]
        format = "cubin"
    else:
        arch = [f"-arch=sm_{compute_version}"]
        format = "cubin"

    ptx = nvcc.compile_cuda(
        code,
        format,
        arch,
        options=[
            "-std=c++17",
            "--use_fast_math",
            "-I" + tl_template_path,
            "-I" + cutlass_path,
        ],
    )
    # with open("save.ptx", "wb") as f:
    #     f.write(ptx)
    return ptx


def extrac_params(func: tir.PrimFunc):
    buffers = [func.buffer_map[var] for var in func.params]
    tensor_types = [relay.TensorType(buffer.shape, buffer.dtype) for buffer in buffers]
    return tensor_types

@tvm.register_func(func_name="tvm_callback_cuda_postproc", override=True)
def tvm_callback_cuda_postproc(code, _):
    code = code.replace("""original code""", 
"""modified code""")
    return code

def lower(func):
    params = extrac_params(func)
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})

    target_host = tvm.target.Target("llvm -keys=cpu")
    target = tvm.target.Target("cuda", target_host)
    mod = tir.transform.BindTarget(target)(mod)

    # print('-'*100 + '\n' + 'after BindTarget\n' + '-'*100)
    # print(mod)

    mod = tl.transform.FrontendLegalize()(mod)

    # print('-'*100 + '\n' + 'after FrontendLegalize\n' + '-'*100)
    # print(mod)

    mod = tir.transform.Simplify()(mod)

    # print('-'*100 + '\n' + 'after Simplify\n' + '-'*100)
    # print(mod)

    mod = tl.transform.LayoutInference()(mod)
 
    # print('-'*100 + '\n' + 'after LayoutInference\n' + '-'*100)
    # print(mod)

    mod = tl.transform.LowerTileOp()(mod)

    # print('-'*100 + '\n' + 'after LowerTileOp\n' + '-'*100)
    # print(mod)

    mod = tir.transform.Simplify()(mod)

    # print('-'*100 + '\n' + 'after Simplify\n' + '-'*100)
    # print(mod)

    if target.arch == "sm_90":
        mod = tl.transform.MultiVersionBuffer()(mod)
        
        # print('-'*100 + '\n' + 'after MultiVersionBuffer\n' + '-'*100)
        # print(mod)

        mod = tl.transform.WarpSpecialized()(mod)
        
        # print('-'*100 + '\n' + 'after WarpSpecialized\n' + '-'*100)
        # print(mod)

        mod = tl.transform.InjectSoftwarePipeline()(mod)
        
        # print('-'*100 + '\n' + 'after InjectSoftwarePipeline\n' + '-'*100)
        # print(mod)
        
        mod = tir.transform.LowerOpaqueBlock()(mod)
        
        # print('-'*100 + '\n' + 'after LowerOpaqueBlock\n' + '-'*100)
        # print(mod)

        # mod = tl.transform.WarpSpecializedPipeline()(mod)

        # print('-'*100 + '\n' + 'after WarpSpecializedPipeline\n' + '-'*100)
        # print(mod)
        
        mod = tl.transform.InjectFenceProxy()(mod)
    
        # print('-'*100 + '\n' + 'after InjectFenceProxy\n' + '-'*100)
        # print(mod)

    else:
        mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        mod = tl.transform.PipelinePlanning()(mod)
        mod = tl.transform.InjectSoftwarePipeline()(mod)

    mod = tir.transform.LowerOpaqueBlock()(mod)
    mod = tir.transform.FlattenBuffer()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tir.transform.Simplify()(mod)

    mod = tir.transform.VectorizeLoop()(mod)
    mod = tir.transform.StorageRewrite()(mod)
    mod = tir.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.RemoveNoOp()(mod)
    mod = tir.transform.RewriteUnsafeSelect()(mod)
    mod = tir.transform.HoistIfThenElse()(mod)

    mod = tir.transform.VerifyMemory()(mod)
    mod = tir.transform.AnnotateEntryFunc()(mod)
    mod = tir.transform.ThreadSync("shared")(mod)
    mod = tir.transform.ThreadSync("shared.dyn")(mod)
    mod = tir.transform.MergeDynamicSharedMemoryAllocations()(mod)
    mod = tl.transform.LowerHopperIntrin()(mod)
    mod = tir.transform.InjectPTXAsyncCopy()(mod)

    mod = tir.transform.AnnotateDeviceRegions()(mod)
    mod = tir.transform.SplitHostDevice()(mod)
    mod = tir.transform.MakePackedAPI()(mod)
    mod = tir.transform.LowerDeviceKernelLaunch()(mod)

    host_mod = tir.transform.Filter(is_host_call)(mod)
    host_mod = tir.transform.BindTarget(target_host)(host_mod)
    host_mod = tir.transform.FP8StorageLegalize()(host_mod)
    host_mod = tir.transform.BF16StorageLegalize()(host_mod)
    host_mod = tir.transform.LowerTVMBuiltin()(host_mod)
    host_mod = tir.transform.LowerCustomDatatypes()(host_mod)
    host_mod = tir.transform.LowerIntrin()(host_mod)
    host_mod = tir.transform.LowerDeviceStorageAccessInfo()(host_mod)
    host_mod = tir.transform.CombineContextCall()(host_mod)
    # host_code = tvm._ffi.get_global_func("target.build.c")(host_mod, target_host).get_source()
    # print("=" * 100)
    # print("host code:")
    # print("=" * 100)
    # print(host_code)
    host_mod = tvm._ffi.get_global_func("target.build.llvm")(host_mod, target)

    device_mod = tir.transform.Filter(is_device_call)(mod)
    device_mod = tir.transform.LowerDeviceStorageAccessInfo()(device_mod)
    device_mod = tir.transform.LowerIntrin()(device_mod)
    device_mod = tir.transform.Simplify()(device_mod)
    # device_code = tvm._ffi.get_global_func("target.build.tl_debug_codegen")(device_mod, target)
    # print("=" * 100)
    # print("device code:")
    # print("=" * 100)
    # print(device_code)
    device_mod = tvm._ffi.get_global_func("target.build.tl")(device_mod, target)

    host_mod.import_module(device_mod)
    return host_mod, params
