# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm
from tvm import IRModule
from tvm.runtime import ndarray
from tvm.driver import lower
from tvm.target import Target
from typing import Tuple, List
from tvm import tir
from bitblas import tilelang as tilelang
from tilelang.engine import is_device_call


def get_annotated_device_mod_from_tl(mod: IRModule, target: Target) -> "IRModule":
    target_host = tvm.target.Target("llvm -keys=cpu")
    target = tvm.target.Target(target, target_host)
    mod = tir.transform.BindTarget(target)(mod)

    mod = tilelang.transform.FrontendLegalize()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tilelang.transform.LayoutInference()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = tir.transform.Simplify()(mod)

    if target.arch == "sm_90":
        mod = tilelang.transform.WarpSpecializedPipeline()(mod)
    else:
        mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        mod = tilelang.transform.PipelinePlanning()(mod)
        mod = tilelang.transform.InjectSoftwarePipeline()(mod)

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
    # TODO(lei): This is a hack to make sure the
    # thread level allreduce pass can be applied
    # in TL. As Tl only use one thread dimension
    # the var binding information will be lost
    # in the lowering process with Legalization
    # and Simplify pass.
    # We can find a way better to create var instead
    # of putting the LowerThreadAllreduce before
    # the Legalization.
    mod = tir.transform.LowerThreadAllreduce()(mod)
    mod = tir.transform.ThreadSync("shared.dyn")(mod)
    mod = tilelang.transform.LowerHopperIntrin()(mod)
    mod = tir.transform.InjectPTXAsyncCopy()(mod)

    mod = tir.transform.AnnotateDeviceRegions()(mod)
    mod = tir.transform.SplitHostDevice()(mod)
    mod = tir.transform.MergeSharedMemoryAllocations()(mod)
    mod = tir.transform.MakePackedAPI()(mod)
    mod = tir.transform.LowerDeviceKernelLaunch()(mod)

    device_mod = tir.transform.Filter(is_device_call)(mod)

    return device_mod


def get_annotated_device_mod_from_tir(mod: IRModule, target: Target) -> "IRModule":
    """
    Lower the given IRModule and create a device module for the specified target.

    Parameters:
    - mod: The input IRModule.
    - target: The compilation target.

    Returns:
    - A device module ready for execution.
    """
    input_mod = lower(mod)
    target_input_mod = {target: input_mod}
    annotated_mods = {}
    runtime = None
    target_host = None
    for tgt, mod in target_input_mod.items():
        if not isinstance(tgt, (str, Target)):
            raise ValueError("The key of inputs must be str or "
                             "Target when inputs is dict.")
        if not isinstance(mod, tvm.IRModule):
            raise ValueError("inputs must be Schedule, IRModule, "
                             "or dict of str to IRModule.")
        annotated_mods[tgt] = mod.with_attr("runtime", runtime)
    annotated_mods, target_host = Target.canon_target_map_and_host(annotated_mods, target_host)
    if not target_host:
        for tar, _ in annotated_mods.items():
            device_type = ndarray.device(tar.kind.name, 0).device_type
            if device_type == ndarray.cpu(0).device_type:
                target_host = tar
                break
    if not target_host:
        target_host = "llvm" if tvm.runtime.enabled("llvm") else "stackvm"
    annotated_mods, target_host = Target.canon_target_map_and_host(annotated_mods, target_host)
    for target, mod in annotated_mods.items():
        mixed_mod_passes = tvm.get_global_func("driver.mixed_mod_passes")
        device_mod_passes = tvm.get_global_func("driver.device_mod_passes")
        mod = mixed_mod_passes(mod, target)(mod)
        device_mod = device_mod_passes(mod, target)(mod)
    return device_mod


def get_annotated_device_mod(mod: IRModule, target: Target, backend="tir") -> "IRModule":
    if backend == "tir":
        return get_annotated_device_mod_from_tir(mod, target)
    elif backend == "tl":
        return get_annotated_device_mod_from_tl(mod, target)
    else:
        raise ValueError("Unsupported backend: {}".format(backend))


def get_thread_block_information(mod: IRModule) -> Tuple[List[int], List[int]]:
    """
    Extracts the thread block and grid dimensions for the reduction block within a given IRModule.

    Parameters:
    - mod: The input IRModule from which to extract thread block and grid information.

    Returns:
    A tuple containing two lists:
    - The first list contains the dimensions of the thread block (threadIdx.x, threadIdx.y, threadIdx.z).
    - The second list contains the dimensions of the grid (blockIdx.x, blockIdx.y, blockIdx.z).
    """

    # Initialize the schedule from the IRModule
    sch = tvm.tir.Schedule(mod)

    # Get the root block and its child blocks
    root_block = sch.get_block("root")
    child_blocks = sch.get_child_blocks(root_block)

    # Initialize default block and grid dimensions (1, 1, 1)
    block_dims, grid_dims = [1, 1, 1], [1, 1, 1]

    for block in child_blocks:
        # Get the loops surrounding the main block
        loops = sch.get_loops(block)

        # Iterate over each loop to extract thread and block bindings
        for loop in loops:
            stmt = sch.get(loop)
            thread_binding = stmt.thread_binding
            extent = int(stmt.extent)

            # Skip loops without thread binding
            if thread_binding:
                if "threadIdx" in thread_binding.thread_tag:
                    block_dims["xyz".index(thread_binding.thread_tag[-1])] = extent
                elif "blockIdx" in thread_binding.thread_tag:
                    grid_dims["xyz".index(thread_binding.thread_tag[-1])] = extent

    return block_dims, grid_dims
