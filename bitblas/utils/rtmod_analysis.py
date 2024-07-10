# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm
from tvm import IRModule
from tvm.runtime import ndarray
from tvm.driver import lower
from tvm.target import Target
from typing import Tuple, List


def get_annotated_device_mod(mod: IRModule, target: Target) -> "IRModule":
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
