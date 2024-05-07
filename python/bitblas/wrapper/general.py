# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tvm
from typing import Optional, List, Dict, Union
from tvm import IRModule
from bitblas import TileDevice
from tvm.runtime import ndarray
from bitblas.utils import match_global_kernel
import re
import ctypes
import os
import tempfile
import subprocess
import logging
from tvm.driver import lower
from tvm.target import Target

logger = logging.getLogger(__name__)

_TYPE_MAP = {
    "float32": "float",
    "float16": "half",
    "bfloat16": "__nv_bfloat162",
    "e4m3_float8": "__nv_fp8_e4m3",
    "e5m2_float8": "__nv_fp8_e5m2",
    "float64": "double",
    "int64": "int64_t",
    "int32": "int",
    "uint32": "unsigned int",
    "bool": "int8_t",
    "int8": "int8_t",
    "uint8": "uint8_t",
    "int16": "int16_t",
    "uchar": "uint8_t",
}


def get_annotated_device_mod(mod: IRModule, target: Target):
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


def get_thread_block_information(mod: IRModule):
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


class CUDASourceWrapper(object):

    def __init__(self, optimized_mod: IRModule, source: str, arch: TileDevice):
        self.mod = optimized_mod
        self.arch = arch
        self.source = source
        self.function_name: Optional[str] = None
        self.dynamic_smem_buf: Optional[int] = None
        self.block_info: Union[List[int], Dict] = [1, 1, 1]
        self.grid_info: Union[List[int], Dict] = [1, 1, 1]
        self.parse_source_information()
        self.src_name: Optional[str] = None
        self.lib_name: Optional[str] = None
        self.lib_code: Optional[str] = self.update_lib_code(source)

    def load_lib(self):
        return ctypes.CDLL(self.lib_name)

    def remove_lib(self):
        if self.lib_name:
            os.remove(self.lib_name)
        self.lib_name = None

    def compile_lib(self, timeout: float = None):
        arch = self.arch
        src = tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False)
        compute_version = arch.compute_capability
        lib_name = src.name.replace(".cu", ".so")

        command = [
            "nvcc",
            "-std=c++17",
            "-Xcudafe",
            "--diag_suppress=177",
            "--compiler-options",
            "'-fPIC'",
            "-lineinfo",
            "--shared",
            src.name,
            "-lcuda",
            f"-gencode=arch=compute_{compute_version},code=compute_{compute_version}",
            "-o",
            lib_name,
        ]
        src.write(self.lib_code)
        src.flush()
        try:
            ret = subprocess.run(command, timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning(f"Compilation Timeout! {command}")
            return None
        if ret.returncode != 0:
            logger.warning(f"Compilation Failed! {command}")
            return None
        self.src_name = src.name
        self.lib_name = lib_name

    def parse_source_information(self):
        device_mod = get_annotated_device_mod(self.mod, self.arch.target)
        assert (len(device_mod.functions) == 1
               ), "Only support one function in the module for static shape kernel."
        for g_var, func in device_mod.functions.items():
            self.function_name = g_var.name_hint
            attrs = func.attrs
            if "dyn_shared_memory_buf" in attrs:
                self.dynamic_smem_buf = int(attrs["dyn_shared_memory_buf"])
            if "thread_extent" in attrs:
                thread_extent = attrs["thread_extent"]
                for tag, extent in thread_extent.items():
                    if "threadIdx" in tag:
                        self.block_info["xyz".index(tag[-1])] = extent
                    elif "blockIdx" in tag:
                        self.grid_info["xyz".index(tag[-1])] = extent

    def get_dynamic_symbolic_set(self, prim_func):
        # Determine the set of dynamic symbols used in the function
        dynamic_symbolic_set = set()
        for param in prim_func.params:
            buffer = prim_func.buffer_map[param]
            for dim in buffer.shape:
                if isinstance(dim, tvm.tir.Var):
                    dynamic_symbolic_set.add(dim.name)
        return dynamic_symbolic_set

    def get_cuda_init_func(self):
        # Initialize an empty string for the CUDA function call
        call_str = """"""
        # If dynamic shared memory buffer is specified, prepare the cudaFuncSetAttribute call
        if self.dynamic_smem_buf is not None:
            call_str = """
        cudaFuncSetAttribute({},
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, {});
                """.format(self.function_name, self.dynamic_smem_buf)
        # Format the initialization function using the call_str
        init_funcs = """
    extern "C" void init() {{
        {}
    }}
            """.format(call_str)
        return init_funcs

    def update_lib_code(self, code: str):
        # Update the library code with the given code string
        self.lib_code = code
        # Find the index of the global kernel function in the code
        index = match_global_kernel(code)
        # Extract the declaration of the function starting from the found index
        declaration = code[index:].split(";")[0]

        function_name = self.function_name
        # Get the CUDA initialization function
        init_func = self.get_cuda_init_func()

        # Locate the opening brace of the function to insert arguments
        index = code.index("{", index)
        function_args = []
        # Populate the function arguments from the primary function's parameters and buffers
        for param in self.prim_func.params:
            buffer = self.prim_func.buffer_map[param]
            function_args.append({
                "name": buffer.name,
                "type": _TYPE_MAP[buffer.dtype] + "* __restrict__",
            })

        dynamic_symbolic_set = self.get_dynamic_symbolic_set(self.prim_func)
        # Add dynamic symbolic parameters as integers to the function arguments
        for dyn_sym in dynamic_symbolic_set:
            function_args.append({"name": dyn_sym, "type": "int"})

        function_args.append({"name": "stream=cudaStreamDefault", "type": "cudaStream_t"},)
        # Format the function arguments for declaration
        def_args = ", ".join([f"{arg['type']} {arg['name']}" for arg in function_args])

        def func_call_args(s, function_args):
            # Extract the function call arguments matching the function definition
            pattern = r"[,\s]*(?:\w+\s*\*+\s*__restrict__\s+)?(\w+)"
            matches = re.findall(pattern, s)
            call_args = []
            for match in matches:
                for arg in function_args:
                    if arg["name"] == match:
                        call_args.append(match)
            return call_args

        call_args = ", ".join(func_call_args(declaration, function_args))
        block_info, grid_info = self.block_info, self.grid_info

        def legalize_c(p):
            # Convert TIR expressions to legal C expressions
            # Directly convert to string since the special case handling
            # does not alter the string representation for `tvm.tir.Var` and `IntImm`.
            # Replace Python's floor division operator with C's division operator
            if isinstance(p, tvm.tir.IntImm):
                p = int(p)
            return str(p).replace("//", "/")

        # Prepare the block and grid dimensions for the CUDA kernel launch
        block_str = "dim3({}, {}, {})".format(
            legalize_c(block_info[0]),
            legalize_c(block_info[1]),
            legalize_c(block_info[2]),
        )
        grid_str = "dim3({}, {}, {})".format(
            legalize_c(grid_info[0]), legalize_c(grid_info[1]), legalize_c(grid_info[2]))
        # Determine the shared memory size, defaulting to 0 if not specified
        smem_str = 0 if self.dynamic_smem_buf is None else self.dynamic_smem_buf
        # Format the CUDA kernel launch string
        call_str = "{}<<<{}, {}, {}, stream>>>({});".format(function_name, grid_str, block_str, smem_str,
                                                       call_args)
        # Create the host function wrapper for the CUDA kernel
        host_func = """
    extern "C" void call({}) {{
        {}
    }}
        """.format(def_args, call_str)
        # Combine the source, initialization function, and host function to form the complete library code
        lib_code = self.source + init_func + host_func
        return lib_code

    @property
    def prim_func(self):
        return self.mod["main"]


class CUDASourceWrapperWithDynamic(CUDASourceWrapper):

    def __init__(self, optimized_mod: IRModule, source: str, arch: TileDevice):
        super().__init__(optimized_mod, source, arch)

    def get_cuda_init_func(self):
        # Initialize an empty string to accumulate CUDA function calls for setting dynamic shared memory
        call_str = """"""
        # Iterate over functions and their dynamic shared memory requirements
        for function_name, dynamic_smem_buf in self.dynamic_smem_buf.items():
            if dynamic_smem_buf is not None:
                # Format the cudaFuncSetAttribute call for dynamic shared memory
                call_str += """
        cudaFuncSetAttribute({},
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, {});
                    """.format(function_name, dynamic_smem_buf)
        # Define the init function that will set the attributes for each kernel
        init_funcs = """
extern "C" void init() {{
    {}
}}
            """.format(call_str)
        return init_funcs

    def create_dispatch_func(self, code, function_informations):
        # Extract the set of dynamic symbolic names used in the primary function
        dynamic_symbolic_set = self.get_dynamic_symbolic_set(self.prim_func)

        # Find the location of the global kernel function in the code
        index = match_global_kernel(code)

        # Analyze the function declaration to prepare for argument extraction
        dummy_declaration = code[index:].split(";")[0]

        function_name = self.function_name

        # Identify the start of the function body to insert arguments
        index = code.index("{", index)
        function_args = []
        # Collect function arguments based on primary function's parameters and buffer mappings
        for param in self.prim_func.params:
            buffer = self.prim_func.buffer_map[param]
            function_args.append({
                "name": buffer.name,
                "type": _TYPE_MAP[buffer.dtype] + "* __restrict__",
            })
        # Add dynamic symbols as integer arguments
        for dyn_sym in dynamic_symbolic_set:
            function_args.append({"name": dyn_sym, "type": "int"})

        function_args.append({"name": "stream=cudaStreamDefault", "type": "cudaStream_t"},)

        # Format the argument definitions for function declaration
        def_args = ", ".join([f"{arg['type']} {arg['name']}" for arg in function_args])

        def func_call_args(s: str, function_args):
            # Extract and clean the function call arguments to match the declaration
            pattern = r"[,\s]*(?:\w+\s*\*+\s*__restrict__\s+)?(\w+)"
            matches = re.findall(pattern, s)
            call_args = []
            for match in matches:
                match = re.sub(r"\d+", "", match)  # Remove numbers
                match = re.sub(r"_", "", match)  # Remove underscores
                for arg in function_args:
                    if arg["name"] == match:
                        call_args.append(match)
            return call_args

        call_args = ", ".join(func_call_args(dummy_declaration, function_args))

        def legalize_c(p):
            # Convert TIR expressions to legal C expressions
            # Directly convert to string since the special case handling
            # does not alter the string representation for `tvm.tir.Var` and `IntImm`.
            # Replace Python's floor division operator with C's division operator
            if isinstance(p, tvm.tir.IntImm):
                p = int(p)
            return str(p).replace("//", "/")

        last_range = 0
        num_items = len(function_informations)
        _call_str = """"""
        for function_name, info in function_informations.items():
            # Prepare block and grid configurations for kernel launches
            block_info, grid_info = info["block_info"], info["grid_info"]
            block_str = "dim3({}, {}, {})".format(
                legalize_c(block_info[0]),
                legalize_c(block_info[1]),
                legalize_c(block_info[2]),
            )
            grid_str = "dim3({}, {}, {})".format(
                legalize_c(grid_info[0]),
                legalize_c(grid_info[1]),
                legalize_c(grid_info[2]),
            )
            # Handle dynamic shared memory specification
            smem_str = (0 if info["dynamic_smem_buf"] is None else info["dynamic_smem_buf"])
            opt_shapes = info["opt_shapes"]
            # Generate conditional kernel launch code based on dynamic symbolic ranges
            (symbolic,) = list(dynamic_symbolic_set)
            range_str = opt_shapes[symbolic]
            if last_range == 0:
                call_str = "if ({} <= {}) {{\n\t\t\t {}<<<{}, {}, {}, stream>>>({}); \n\t\t}}\n".format(
                    symbolic,
                    range_str,
                    function_name,
                    grid_str,
                    block_str,
                    smem_str,
                    call_args,
                )
            else:
                call_str = "\t\telse if ({} <= {}) {{\n\t\t\t {}<<<{}, {}, {}, stream>>>({}); \n\t\t}}\n".format(
                    symbolic,
                    range_str,
                    function_name,
                    grid_str,
                    block_str,
                    smem_str,
                    call_args,
                )
            if last_range == num_items - 1:
                call_str += (
                    "\t\telse {{\n\t\t\t {}<<<{}, {}, {}, stream>>>({}); \n\t\t}}\n".format(
                        function_name, grid_str, block_str, smem_str, call_args))
            last_range += 1
            _call_str += call_str

        # Wrap the kernel dispatch logic in an external C function
        host_func = """
extern "C" void call({}) {{
    {}
}}
        """.format(def_args, _call_str)
        return host_func

    def parse_source_information(self):
        # Parse device module to extract execution configurations for each function
        device_mod = get_annotated_device_mod(self.mod, self.arch.target)
        block_info_map = {}
        grid_info_map = {}
        dynamic_smem_buf_map = {}
        for g_var, func in device_mod.functions.items():
            # Default block and grid configurations
            block_info = [1, 1, 1]
            grid_info = [1, 1, 1]
            function_name = g_var.name_hint
            attrs = func.attrs
            dynamic_smem_buf = None
            if "dyn_shared_memory_buf" in attrs:
                dynamic_smem_buf = int(attrs["dyn_shared_memory_buf"])
            if "thread_extent" in attrs:
                # Extract block and grid sizes from thread extents
                thread_extent = attrs["thread_extent"]
                for tag, extent in thread_extent.items():
                    if "threadIdx" in tag:
                        block_info["xyz".index(tag[-1])] = extent
                    elif "blockIdx" in tag:
                        grid_info["xyz".index(tag[-1])] = extent
            # Map the extracted configurations to each function
            block_info_map[function_name] = block_info
            grid_info_map[function_name] = grid_info
            dynamic_smem_buf_map[function_name] = dynamic_smem_buf
        # Store the mappings for use in code generation
        self.block_info = block_info_map
        self.grid_info = grid_info_map
        self.dynamic_smem_buf = dynamic_smem_buf_map

    def update_lib_code(self, code: str):
        # Organize function information for code generation
        function_informations = {}
        for g_var, func in self.mod.functions.items():
            if g_var.name_hint == "main":
                continue
            function_name = g_var.name_hint
            attrs = func.attrs
            assert "opt_shapes" in attrs
            opt_shapes = attrs["opt_shapes"]
            function_informations[function_name] = {
                "function_name": function_name,
                "opt_shapes": opt_shapes,
                "block_info": self.block_info[function_name],
                "grid_info": self.grid_info[function_name],
                "dynamic_smem_buf": self.dynamic_smem_buf[function_name],
            }

        def compare_map_objects(map_obj):
            comparable_representation = list(map_obj.values())
            return comparable_representation

        function_informations = dict(
            sorted(
                function_informations.items(),
                key=lambda item: compare_map_objects(item[1]["opt_shapes"])))

        self.lib_code = code

        # Generate the initialization and dispatch functions
        init_func = self.get_cuda_init_func()
        host_func = self.create_dispatch_func(code, function_informations)
        # Concatenate source code with generated code segments
        lib_code = self.source + init_func + host_func
        return lib_code

    @property
    def prim_func(self):
        return self.mod["main"]
