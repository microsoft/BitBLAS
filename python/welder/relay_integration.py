import tvm
from tvm import relay
from tvm import _ffi
from .utils import CompileResult
from typing import Dict
import tempfile
import os
import subprocess

_global_dict: Dict[str, CompileResult] = {}

_source_mod_cache = {}

def add_source(key, cpresult: CompileResult) -> None:
    _global_dict[key] = cpresult

@tvm._ffi.register_func("relay.ext.welder")
def _compiler(func):
    v = _ffi.get_global_func("runtime.CSourceModuleCreate")
    tvm_symbol = func.attrs["global_symbol"]
    target = func.body.op.attrs["Composite"]
    cpresult = _global_dict[target]
    if cpresult.origin is not None:
        cpresult = cpresult.origin
    symbol = cpresult.name + "_host"
    link_code = cpresult.create_tvm_link_code(tvm_symbol, symbol)
    link_mod = v(link_code, "cc", [tvm_symbol], [])
    if cpresult not in _source_mod_cache:
        source_code = cpresult.create_code_for_tvm(symbol)
        source_mod = v(source_code, "cu", [symbol], [])
        _source_mod_cache[cpresult] = source_mod
    source_mod = _source_mod_cache[cpresult]
    link_mod.import_module(source_mod)
    return link_mod

def call_cuda_compile(output, objects, options=None, cc="nvcc"):
    procs = []
    objects_to_link = []
    temp_objs = []
    for object in objects:
        if object.endswith('.o'):
            objects_to_link.append(object)
        else:
            obj = tempfile.mktemp(suffix=".o")
            temp_objs.append(obj)
            objects_to_link.append(obj)
            commands = [cc, "-c", object, "-o", obj, "--compiler-options", "-fPIC"] + options
            proc = subprocess.Popen(commands, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            procs.append(proc)
    for proc in procs:
        proc.wait()
    for proc in procs:
        if proc.returncode != 0:
            msg = proc.stdout.read().decode('utf-8')
            raise RuntimeError("Compilation error: " + msg)
    subprocess.run(["nvcc", "--shared", *objects_to_link, "-o", output], check=True)
    for obj in temp_objs:
        os.remove(obj)

def update_lib(lib, arch, lib_path):
    compute_version = arch.compute_capability
    cutlass_dir = os.path.expanduser("~/cutlass/include")
    options = ["-std=c++17",
               f"-gencode=arch=compute_{compute_version},code=compute_{compute_version}",
               f"-I{cutlass_dir}"]
    lib.export_library(lib_path, fcompile=call_cuda_compile, options=options)
    lib = tvm.runtime.load_module(lib_path)
    return lib
