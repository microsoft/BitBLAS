from typing import Union, Optional
from bitblas import tilelang as tilelang
from tilelang import tvm as tvm
from tvm import tir
from tvm.target import Target

def tl_lower(
    func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
    target: Union[str, Target] = "auto",
    target_host: Optional[Union[str, Target]] = None,
    runtime_only=False,
):

    result = tilelang.lower(
        func_or_mod, 
        target=target, 
        target_host=target_host,
        runtime_only=runtime_only,
        enable_device_compile=True,
        enable_host_codegen=True,
    )

    if runtime_only is True:
        return result.rt_mod
    else:
        return result.rt_mod, result.params