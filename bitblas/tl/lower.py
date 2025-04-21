from bitblas import tilelang as tilelang
from bitblas import tvm
from tvm.target import Target
from tvm import tir
from typing import Optional, Union

def tl_lower(
    func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
    target: Union[str, Target] = "auto",
    target_host: Optional[Union[str, Target]] = None,
    runtime_only=False,
):
    result = tilelang.lower(
        func_or_mod,
        target,
        target_host=target_host,
        runtime_only=runtime_only,
    )
    if runtime_only is True:
        return result.host_mod
    else:
        return result.host_mod, result.parms

