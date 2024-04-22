# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tvm._ffi


def WelderFuseOps():
    """Preliminary fusion for welder compiler.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for operator fusion.
    """
    return tvm._ffi.get_global_func("relay._transform.WelderFuseOps")()
