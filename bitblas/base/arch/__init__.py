# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from .arch_base import TileDevice
from .cuda import *
from .cpu import *
from .cdna import *


def get_arch(target: tvm.target.Target) -> TileDevice:
    if target.kind.name == "cuda":
        return CUDA(target)
    elif target.kind.name == "llvm":
        return CPU(target)
    elif target.kind.name == "hip":
        return CDNA(target)
    else:
        raise ValueError(f"Unsupported target: {target.kind.name}")
