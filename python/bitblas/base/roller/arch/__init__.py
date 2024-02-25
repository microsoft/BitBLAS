# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from .arch_base import Arch
from .cuda import *
from .cpu import *


def get_arch(target: tvm.target.Target) -> Arch:
    if target.kind.name == "cuda":
        return CUDA(target)
    elif target.kind.name == "llvm":
        return CPU(target)
    else:
        raise ValueError(f"Unsupported target: {target.kind.name}")
