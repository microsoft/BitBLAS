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


def is_ampere_arch(arch: TileDevice) -> bool:
    conditions = [True]
    conditions.append(isinstance(arch, CUDA))
    conditions.append(arch.sm_version >= 80)
    return all(conditions)


def is_volta_arch(arch: TileDevice) -> bool:
    conditions = [True]
    conditions.append(isinstance(arch, CUDA))
    conditions.append(arch.sm_version >= 70)
    conditions.append(arch.sm_version < 80)
    return all(conditions)
