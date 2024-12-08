# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from .arch_base import TileDevice
from .cuda import *
from .cpu import *
from .cdna import *
from typing import Union


def get_arch(target: Union[str, tvm.target.Target] = "cuda") -> TileDevice:
    if isinstance(target, str):
        target = tvm.target.Target(target)

    if target.kind.name == "cuda":
        return CUDA(target)
    elif target.kind.name == "llvm":
        return CPU(target)
    elif target.kind.name == "hip":
        return CDNA(target)
    else:
        raise ValueError(f"Unsupported target: {target.kind.name}")


def auto_infer_current_arch() -> TileDevice:
    # TODO(lei): This is a temporary solution to infer the current architecture
    # Can be replaced by a more sophisticated method in the future
    return get_arch("cuda")


def is_cpu_arch(arch: TileDevice) -> bool:
    return isinstance(arch, CPU)


def is_cuda_arch(arch: TileDevice) -> bool:
    return isinstance(arch, CUDA)


def is_ampere_arch(arch: TileDevice) -> bool:
    conditions = [True]
    conditions.append(is_cuda_arch(arch))
    conditions.append(arch.sm_version >= 80 and arch.sm_version < 90)
    return all(conditions)


def is_volta_arch(arch: TileDevice) -> bool:
    conditions = [True]
    conditions.append(is_cuda_arch(arch))
    conditions.append(arch.sm_version >= 70)
    conditions.append(arch.sm_version < 80)
    return all(conditions)


def is_cdna_arch(arch: TileDevice) -> bool:
    return isinstance(arch, CDNA)


def has_mma_support(arch: TileDevice) -> bool:
    conditions = [True]
    conditions.append(is_cuda_arch(arch))
    conditions.append(arch.sm_version >= 80)
    return all(conditions)


def is_tensorcore_supported_precision(in_dtype: str, accum_dtype: str, arch: TileDevice) -> bool:
    volta_tensorcore_supported = [
        ("float16", "float32"),
        ("float16", "float16"),
    ]
    ampere_tensorcore_supported = [
        ("float16", "float32"),
        ("float16", "float16"),
        ("int8", "int32"),
        ("int4", "int32"),
        ("int2", "int32"),
        ("int1", "int32"),
    ]

    if is_volta_arch(arch):
        return (in_dtype, accum_dtype) in volta_tensorcore_supported
    elif is_ampere_arch(arch):
        return (in_dtype, accum_dtype) in ampere_tensorcore_supported
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
