# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: disable=missing-docstring
"""Utility methods for generic GPU."""
from typing import List, Optional

from tvm import tir
from tvm.target import Target


def max_threads_per_block(target: Target) -> int:
    """Get the maximum number of threads per block for a given target.

    Parameters
    ----------
    target : Target
        The target to get the maximum number of threads per block for.

    Returns
    -------
    max_threads_per_block : int
        The maximum number of threads per block for the given target.
    """
    for name in ["max_threads_per_block", "max_num_threads"]:
        result = target.attrs.get(name, None)
        if result is not None:
            return result
    if target.kind.name == "cuda":
        return 1024
    return 256


def suggest_threads_per_block(
    target: Target,
    loops: List[tir.For],
    max_threads_for_dynamic_loop: int = 32,
) -> List[int]:
    if target.kind.name == "cuda":
        threads = 1024
    elif target.kind.name == "rocm" or target.kind.name == "metal":
        threads = 256
    else:
        threads = 64
    results: List[Optional[int]] = []
    dynamic: List[int] = []
    for i, loop in enumerate(loops):
        loop_extent = loop.extent
        if isinstance(loop_extent, tir.IntImm):
            loop_extent = loop_extent.value
            extent = 1
            while extent <= loop_extent and extent <= threads:
                extent *= 2
            extent //= 2
            assert extent >= 1
            assert threads % extent == 0
            threads //= extent
            results.append(extent)
        else:
            results.append(None)
            dynamic.append(i)

    for i in dynamic:
        extent = 1
        while extent <= max_threads_for_dynamic_loop and extent <= threads:
            extent *= 2
        extent //= 2
        assert extent >= 1
        assert threads % extent == 0
        threads //= extent
        results[i] = extent

    if dynamic:
        results[dynamic[0]] *= threads

    return results


def get_sm_version(target: Target) -> int:
    if target.kind.name != "cuda":
        return -1
    arch = target.arch
    sm_version = arch.replace("sm_", "")
    return int(sm_version) if sm_version.isdigit() else -1
