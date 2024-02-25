# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from .tensor_adapter import tvm_tensor_to_torch
import re

def match_global_kernel(source: str) -> int:
    pattern = r"__global__\s+void\s+[__launch_bounds__\(\d+\)\s+]\w+"
    matched = re.findall(pattern, source)
    assert len(matched) > 1  # may have statement before kernel
    return source.index(matched[0])
