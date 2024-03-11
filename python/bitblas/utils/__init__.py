# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from .post_process import match_global_kernel, tensor_replace_dp4a
from .tensor_adapter import tvm_tensor_to_torch
import os

def get_target_from_env() -> str:
    return os.environ.get("TVM_TARGET") or "cuda"
