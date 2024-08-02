# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from .post_process import match_global_kernel, tensor_replace_dp4a, tensor_remove_make_int4, tensor_remove_make_int2  # noqa: F401
from .tensor_adapter import tvm_tensor_to_torch, lazy_tvm_tensor_to_torch, lazy_torch_to_tvm_tensor  # noqa: F401
from .target_detector import get_all_nvidia_targets, auto_detect_nvidia_target  # noqa: F401
from .rtmod_analysis import get_annotated_device_mod  # noqa: F401
from .weight_propagate import apply_transform_on_input  # noqa: F401

import os
import subprocess

BITBLAS_DEFAULT_CACHE_PATH = os.path.expanduser("~/.cache/bitblas")


def get_commit_id():
    try:
        commit_id = (subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8"))
        return commit_id
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.output}")
        return None


def get_default_cache_path():
    return BITBLAS_DEFAULT_CACHE_PATH
