# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import subprocess
from typing import List
from thefuzz import process
from tvm.target import Target
from tvm.target.tag import list_tags

import logging

logger = logging.getLogger(__name__)

TARGET_MISSING_ERROR = (
    "TVM target not found. Please set the TVM target environment variable using `export TVM_TARGET=<target>`, "
    "where <target> is one of the available targets can be found in the output of `tools/get_available_targets.py`."
)

# Nvidia produces non-public oem gpu models that are part of drivers but not mapped to correct tvm target
# Remap list to match the oem model name to the closest public model name
NVIDIA_GPU_REMAP = {
    "NVIDIA PG506-230": "NVIDIA A100",
    "NVIDIA PG506-232": "NVIDIA A100",
}


def get_gpu_model_from_nvidia_smi(gpu_id: int = 0):
    """
    Executes the 'nvidia-smi' command to fetch the name of the first available NVIDIA GPU.

    Returns:
        str: The name of the GPU, or None if 'nvidia-smi' command fails.
    """
    try:
        # Execute nvidia-smi command to get the GPU name
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
            encoding="utf-8",
        ).strip()
    except subprocess.CalledProcessError as e:
        logger.info("nvidia-smi failed with error: %s", e)
        return None

    gpus = output.split("\n")

    # for multiple gpus, CUDA_DEVICE_ORDER=PCI_BUS_ID must be set to match nvidia-smi or else wrong
    # gpu is returned for gpu_id
    if len(gpus) > 1 and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise EnvironmentError("Multi-gpu environment must set `CUDA_DEVICE_ORDER=PCI_BUS_ID`.")

    if gpu_id >= len(gpus) or gpu_id < 0:
        raise ValueError(f"Passed gpu_id:{gpu_id} but there are {len(gpus)} detected Nvidia gpus.")

    return gpus[gpu_id]


def find_best_match(tags, query):
    """
    Finds the best match for a query within a list of tags using fuzzy string matching.
    """
    MATCH_THRESHOLD = 25
    best_match, score = process.extractOne(query, tags)

    def check_target(best, default):
        return best if Target(best).arch == Target(default).arch else default

    if check_target(best_match, "cuda") == best_match:
        return best_match if score >= MATCH_THRESHOLD else "cuda"
    else:
        logger.warning(TARGET_MISSING_ERROR)
        return "cuda"


def get_all_nvidia_targets() -> List[str]:
    """
    Returns all available NVIDIA targets.
    """
    all_tags = list_tags()
    return [tag for tag in all_tags if "nvidia" in tag]


def auto_detect_nvidia_target(gpu_id: int = 0) -> str:
    """
    Automatically detects the NVIDIA GPU architecture to set the appropriate TVM target.

    Returns:
        str: The detected TVM target architecture.
    """
    # Return a predefined target if specified in the environment variable
    # if "TVM_TARGET" in os.environ:
    #     return os.environ["TVM_TARGET"]

    # Fetch all available tags and filter for NVIDIA tags
    all_tags = list_tags()
    nvidia_tags = [tag for tag in all_tags if "nvidia" in tag]

    # Get the current GPU model and find the best matching target
    gpu_model = get_gpu_model_from_nvidia_smi(gpu_id=gpu_id)

    # Compat: remap oem devices to their correct non-oem model names for tvm target
    if gpu_model in NVIDIA_GPU_REMAP:
        gpu_model = NVIDIA_GPU_REMAP[gpu_model]

    target = find_best_match(nvidia_tags, gpu_model) if gpu_model else "cuda"
    return target


def auto_detect_target(target: str = "auto", gpu_id: int = 0) -> str:
    """Detect the computing target (CUDA or ROCm) based on the environment.

    Args:
        target (str): The target to detect. Use "auto" for automatic detection.
                      Can also specify "cuda" or "hip" explicitly.

    Returns:
        str: The detected target, either "cuda" or "hip".

    Raises:
        ValueError: If auto-detection is enabled and no valid target is found.
    """

    from tvm.contrib import nvcc, rocm

    def is_cuda_available() -> bool:
        """Check if CUDA is available."""
        try:
            nvcc.find_cuda_path()
            return True
        except RuntimeError:
            return False

    def is_rocm_available() -> bool:
        """Check if ROCm is available."""
        try:
            rocm.find_rocm_path()
            return True
        except RuntimeError:
            return False

    if target == "auto":
        if is_cuda_available():
            return auto_detect_nvidia_target(gpu_id=gpu_id)
        if is_rocm_available():
            return "hip"
        raise ValueError("Cannot detect the target: no CUDA or ROCm installation found.")

    if target not in {"cuda", "hip"}:
        raise ValueError(f"Invalid target: {target}. Must be 'cuda', 'hip', or 'auto'.")

    return target
