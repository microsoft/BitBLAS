# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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

def get_gpu_model_from_nvidia_smi(gpu_id: int = 0):
    """
    Executes the 'nvidia-smi' command to fetch the name of the first available NVIDIA GPU.

    Returns:
        str: The name of the GPU, or None if 'nvidia-smi' command fails.
    """
    try:
        # Execute nvidia-smi command to get the GPU name
        output = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_id}", "--query-gpu=gpu_name", "--format=csv,noheader"],
            encoding="utf-8",
        ).strip()
    except subprocess.CalledProcessError as e:
        logger.info("nvidia-smi failed with error: %s", e)
        return None

    # Return the name of the first GPU if multiple are present
    return output.split("\n")[0]


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

    # TODO: move to a more res-usable device remapping util method
    # compat: Nvidia makes several oem (non-public) versions of A100 and perhaps other models that
    # do not have clearly defined TVM matching target so we need to manually map them to the correct one.
    if gpu_model == "NVIDIA PG506-230":
        gpu_model = "NVIDIA A100"

    target = find_best_match(nvidia_tags, gpu_model) if gpu_model else "cuda"
    return target
