# Import necessary libraries
import os
import subprocess
import logging
from rapidfuzz import process, fuzz
from tvm.target.tag import list_tags

logger = logging.getLogger(__name__)


def get_gpu_model_from_nvidia_smi():
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

    # Return the name of the first GPU if multiple are present
    return output.split("\n")[0]


def find_best_match(tags, query):
    """
    Finds the best match for a query within a list of tags using fuzzy string matching.
    """
    MATCH_THRESHOLD = 80
    best_match, score = process.extractOne(query, tags, scorer=fuzz.WRatio)
    return best_match if score >= MATCH_THRESHOLD else "cuda"


def auto_detect_nvidia_target() -> str:
    """
    Automatically detects the NVIDIA GPU architecture to set the appropriate TVM target.

    Returns:
        str: The detected TVM target architecture.
    """
    # Return a predefined target if specified in the environment variable
    if "TVM_TARGET" in os.environ:
        return os.environ["TVM_TARGET"]

    # Fetch all available tags and filter for NVIDIA tags
    all_tags = list_tags()
    nvidia_tags = [tag for tag in all_tags if "nvidia" in tag]

    # Get the current GPU model and find the best matching target
    gpu_model = get_gpu_model_from_nvidia_smi()
    if gpu_model:
        target = find_best_match(nvidia_tags, gpu_model)
    else:
        target = "cuda"  # Default to 'cuda' if the GPU model cannot be determined

    return target
