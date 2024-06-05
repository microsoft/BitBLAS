# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys
import os

# installing tvm
install_tvm_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "3rdparty", "tvm", "python")
if os.path.exists(install_tvm_path) and install_tvm_path not in sys.path:
    os.environ["PYTHONPATH"] = install_tvm_path + ":" + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, install_tvm_path)

develop_tvm_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "3rdparty", "tvm", "python")
if os.path.exists(develop_tvm_path) and develop_tvm_path not in sys.path:
    os.environ["PYTHONPATH"] = develop_tvm_path + ":" + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, develop_tvm_path)

from . import gpu  # noqa: F401
from .base import (
    TileDevice,  # noqa: F401
    fast_tune,  # noqa: F401
    ApplyDefaultSchedule,  # noqa: F401
    ApplyFastTuning,  # noqa: F401
    BlockInfo,  # noqa: F401
    IterInfo,  # noqa: F401
    ScheduleRule,  # noqa: F401
    normalize_prim_func,  # noqa: F401
    try_inline,  # noqa: F401
    try_inline_contiguous_spatial,  # noqa: F401
)

from . import testing  # noqa: F401
from .utils import auto_detect_nvidia_target  # noqa: F401
from .ops.general_matmul import MatmulConfig, Matmul  # noqa: F401
from .ops.matmul_dequantize import MatmulWeightOnlyDequantizeConfig, MatmulWeightOnlyDequantize  # noqa: F401
from .module import Linear  # noqa: F401

import logging
from tqdm import tqdm

class TqdmLoggingHandler(logging.Handler):
    """ Custom logging handler that directs log output to tqdm progress bar to avoid interference. """

    def __init__(self, level=logging.NOTSET):
        """ Initialize the handler with an optional log level. """
        super().__init__(level)

    def emit(self, record):
        """ Emit a log record. Messages are written to tqdm to ensure output in progress bars isn't corrupted. """
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def set_log_level(level):
    """ Set the logging level for the module's logger.
    
    Args:
        level (str or int): Can be the string name of the level (e.g., 'INFO') or the actual level (e.g., logging.INFO).
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)


def _init_logger():
    """ Initialize the logger specific for this module with custom settings and a Tqdm-based handler. """
    logger = logging.getLogger(__name__)
    handler = TqdmLoggingHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s [BitBLAS:%(levelname)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    set_log_level('WARNING')


_init_logger()

__version__ = "0.0.1.dev8"
