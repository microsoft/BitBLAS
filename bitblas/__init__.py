# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys
import os

# installing tvm
install_tvm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3rdparty", "tvm")
install_cutlass_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "3rdparty", "cutlass")
if os.path.exists(install_tvm_path) and install_tvm_path not in sys.path:
    os.environ["PYTHONPATH"] = install_tvm_path + "/python:" + os.environ.get("PYTHONPATH", "")
    os.environ["TL_CUTLASS_PATH"] = install_cutlass_path + "/include"
    sys.path.insert(0, install_tvm_path + "/python")

develop_tvm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "3rdparty", "tvm")
develop_cutlass_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "3rdparty", "cutlass")
if os.path.exists(develop_tvm_path) and develop_tvm_path not in sys.path:
    os.environ["PYTHONPATH"] = develop_tvm_path + "/python:" + os.environ.get("PYTHONPATH", "")
    os.environ["TL_CUTLASS_PATH"] = develop_cutlass_path + "/include"
    sys.path.insert(0, develop_tvm_path + "/python")

import tvm as tvm  # noqa: E402
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
from .utils import auto_detect_nvidia_target, apply_transform_on_input  # noqa: F401
from .ops.general_matmul import MatmulConfig, Matmul  # noqa: F401
from .ops.general_matmul_splitk import MatmulConfigWithSplitK, MatmulWithSplitK  # noqa: F401
from .module import Linear  # noqa: F401

import warnings
import functools
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
        OPTIONS: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
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


def deprecated(reason):
    """
    This is a decorator which can be used to mark functions as deprecated.
    It will result in a warning being emitted when the function is used.
    """

    def decorator(func):

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            warnings.warn(
                f"Call to deprecated function {func.__name__} ({reason}).",
                category=DeprecationWarning,
                stacklevel=2)
            return func(*args, **kwargs)

        return new_func

    return decorator


__version__ = "0.0.1.dev15"
