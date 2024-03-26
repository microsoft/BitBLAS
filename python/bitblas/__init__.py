# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys
import os

# tvm path is under the root of the project
tvm_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "3rdparty", "tvm", "python")
if tvm_path not in sys.path:
    sys.path.append(tvm_path)

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

import logging
from tqdm import tqdm


# target logger into tqdm.write
class TqdmLoggingHandler(logging.Handler):

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def set_log_level(level):
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger = logging.getLogger(__name__)
    logger.setLevel(level)


def _init_logger():
    logger = logging.getLogger(__name__)
    handler = TqdmLoggingHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s [BitBLAS:%(levelname)s]: %(message)s", datefmt="%F %T")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    set_log_level(logging.INFO)


_init_logger()
