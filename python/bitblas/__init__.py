# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DLight package provides efficient schedules out-of-box for deep learning workloads."""
from . import gpu
from .base import (
    fast_tune,
    ApplyDefaultSchedule,
    ApplyFastTuning,
    BlockInfo,
    IterInfo,
    ScheduleRule,
    normalize_prim_func,
    try_inline,
    try_inline_contiguous_spatial,
)

from .relax import transform
from . import testing

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
    logger = logging.getLogger(__name__)
    logger.setLevel(level)


def _init_logger():
    logger = logging.getLogger(__name__)
    handler = TqdmLoggingHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s [BitBLAS:%(levelname)s]: %(message)s", datefmt="%F %T"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    set_log_level(logging.INFO)


_init_logger()
