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
