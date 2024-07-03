# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Base infra"""
from .analysis import (
    BlockInfo,
    IterInfo,
    collect_block_iter_vars_used_in_access_region,
    collect_vars_used_in_prim_expr,
    detect_dominant_read,
    is_broadcast_epilogue,
    normalize_prim_func,
)
from .common_schedules import get_block, get_output_blocks, try_inline, try_inline_contiguous_spatial
from .schedule_rule import ScheduleRule
from .transform import ApplyDefaultSchedule, ApplyFastTuning
from .utils import fast_tune, fast_tune_with_dynamic_range
from .roller import *
