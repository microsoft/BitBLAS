# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .flashatten import flashatten_blocked  # noqa: F401
from .flashatten import FlashAttenScheduler  # noqa: F401


def parse_layout(layout: str):
    trans_Q = False
    trans_K = layout[1] == 't'
    trans_V = False
    return trans_Q, trans_K, trans_V


def select_scheduler(
    batch=None,
    heads=None,
    seq_len=None,
    dim=None,
    layout="nnn",
    dtype_QKV="float16",
    dtype_Out="float16",
    dtype_Accu="float32",
    is_causal=False,
):
    trans_list = parse_layout(layout)
    trans_K = trans_list[1]
    return FlashAttenScheduler(
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        dim=dim,
        trans_K=trans_K,
        dtype_QKV=dtype_QKV,
        dtype_Out=dtype_Out,
        dtype_Accu=dtype_Accu,
        is_causal=is_causal,
    )
