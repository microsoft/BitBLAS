# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .matmul_dequantize_tile import (
    MatmulDequantizeTileLibraryScheduler,  # noqa: F401
)

from .matmul_dequantize_mma import (
    MatmulDequantizeMMAScheduler,  # noqa: F401
    MatmulINT4DequantizeMMAScheduler,  # noqa: F401
)

from .matmul_dequantize_mma_weight_transform import (
    MatmulDequantizeMMAWeightPropagationScheduler,  # noqa: F401
    MatmulINT4DequantizeMMAWeightPropagationScheduler,  # noqa: F401
)

from .matmul_dequantize import MatmulDequantizeScheduler

from bitblas.base.operator_common import TransformKind
from typing import Union


def parse_layout(layout: str):
    if len(layout) != 2 or layout[0] not in "nt" or layout[1] not in "nt":
        raise ValueError(f"Invalid layout: {layout}")

    trans_A = layout[0] == 't'
    trans_B = layout[1] == 't'

    return trans_A, trans_B


def is_non_transform_kind(kind) -> bool:
    return kind == TransformKind.NonTransform


def select_scheduler(
    M=None,
    N=1024,
    K=1024,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    bit=4,
    storage_dtype="int8",
    source_format="uint",
    with_scaling=False,
    with_zeros=False,
    group_size=-1,
    fast_decoding=False,
    with_bias=False,
    layout="nt",
    zeros_mode="original",
    propagate_a: Union[int, TransformKind] = TransformKind.NonTransform,
    propagate_b: Union[int, TransformKind] = TransformKind.NonTransform,
):
    if isinstance(propagate_a, int):
        propagate_a = TransformKind(propagate_a)
    if isinstance(propagate_b, int):
        propagate_b = TransformKind(propagate_b)

    trans_A, trans_B = parse_layout(layout)
    return MatmulDequantizeScheduler(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        trans_A=trans_A,
        trans_B=trans_B,
        num_bits=bit,
        storage_dtype=storage_dtype,
        source_format=source_format,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        group_size=group_size,
        fast_decoding=fast_decoding,
        with_bias=with_bias,
        zeros_mode=zeros_mode,
        input_transform_kind=propagate_a,
        weight_transform_kind=propagate_b,
    )
