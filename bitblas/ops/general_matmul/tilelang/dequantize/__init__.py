# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .block_primitive_tensorcore import (
    MatmulDequantizeScheduler,  # noqa: F401
)

from .finegrained_primitive_tensorcore import (
    MatmulDequantizeFineGrainedScheduler,  # noqa: F401
)

from .ladder_weight_transform_tensorcore import (
    MatmulDequantizeWeightPropagationScheduler,  # noqa: F401
)

from bitblas.ops.common import TransformKind
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
    '''
        Fine-grained Interface is preferred as it provides more flexibility
        and can be used to implement high performance kernel.
    '''
    if isinstance(propagate_a, int):
        propagate_a = TransformKind(propagate_a)
    if isinstance(propagate_b, int):
        propagate_b = TransformKind(propagate_b)
    if with_bias:
        raise NotImplementedError

    trans_A, trans_B = parse_layout(layout)

    def can_apply_fine_grain_scheduler(trans_A, trans_B, propagate_a, propagate_b):
        conditions = []
        conditions.append(trans_A is False)
        conditions.append(trans_B is True)
        conditions.append(propagate_a == TransformKind.NonTransform)
        conditions.append(propagate_b == TransformKind.NonTransform)
        return all(conditions)

    def can_apply_weight_propagation_scheduler(trans_A, trans_B, propagate_a, propagate_b):
        conditions = []
        conditions.append(trans_A is False)
        conditions.append(trans_B is True)
        conditions.append(propagate_a == TransformKind.NonTransform)
        conditions.append(propagate_b == TransformKind.LDMatrixTransform)
        return all(conditions)

    def can_apply_block_scheduler(propagate_a, propagate_b):
        conditions = []
        conditions.append(propagate_a == TransformKind.NonTransform)
        conditions.append(propagate_b == TransformKind.NonTransform)
        return all(conditions)

    if can_apply_block_scheduler(propagate_a, propagate_b):
        return MatmulDequantizeScheduler(
            M=M,
            N=N,
            K=K,
            trans_A=trans_A,
            trans_B=trans_B,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            accum_dtype=accum_dtype,
            num_bits=bit,
            storage_dtype=storage_dtype,
            source_format=source_format,
            with_scaling=with_scaling,
            with_zeros=with_zeros,
            group_size=group_size,
            fast_decoding=fast_decoding,
            with_bias=with_bias,
            zeros_mode=zeros_mode,
        )
    else:
        raise ValueError(f"Unsupported configuration: {layout}, {propagate_a}, {propagate_b}")
