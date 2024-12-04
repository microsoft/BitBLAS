# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .matmul_dequantize_tensorcore import (
    MatmulDequantizeBlockScheduler,  # noqa: F401
)

from .matmul_dequantize_tensorcore_finegrained import (
    MatmulDequantizeFineGrainedScheduler,  # noqa: F401
    MatmulINT4DequantizeFineGrainedScheduler,  # noqa: F401
)

from .matmul_dequantize_tensorcore_weight_transform import (
    MatmulDequantizeWeightPropagationScheduler,  # noqa: F401
    MatmulINT4DequantizeWeightPropagationScheduler,  # noqa: F401
)

from .matmul_dequantize import MatmulDequantizeScheduler

from bitblas.base.roller import TileDevice
from bitblas.base.arch import (
    is_ampere_arch,
    is_volta_arch,
)
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


def volta_select_scheduler(
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

    trans_A, trans_B = parse_layout(layout)

    def check_if_not_supported():
        conditions = [True]
        conditions.append(propagate_a == TransformKind.NonTransform)
        conditions.append(propagate_b == TransformKind.NonTransform)
        conditions.append(trans_A is False)
        conditions.append(trans_B is True)
        conditions.append(in_dtype in ["int8", "float16", "float32"])
        conditions.append(accum_dtype in ["int32", "float32"])
        return all(conditions)

    if not check_if_not_supported():
        raise ValueError(f"Unsupported configuration: {layout}, {propagate_a}, {propagate_b}")

    raise NotImplementedError


def ampere_select_scheduler(
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

    def is_int4_dtype(dtype):
        return dtype == "int4" or dtype == "uint4"

    if can_apply_weight_propagation_scheduler(trans_A, trans_B, propagate_a, propagate_b):
        Scheduler = MatmulDequantizeWeightPropagationScheduler if not is_int4_dtype(
            in_dtype) else MatmulINT4DequantizeWeightPropagationScheduler
        return Scheduler(
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
    if can_apply_fine_grain_scheduler(trans_A, trans_B, propagate_a, propagate_b):
        Scheduler = MatmulDequantizeFineGrainedScheduler if not is_int4_dtype(
            in_dtype) else MatmulINT4DequantizeFineGrainedScheduler
        return Scheduler(
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
