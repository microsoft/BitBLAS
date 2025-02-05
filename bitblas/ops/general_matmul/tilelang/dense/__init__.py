# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .matmul_simt import (
    MatmulFineGrainSIMTScheduler,  # noqa: F401
)

from .matmul_tile import (
    MatmulTileLibraryScheduler,)

from .matmul_mma import (
    MatmulMMAScheduler,
    MatmulMMAWeightPropagationScheduler,
    MatmulINT4MMAScheduler,
    MatmulINT4MMAWeightPropagationScheduler,
)

from .matmul import MatmulScheduler
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


def volta_select_schduler(
    M=None,
    N=16384,
    K=16384,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
    layout="nt",
    propagate_a: Union[int, TransformKind] = TransformKind.NonTransform,
    propagate_b: Union[int, TransformKind] = TransformKind.NonTransform,
):
    trans_A, trans_B = parse_layout(layout)
    if isinstance(propagate_a, int):
        propagate_a = TransformKind(propagate_a)
    if isinstance(propagate_b, int):
        propagate_b = TransformKind(propagate_b)

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
        raise ValueError(f"Unsupported configuration: {layout=}, {propagate_a=}, {propagate_b=}")

    Scheduler = MatmulFineGrainSIMTScheduler
    return Scheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        with_bias=with_bias,
    )


def ampere_select_scheduler(
    M=None,
    N=16384,
    K=16384,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
    layout="nt",
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

    def can_apply_mma_scheduler(trans_A, trans_B, propagate_a, propagate_b):
        conditions = []
        conditions.append(trans_A is False)
        conditions.append(trans_B is True)
        conditions.append(propagate_a == TransformKind.NonTransform)
        conditions.append(propagate_b == TransformKind.NonTransform)
        return all(conditions)

    def can_apply_block_scheduler(propagate_a, propagate_b):
        conditions = []
        conditions.append(propagate_a == TransformKind.NonTransform)
        conditions.append(propagate_b == TransformKind.NonTransform)
        return all(conditions)

    def can_apply_mma_weight_propagation_scheduler(trans_A, trans_B, propagate_a, propagate_b):
        conditions = []
        conditions.append(trans_A is False)
        conditions.append(trans_B is True)
        conditions.append(propagate_a == TransformKind.NonTransform)
        conditions.append(propagate_b == TransformKind.LDMatrixTransform)
        return all(conditions)

    def is_int4_dtype(dtype):
        return dtype == "int4" or dtype == "uint4"

    if can_apply_mma_weight_propagation_scheduler(trans_A, trans_B, propagate_a, propagate_b):
        Scheduler = MatmulMMAWeightPropagationScheduler if not is_int4_dtype(
            in_dtype) else MatmulINT4MMAWeightPropagationScheduler
        return Scheduler(
            M=M,
            N=N,
            K=K,
            trans_A=trans_A,
            trans_B=trans_B,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            accum_dtype=accum_dtype,
            with_bias=with_bias,
        )
    if can_apply_mma_scheduler(trans_A, trans_B, propagate_a, propagate_b):
        Scheduler = MatmulMMAScheduler if not is_int4_dtype(in_dtype) else MatmulINT4MMAScheduler
        return Scheduler(
            M=M,
            N=N,
            K=K,
            trans_A=trans_A,
            trans_B=trans_B,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            accum_dtype=accum_dtype,
            with_bias=with_bias,
        )
    elif can_apply_block_scheduler(propagate_a, propagate_b):
        return MatmulTileLibraryScheduler(
            M=M,
            N=N,
            K=K,
            trans_A=trans_A,
            trans_B=trans_B,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            accum_dtype=accum_dtype,
            with_bias=with_bias,
        )
    else:
        raise ValueError(f"Unsupported configuration: {layout}, {propagate_a}, {propagate_b}")


def select_scheduler(
    M=None,
    N=16384,
    K=16384,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
    layout="nt",
    propagate_a: Union[int, TransformKind] = TransformKind.NonTransform,
    propagate_b: Union[int, TransformKind] = TransformKind.NonTransform,
):
    if isinstance(propagate_a, int):
        propagate_a = TransformKind(propagate_a)
    if isinstance(propagate_b, int):
        propagate_b = TransformKind(propagate_b)

    trans_A, trans_B = parse_layout(layout)

    return MatmulScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        with_bias=with_bias,
        input_transform_kind=propagate_a,
        weight_transform_kind=propagate_b,
    )
