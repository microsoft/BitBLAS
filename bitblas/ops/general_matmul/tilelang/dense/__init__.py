# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .matmul_simt import (
    MatmulFineGrainSIMTScheduler,  # noqa: F401
)

from .matmul_tensorcore import (
    matmul_blocked,  # noqa: F401
    matmul_macro_tensorcore,  # noqa: F401
    matmul_macro_tensorcore_weight_propagation_level_ldmatrix  # noqa: F401
)

from .matmul_tensorcore import (
    MatmulScheduler,  # noqa: F401
    MatmulFineGrainScheduler,  # noqa: F401
    MatmulWeightPropagationScheduler,  # noqa: F401
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
    if with_bias:
        raise NotImplementedError

    trans_A, trans_B = parse_layout(layout)
    if is_non_transform_kind(propagate_a) and is_non_transform_kind(propagate_b):
        return MatmulScheduler(
            M=M,
            N=N,
            K=K,
            trans_A=trans_A,
            trans_B=trans_B,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            accum_dtype=accum_dtype,
        )
    else:
        raise ValueError(f"Unsupported transform kind: {propagate_a}, {propagate_b}")
