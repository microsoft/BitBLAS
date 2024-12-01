# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm as tvm
from typing import Optional
from dataclasses import dataclass
from bitblas.base.base_scheduler import BaseScheduler
from bitblas.base.operator_common import TransformKind


@dataclass
class MatmulBaseParams(BaseScheduler):
    # OP Related Config
    M: Optional[int] = None
    N: Optional[int] = None
    K: Optional[int] = None
    trans_A: bool = False
    trans_B: bool = False
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float16"
    with_bias: bool = False

    # Ladder Transform Config
    input_transform_kind: TransformKind = TransformKind.NonTransform
    weight_transform_kind: TransformKind = TransformKind.NonTransform
