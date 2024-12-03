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

    def params_as_dict(self):
        return {
            "M": self.M,
            "N": self.N,
            "K": self.K,
            "trans_A": self.trans_A,
            "trans_B": self.trans_B,
            "in_dtype": self.in_dtype,
            "out_dtype": self.out_dtype,
            "accum_dtype": self.accum_dtype,
            "with_bias": self.with_bias,
            "input_transform_kind": self.input_transform_kind,
            "weight_transform_kind": self.weight_transform_kind,
        }

    @property
    def class_attributes(self):
        return self.params_as_dict()

    @property
    def global_symbol(self):
        # For kernel name generation
        return "matmul"

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        fields = self.class_attributes
        field_str = ", ".join(f"{key}={value!r}" for key, value in fields.items())
        return f"{cls_name}({field_str})"
