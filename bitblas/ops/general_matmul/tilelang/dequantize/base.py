# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm as tvm
from typing import Optional, Literal
from dataclasses import dataclass
from bitblas.base.base_scheduler import BaseScheduler
from bitblas.base.operator_common import TransformKind


@dataclass
class MatmulDequantizeBaseParams(BaseScheduler):
    # OP Related Config
    M: Optional[int] = None
    N: Optional[int] = None
    K: Optional[int] = None
    trans_A: bool = False
    trans_B: bool = False
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float16"

    # Dequantize Config
    num_bits: int = 4
    storage_dtype: str = "int8"
    source_format: str = "uint"
    with_scaling: bool = False
    with_zeros: bool = False
    group_size: int = -1
    fast_decoding: bool = False
    with_bias: bool = False
    zeros_mode: Literal["original", "rescale", "quantized"] = "original"

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
            "num_bits": self.num_bits,
            "storage_dtype": self.storage_dtype,
            "source_format": self.source_format,
            "with_scaling": self.with_scaling,
            "with_zeros": self.with_zeros,
            "group_size": self.group_size,
            "fast_decoding": self.fast_decoding,
            "with_bias": self.with_bias,
            "zeros_mode": self.zeros_mode,
            "input_transform_kind": self.input_transform_kind,
            "weight_transform_kind": self.weight_transform_kind,
        }

    @property
    def class_attributes(self):
        return self.params_as_dict()

    @property
    def global_symbol(self):
        # For kernel name generation
        return "matmul_dequantize"

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        fields = self.class_attributes
        field_str = ", ".join(f"{key}={value!r}" for key, value in fields.items())
        return f"{cls_name}({field_str})"

    def __post_init__(self):
        # Legalize group_size
        if self.with_scaling and self.group_size == -1:
            object.__setattr__(self, "group_size", self.K)

        # Legalization Check
        # 1. Validate the matrix transpose settings
        assert (self.trans_A is False), "Currently only support Matrix A not transposed"
        assert (self.trans_B is True), "Currently only support Matrix B transposed"
        # 2. Validate the Fast Decoding settings
        if self.fast_decoding:
            # TODO(lei): However, I think it's also possible for us to leverage fast
            # decoding for nf4 format.
            assert self.source_format in {"uint", "int"
                                         }, "Fast Decoding only support uint/int source format"
        # 3. Validate the quant config
        if self.source_format == "nf":
            assert not self.with_scaling, "NF format does not support scaling"
            assert not self.fast_decoding, "NF format does not support fast decoding"
        return
