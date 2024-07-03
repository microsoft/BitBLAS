# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from tvm.target import Target
from typing import Literal, Union
from .operator import Operator
from .impl.ladder_permutate_impl import select_implementation
from dataclasses import dataclass


@dataclass(frozen=True)
class LadderPermutateConfig:
    M: int
    N: int
    datatype: Literal["int8", "e4m3_float8", "e5m2_float8"] = "float16"
    dequantize_bits: int = -1
    storage_dtype: Literal["float16", "int8", "uint8", "int32", "uint32"] = "float16"
    propagate_kind: Literal["A", "B"] = "B"  # "A" or "B"
    transpose_matrix: bool = False
    transform_kind: int = 2  # 0: none, 1: inter_warp 2: intra_warp
    target_instruction: Literal["nvidia-mma"] = (
        "nvidia-mma"  # maybe extend to "cdna-mfma" in future.
    )


class LadderPermutate(Operator):

    def __init__(
        self,
        config: LadderPermutateConfig,
        name: str = "permutate",
        target: Union[str, Target] = "llvm",  # assume to do permutation on cpu.
        enable_tuning: bool = False,
        from_database: bool = False,
    ):
        # consider to warp the arguments to MatmulConfig
        super().__init__(name, config, target)

        target = self.target
        if target.kind.name == "cuda":
            self.optimized_func = self.apply_default_schedule(self.prim_func_mod, target)
            if enable_tuning:
                self.hardware_aware_finetune()
        if not from_database:
            self._build_runtime_module(target)

    # select implementation based on the Operator config
    def _select_implementation(self):
        return select_implementation(
            M=self.M,
            N=self.N,
            datatype=self.datatype,
            dequantize_bits=self.dequantize_bits,
            storage_dtype=self.storage_dtype,
            propagate_kind=self.propagate_kind,
            transpose_matrix=self.transpose_matrix,
            transform_kind=self.transform_kind,
            target_instruction=self.target_instruction,
        )

    @property
    def M(self):
        return self.config.M

    @property
    def N(self):
        return self.config.N

    @property
    def datatype(self):
        return self.config.datatype

    @property
    def dequantize_bits(self):
        return self.config.dequantize_bits

    @property
    def storage_dtype(self):
        return self.config.storage_dtype

    @property
    def propagate_kind(self):
        return self.config.propagate_kind

    @property
    def transpose_matrix(self):
        return self.config.transpose_matrix

    @property
    def transform_kind(self):
        return self.config.transform_kind

    @property
    def target_instruction(self):
        return self.config.target_instruction


__all__ = ["LadderPermutate", "LadderPermutateConfig"]
