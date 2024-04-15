# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from tvm.target import Target
from typing import Literal, Union
from .operator import Operator, TransformKind
from .impl.param_permutate_impl import select_implementation
from dataclasses import dataclass


@dataclass(frozen=True)
class ParamPermutateConfig:
    M: int
    N: int
    datatype: Literal["float16"] = "float16"
    transpose_matrix: bool = True
    group_size: int = -1
    propagate_kind: TransformKind = TransformKind.NonTransform
    target_instruction: Literal["nvidia-mma"] = (
        "nvidia-mma"  # maybe extend to "cdna-mfma" in future.
    )

    def __post_init__(self):
        if isinstance(self.propagate_kind, bool):
            object.__setattr__(
                self,
                "propagate_kind",
                (TransformKind.IntraWarpTransform
                 if self.propagate_kind else TransformKind.NonTransform),
            )
        elif isinstance(self.propagate_kind, int):
            object.__setattr__(self, "propagate_kind", TransformKind(self.propagate_kind))


class ParamPermutate(Operator):

    def __init__(
            self,
            config: ParamPermutateConfig,
            name: str = "permutate",
            target: Union[str, Target] = "llvm",  # assume to do permutation on cpu.
    ):
        super().__init__(name, config, target)

        if target.kind.name != "llvm":
            raise ValueError("Currently only support llvm target for Permutation")

        self.target = target
        self._build_runtime_module(target)

    # select implementation based on the Operator config
    def _select_implementation(self):
        return select_implementation(
            M=self.M,
            N=self.N,
            datatype=self.datatype,
            transpose_matrix=self.transpose_matrix,
            group_size=self.group_size,
            propagate_kind=self.propagate_kind,
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
    def propagate_kind(self):
        return self.config.propagate_kind

    @property
    def transpose_matrix(self):
        return self.config.transpose_matrix

    @property
    def group_size(self):
        return self.config.group_size

    @property
    def target_instruction(self):
        return self.config.target_instruction


__all__ = ["ParamPermutate", "ParamPermutateConfig"]
