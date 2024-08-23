# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from tvm.target import Target
from typing import Literal, Union
from ..operator import Operator
from .lop3_permutate_impl import select_implementation
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class LOP3PermutateConfig:
    M: int
    N: int
    datatype: Literal["float16", "int8"] = "float16"
    storage_dtype: Literal["int8", "uint8", "int32", "uint32"] = "int8"
    dequantize_bits: int = 4


class LOP3Permutate(Operator):

    def __init__(
            self,
            config: LOP3PermutateConfig,
            name: str = "permutate",
            target: Union[str, Target] = "llvm",  # assume to do permutation on cpu.
    ):
        # consider to warp the arguments to MatmulConfig
        super().__init__(name, config, target)

        target = self.target
        if target.kind.name != "llvm":
            raise ValueError("Currently only support llvm target for Permutation")

        self._build_runtime_module(target)

    def _select_implementation(self):
        return select_implementation(
            M=self.M,
            N=self.N,
            datatype=self.datatype,
            dequantize_bits=self.dequantize_bits,
        )

    def forward(self, inp, out=None):
        out_shape = inp.shape
        out_dtype = inp.dtype
        if out is None:
            # lop3 transform does not change the shape of the input tensor
            out = torch.zeros(out_shape, dtype=out_dtype)
        # reinterpret the input tensor to int32 format
        shape_2dim = self.retrieve_2d_weight_shape()
        args = [arg.view(inp.dtype).view(shape_2dim).view(torch.int32) for arg in [inp, out]]
        self.torch_func(*args)
        return args[-1].view(out_dtype).view(out_shape)

    def retrieve_2d_weight_shape(self):
        storage_nbit = int("".join(c for c in self.storage_dtype if c.isdigit()))
        elems_per_byte = storage_nbit // self.dequantize_bits
        weight_shape = (self.M, self.N // elems_per_byte)
        return weight_shape

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
    def storage_dtype(self):
        return self.config.storage_dtype

    @property
    def dequantize_bits(self):
        return self.config.dequantize_bits


__all__ = ["LOP3Permutate", "LOP3PermutateConfig"]
