# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm as tvm
from tvm.target import Target
from typing import Literal, Union
from ..operator import Operator
from .quant_compress_impl import select_implementation
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class QuantCompressConfig:
    M: int
    N: int
    input_dtype: Literal["int8", "int32"] = "int8"
    storage_dtype: Literal["int8", "int32"] = "int8"
    dequantize_bits: int = 4


class QuantCompress(Operator):

    def __init__(
            self,
            config: QuantCompressConfig,
            name: str = "quant_compress",
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
            input_dtype=self.input_dtype,
            storage_dtype=self.storage_dtype,
            dequantize_bits=self.dequantize_bits,
        )

    def forward(self, inp, out=None):
        out_shape = out.shape if out is not None else None
        if out is None:
            out_shape = self.retrieve_qweight_shape()
            out = torch.empty(out_shape, dtype=getattr(torch, self.storage_dtype))
        args = [inp.view((self.M, self.N)), out]
        self.torch_func(*args)
        return out.view(out_shape)

    def retrieve_qweight_shape(self):
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
    def input_dtype(self):
        return self.config.input_dtype

    @property
    def storage_dtype(self):
        return self.config.storage_dtype

    @property
    def dequantize_bits(self):
        return self.config.dequantize_bits


__all__ = ["QuantCompress", "QuantCompressConfig"]
