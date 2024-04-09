# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from logging import getLogger

import numpy as np
import torch
import torch.nn as nn

from typing import List, Union, Literal, Optional

logger = getLogger(__name__)

try:
    import bitblas  # noqa: F401
except ImportError as e:
    bitblas_import_exception = e

    def error_raiser_bitblas(*args, **kwargs):
        raise ValueError(
            f"Trying to use the bitblas backend, but could not import dependencies with the following error: {bitblas_import_exception}"
        )

    autogptq_bitblas_cuda = bitblas_import_exception

from bitblas.utils import auto_detect_nvidia_target  # noqa: E402
from bitblas.ops.matmul import MatmulConfig, Matmul  # noqa: E402


class Linear(nn.Module):

    def __init__(
        self,
        infeatures: int,
        outfeatures: int,
        opt_m: Union[int, List[int]] = 1,
        bias: bool = False,
        dtype: torch.dtype = torch.float16,
        propagate_a: bool = False,
        propagate_b: bool = False,
        enable_tuning: bool = False,
        trainable: bool = False,
        layout: Literal["nn", "nt"] = "nt",
        target: Optional[str] = None,
    ):
        """
        @opt_m: optimize range of the input shape for dynamic symbolic
        if the input shape is a range, we will optimize the matmul with dynamic symbolic.
        if the input shape is int, we will optimize the matmul with static symbolic.
        """
        super().__init__()
        if trainable:
            raise NotImplementedError("Bitblas does not support train.")

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.opt_m = opt_m
        self.dtype = dtype
        self.propagate_a = propagate_a
        self.propagate_b = propagate_b
        self.enable_tuning = enable_tuning
        self.weight = nn.Parameter(torch.empty((outfeatures, infeatures), dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(outfeatures, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        BITBLAS_DTYPES = {
            torch.float32: "float32",
            torch.float16: "float16",
            torch.int8: "int8",
        }
        assert dtype in BITBLAS_DTYPES, f"Unsupported dtype: {dtype}"

        bitblas_dtype = BITBLAS_DTYPES[dtype]
        self.target = target or auto_detect_nvidia_target()
        matmul_config = MatmulConfig(
            M=self.opt_m,
            N=self.outfeatures,
            K=self.infeatures,
            in_dtype=bitblas_dtype,
            out_dtype=bitblas_dtype,
            accum_dtype="int32" if bitblas_dtype == "int8" else bitblas_dtype,
            with_bias=bias,
            propagate_a=propagate_a,
            propagate_b=propagate_b,
            layout=layout,
        )

        self.bitblas_matmul = Matmul(
            config=matmul_config,
            target=self.target,
        )

        if enable_tuning:
            self.bitblas_matmul.hardware_aware_finetune(topk=20)

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            stdv = 1.0 / np.sqrt(self.weight.shape[1])
            self.weight.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.uniform_(-stdv, stdv)

    def forward(self, A, Output=None):
        args = [
            A,
            self.weight,
        ]
        if self.bias is not None:
            args.append(self.bias)
        if Output is None:
            Output = torch.empty(A.shape[:-1] + (self.outfeatures,), dtype=A.dtype, device=A.device)
        args.append(Output)

        self.bitblas_matmul(*args)

        return Output


__all__ = ["Linear"]
