# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ctypes
import operator
from functools import reduce
from logging import getLogger

import torch
import torch.nn as nn


logger = getLogger(__name__)

try:
    import bitblas
except ImportError as e:
    bitblas_import_exception = e

    def error_raiser_bitblas(*args, **kwargs):
        raise ValueError(
            f"Trying to use the bitblas backend, but could not import dependencies with the following error: {bitblas_import_exception}"
        )

    autogptq_bitblas_cuda = bitblas_import_exception

from typing import List, Union

from bitblas.cache import global_operator_cache
from bitblas import Matmul, MatmulConfig
from bitblas.quantization.utils import general_compress
from bitblas import auto_detect_nvidia_target

BITBLAS_TARGET = auto_detect_nvidia_target()
BITBLAS_DATABASE_PATH = ".bitblas_database"
BITBLAS_PROPAGATE_WEIGHTS = False
global_operator_cache.load_from_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)


class Linear(nn.Module):
    OPT_FEATURES = [1, 16, 32, 64, 128, 256, 512]
    STORAGE_DTYPE = "int8"  # assume int8 storage
    TORCH_STORAGE_DTYPE = getattr(torch, STORAGE_DTYPE)
    BITBLAS_DTYPES = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.half: "float16",
        torch.int8: "int8",
    }

    def __init__(
        self,
        infeatures: int,
        outfeatures: int,
        bias: bool = False,
        in_dtype: str = "float16",
        weight_dtype: str = "float16",
        accum_dtype: str = "float16",
        out_dtype: str = "float16",
        # configs for weight only quantization
        group_size: int = -1,
        with_scaling: bool = None,
        with_zeros: bool = False,
        zeros_mode: str = None,
        opt_features: Union[int, List[int]] = OPT_FEATURES,
        # performance related configs
        enable_tuning: bool = True,
        fast_decoding: bool = True,
        propagate_b: bool = False,
    ):
        """
        @opt_features: optimize range of the input shape for dynamic symbolic
        if the input shape is a range, we will optimize the matmul with dynamic symbolic.
        if the input shape is int, we will optimize the matmul with static symbolic.
        """
        super().__init__()

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.opt_features = opt_features
        self.group_size = self._set_group_size(group_size, infeatures)
        self.torch_dtype = getattr(torch, in_dtype)
        self.is_consitent = in_dtype == weight_dtype
        self.zeros_mode = zeros_mode
        self._validate_parameters(self.group_size, infeatures, outfeatures)
        self._configure_bitblas_matmul(
            in_dtype,
            weight_dtype,
            accum_dtype,
            out_dtype,
            with_scaling,
            with_zeros,
            zeros_mode,
            enable_tuning,
            fast_decoding,
            bias,
            propagate_b,
        )
        self._initialize_buffers(infeatures, outfeatures, bias)

    def init_params(self):
        # eliminate runtime overhead like exllama state
        if self.is_consitent:
            param_list = [self.weight]
            if self.bitblas_matmul.config.with_bias:
                param_list.append(self.bias)
            self.q_params = [ctypes.c_void_p(arr.data_ptr()) for arr in param_list]
        else:
            param_list = [self.qweight]
            if self.bitblas_matmul.config.with_scaling:
                param_list.append(self.scales)
            if self.bitblas_matmul.config.with_zeros:
                param_list.append(self.zeros)
            if self.bitblas_matmul.config.with_bias:
                param_list.append(self.bias)
            self.q_params = [ctypes.c_void_p(arr.data_ptr()) for arr in param_list]

    def _validate_parameters(self, group_size, infeatures, outfeatures):
        if infeatures % 16 != 0 or outfeatures % 16 != 0:
            raise ValueError("`infeatures` and `outfeatures` must be divisible by 16.")
        if infeatures % group_size != 0:
            raise ValueError("`infeatures` must be divisible by `group_size`.")

    def _set_group_size(self, group_size, infeatures):
        return infeatures if group_size == -1 else group_size

    def _initialize_buffers(self, infeatures, outfeatures, bias):
        if self.consistent:
            self.register_buffer(
                "weight",
                torch.zeros(
                    (outfeatures, infeatures // self.group_size), dtype=self.torch_dtype
                ),
            )
        else:
            self.register_buffer(
                "qweight",
                torch.zeros(
                    self.bitblas_matmul.retrieve_weight_shape(),
                    dtype=self.TORCH_STORAGE_DTYPE,
                ),
            )
            self.register_buffer(
                "scales",
                torch.zeros(
                    (outfeatures, infeatures // self.group_size), dtype=self.torch_dtype
                ),
            )
            if self.zeros_mode == "quantized":
                storage_nbit = int("".join(c for c in self.STORAGE_DTYPE if c.isdigit()))
                self.register_buffer(
                    "zeros",
                    torch.zeros(
                        (
                            infeatures // self.group_size,
                            outfeatures // storage_nbit * self.bits,
                        ),
                        dtype=self.TORCH_STORAGE_DTYPE,
                    ),
                )
            else:
                self.register_buffer(
                    "zeros",
                    torch.zeros(
                        (outfeatures, infeatures // self.group_size), dtype=self.torch_dtype
                    ),
                )
        if bias:
            self.register_buffer(
                "bias", torch.zeros((outfeatures), dtype=self.torch_dtype)
            )
        else:
            self.bias = None

    def _configure_bitblas_matmul(
        self,
        in_dtype,
        weight_dtype,
        accum_dtype,
        out_dtype,
        with_scaling,
        with_zeros,
        zeros_mode,
        enable_tuning,
        fast_decoding,
        bias,
        propagate_b,
    ):
        matmul_config = MatmulConfig(
            M=self.opt_features,
            N=self.outfeatures,
            K=self.infeatures,
            in_dtype=in_dtype,
            weight_dtype=weight_dtype,
            accum_dtype=accum_dtype,
            out_dtype=out_dtype,
            storage_dtype=self.STORAGE_DTYPE,
            with_scaling=with_scaling,
            with_zeros=with_zeros,
            group_size=self.group_size,
            fast_decoding=fast_decoding,
            with_bias=bias,
            propagate_b=propagate_b,
            zeros_mode=zeros_mode,
        )
        self.bitblas_matmul = self._get_or_create_bitblas_operator(
            matmul_config, enable_tuning
        )

    def _get_or_create_bitblas_operator(self, config, enable_tuning):
        bitblas_matmul = global_operator_cache.get(config)
        if bitblas_matmul is None:
            bitblas_matmul = Matmul(config, target=BITBLAS_TARGET)
            if enable_tuning:
                bitblas_matmul.hardware_aware_finetune(topk=20)
                global_operator_cache.add(config, bitblas_matmul)
                global_operator_cache.save_into_database(
                    BITBLAS_DATABASE_PATH, BITBLAS_TARGET
                )
                print(
                    "BitBLAS Tuning done, appended operator to global_operator_cache."
                )
            else:
                print("BitBLAS Operator created.")
        else:
            print("BitBLAS Operator found in global_operator_cache.")
        return bitblas_matmul

    def warmup(self, topk=20):
        self.bitblas_matmul.hardware_aware_finetune(topk=topk)

    def forward(self, A, Output=None):
        if A.dtype != torch.float16:
            A = A.half()
        
        # can be lifted to post init.
        self.init_params()

        if Output is None:
            Output = torch.empty(
                A.shape[:-1] + (self.outfeatures,), dtype=A.dtype, device=A.device
            )

        A_void = ctypes.c_void_p(A.data_ptr())
        # m is the product of the last n - 1 dimensions of A
        m = ctypes.c_int32(reduce(operator.mul, A.shape[:-1], 1))
        self.bitblas_matmul.lib.call(
            A_void, *self.q_params, ctypes.c_void_p(Output.data_ptr()), m
        )

        return Output

    @property
    def consistent(self):
        return self.is_consitent

__all__ = ["Linear"]
