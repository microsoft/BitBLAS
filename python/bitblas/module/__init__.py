# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ctypes
import operator
from functools import reduce
from logging import getLogger

import torch
import torch.nn as nn

logger = getLogger(__name__)

from typing import List, Union, Optional

from bitblas.cache import global_operator_cache, get_database_path
from bitblas import Matmul, MatmulConfig
from bitblas.quantization.utils import general_compress
from bitblas import auto_detect_nvidia_target

BITBLAS_TARGET = auto_detect_nvidia_target()
BITBLAS_DATABASE_PATH = get_database_path()


def unpack_qzeros(qzeros, bits):
    qzeros = qzeros.view(torch.int32)
    elems_per_int32 = 32 // bits
    unpacked_zeros = torch.zeros(
        (qzeros.shape[0], qzeros.shape[1] * elems_per_int32),
        dtype=torch.int8,
        device=qzeros.device,
        requires_grad=False,
    )

    for col in range(unpacked_zeros.shape[1]):
        i = col % elems_per_int32
        unpacked_zeros[:, col] = (qzeros[:, col // elems_per_int32] >> (bits * i)) & 0xF

    return unpacked_zeros + 1


class Linear(nn.Module):
    opt_M = [1, 16, 32, 64, 128, 256, 512]
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
        in_features: int,
        out_features: int,
        bias: bool = False,
        A_dtype: str = "float16",
        W_dtype: str = "float16",
        accum_dtype: str = "float16",
        out_dtype: str = "float16",
        # configs for weight only quantization
        group_size: int = -1,
        with_scaling: bool = None,
        with_zeros: bool = False,
        zeros_mode: str = None,
        opt_M: Union[int, List[int]] = opt_M,
        # performance related configs
        enable_tuning: bool = True,
        fast_decoding: Optional[bool] = None,
        propagate_b: bool = False,
    ):
        """
        @opt_M: optimize range of the input shape for dynamic symbolic
        if the input shape is a range, we will optimize the matmul with dynamic symbolic.
        if the input shape is int, we will optimize the matmul with static symbolic.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.opt_M = opt_M
        self.group_size = self._set_group_size(group_size, in_features)
        self.torch_dtype = getattr(torch, A_dtype)
        self.is_consitent = A_dtype == W_dtype
        self.zeros_mode = zeros_mode
        self._validate_parameters(self.group_size, in_features, out_features)
        self._configure_bitblas_matmul(
            A_dtype,
            W_dtype,
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
        self._initialize_buffers(in_features, out_features, bias)

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

    def _validate_parameters(self, group_size, in_features, out_features):
        if in_features % 16 != 0 or out_features % 16 != 0:
            raise ValueError("`in_features` and `out_features` must be divisible by 16.")
        if in_features % group_size != 0:
            raise ValueError("`in_features` must be divisible by `group_size`.")

    def _set_group_size(self, group_size, in_features):
        return in_features if (group_size == -1 or group_size is None) else group_size

    def _initialize_buffers(self, in_features, out_features, bias):
        if self.consistent:
            self.register_buffer(
                "weight",
                torch.zeros((out_features, in_features // self.group_size), dtype=self.torch_dtype),
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
                torch.zeros((out_features, in_features // self.group_size), dtype=self.torch_dtype),
            )
            if self.zeros_mode == "quantized":
                storage_nbit = int("".join(c for c in self.STORAGE_DTYPE if c.isdigit()))
                self.register_buffer(
                    "zeros",
                    torch.zeros(
                        (
                            in_features // self.group_size,
                            out_features // storage_nbit * self.bits,
                        ),
                        dtype=self.TORCH_STORAGE_DTYPE,
                    ),
                )
            else:
                self.register_buffer(
                    "zeros",
                    torch.zeros(
                        (out_features, in_features // self.group_size),
                        dtype=self.torch_dtype,
                    ),
                )
        if bias:
            self.register_buffer("bias", torch.zeros((out_features), dtype=self.torch_dtype))
        else:
            self.bias = None

    def _configure_bitblas_matmul(
        self,
        A_dtype,
        W_dtype,
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
            M=self.opt_M,
            N=self.out_features,
            K=self.in_features,
            A_dtype=A_dtype,
            W_dtype=W_dtype,
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
        self.bitblas_matmul = self._get_or_create_bitblas_operator(matmul_config, enable_tuning)
        self.bits = self.bitblas_matmul.bit
        self.source_format = self.bitblas_matmul.source_format

    def _get_or_create_bitblas_operator(self, config, enable_tuning):
        if global_operator_cache.size() == 0:
            global_operator_cache.load_from_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
            logger.info(f"Loaded {global_operator_cache.size()} operators from database.")

        bitblas_matmul = global_operator_cache.get(config)
        if bitblas_matmul is None:
            # should disable tuning for the first time because we may require loading bitblas operator from database.
            bitblas_matmul = Matmul(config, target=BITBLAS_TARGET, enable_tuning=False)
            if enable_tuning:
                bitblas_matmul.hardware_aware_finetune(topk=20)
                global_operator_cache.add(config, bitblas_matmul)
                global_operator_cache.save_into_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
                print("BitBLAS Tuning done, appended operator to global_operator_cache.")
            else:
                print("BitBLAS Operator created.")
        else:
            print("BitBLAS Operator found in global_operator_cache.")
        return bitblas_matmul

    def warmup(self, topk=20):
        self.bitblas_matmul.hardware_aware_finetune(topk=topk)

    def forward(self, A, output=None):
        if A.dtype != torch.float16:
            A = A.half()
        # can be lifted to post init.
        self.init_params()

        if output is None:
            output = torch.empty(
                A.shape[:-1] + (self.out_features,), dtype=A.dtype, device=A.device)
        m = ctypes.c_int32(reduce(operator.mul, A.shape[:-1], 1))
        A = self.bitblas_matmul.transform_input(A)
        A_void = ctypes.c_void_p(A.data_ptr())
        # m is the product of the last n - 1 dimensions of A
        self.bitblas_matmul.lib.call(A_void, *self.q_params, ctypes.c_void_p(output.data_ptr()), m)

        return output

    def load_and_transform_weight(
        self,
        weight: torch.Tensor,
        scales: torch.Tensor = None,
        zeros: torch.Tensor = None,
        bias: torch.Tensor = None,
    ):
        if self.consistent:
            assert scales is None, "scales should be None for consistent mode."
            assert zeros is None, "zeros should be None for consistent mode."
            weight = self.bitblas_matmul.transform_weight(weight)
            self.weight = nn.Parameter(weight)
            if bias is not None:
                self.bias = bias
        else:
            weight = self.bitblas_matmul.transform_weight(weight)
            self.qweight = weight
            if scales is not None:
                self.scales = scales
            if zeros is not None:
                self.zeros = zeros
            if bias is not None:
                self.bias = bias

    def repack_from_gptq(self, gptq_module):
        # qweight in gptq old quant linear stored with (out_features, in_features), should be transposed.
        qweight = gptq_module.qweight.T.contiguous().view(self.TORCH_STORAGE_DTYPE)
        if self.bitblas_matmul.weight_transform is not None:
            qweight = self.bitblas_matmul.weight_transform(qweight.cpu()).cuda()
        self.qweight = qweight
        # scales in gptq old quant linear stored with (in_features // group_size, out_features), should be transposed.
        scales = gptq_module.scales.T.contiguous().view(self.torch_dtype)
        self.scales = scales
        # qzeros should be dequantized to int zeros.
        intzeros = unpack_qzeros(gptq_module.qzeros, self.bits).T.contiguous()
        if self.bitblas_matmul.config.zeros_mode == "original":
            self.zeros = intzeros.to(torch.float16).contiguous()
        elif self.bitblas_matmul.config.zeros_mode == "rescale":
            self.zeros[:, :] = intzeros.to(torch.float16)[:, :] * self.scales[:, :]
        elif self.bitblas_matmul.config.zeros_mode == "quantized":
            self.zeros = (
                torch.Tensor(general_compress(intzeros.T.contiguous().cpu().numpy(), self.bits)).to(
                    self.qweight.device).to(self.zeros.dtype).contiguous())
        else:
            raise ValueError(f"Unsupported zeros type: {self.bitblas_matmul.config.zeros_mode}")
        if self.bias is not None:
            self.bias = gptq_module.bias.data.to(torch.float16).contiguous()

    @property
    def consistent(self):
        return self.is_consitent


__all__ = ["Linear"]
