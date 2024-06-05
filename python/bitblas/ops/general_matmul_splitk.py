# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from tvm.target import Target
import operator
from functools import reduce
from typing import Any, Optional, Union
from .operator import TransformKind
from .impl.matmul_splitk_impl import select_implementation as consistent_implementation
from .impl.matmul_dequantize_splitk_impl import select_implementation as weight_dequantize_implementation
from dataclasses import dataclass
import logging
import torch
from .general_matmul import MatmulConfig, Matmul
from .general_matmul import is_native_compute

logger = logging.getLogger(__name__)

WORKSPACE_SIZE = 1024 * 1024 * 256


@dataclass(frozen=True)
class MatmulConfigWithSplitK(MatmulConfig):
    k_split: int = 1  # split K dimension


class MatmulWithSplitK(Matmul):

    def __init__(
        self,
        config: MatmulConfig,
        name: str = "matmul",
        target: Optional[Union[str, Target]] = None,
        enable_tuning: bool = True,
        from_database: bool = False,
    ):
        super().__init__(config, name, target, enable_tuning, from_database)

    def _select_implementation(self):
        # the major implementation
        if is_native_compute(self.A_dtype, self.W_dtype):
            return consistent_implementation(
                SplitK=self.k_split,
                M=self.M,
                N=self.N,
                K=self.K,
                in_dtype=self.A_dtype,
                out_dtype=self.out_dtype,
                accum_dtype=self.accum_dtype,
                with_bias=self.with_bias,
                layout=self.layout,
                propagate_a=self.propagate_a,
                propagate_b=self.propagate_b,
            )
        else:
            return weight_dequantize_implementation(
                SplitK=self.k_split,
                M=self.M,
                N=self.N,
                K=self.K,
                in_dtype=self.A_dtype,
                out_dtype=self.out_dtype,
                accum_dtype=self.accum_dtype,
                bit=self.bit,
                storage_dtype=self.storage_dtype,
                source_format=self.source_format,
                with_scaling=self.with_scaling,
                with_zeros=self.with_zeros,
                group_size=self.group_size,
                fast_decoding=self.fast_decoding,
                with_bias=self.with_bias,
                layout=self.layout,
                zeros_mode=self.zeros_mode,
                propagate_a=self.propagate_a,
                propagate_b=self.propagate_b,
            )

    def retrieve_weight_shape(self):
        return [int(i) for i in self.prim_func.buffer_map[self.prim_func.params[1]].shape]

    def transform_weight(self, weight, scale=None, zeros=None, bias=None):
        """
        Transforms the given weight tensor based on the specified quantization parameters and
        returns the transformed weight along with optional scale, zeros, and bias.

        Parameters:
        - weight: The input weight tensor to be transformed.
        - scale: Optional scaling factor for the weight tensor.
        - zeros: Optional zero-point adjustment for the weight tensor.
        - bias: Optional bias to be added to the weight tensor.

        Returns:
        A list containing the transformed weight tensor and optionally the scale, zeros, and bias.
        """
        weight = weight.contiguous()
        if self.W_dtype == self.A_dtype:
            if self.weight_transform is not None:
                return self.weight_transform(weight.cpu()).cuda().contiguous()
            return weight

        from bitblas.quantization import general_compress
        import torch
        import numpy as np

        source_format, bit = self.source_format, self.bit

        # Process integer source format
        if source_format == "int" and bit < 8:
            assert not self.with_scaling, "scale should be False for int source format"
            assert not self.with_zeros, "zeros should be False for int source format"
            maxq = 2**(bit - 1)
            # Clamp weight values to be within the quantizable range and adjust
            weight = torch.clamp(weight, -maxq, maxq).int() + maxq
        elif source_format in ["fp_e5m2", "fp_e4m3"]:
            weight = weight.view(torch.int8)
            weight = weight.int()
        else:
            # For non-integer formats, simply convert weights to integers
            weight = weight.int()

        np_storage_dtype = getattr(np, self.storage_dtype)

        weight = general_compress(
            weight.cpu().numpy(), source_bits=bit, storage_dtype=np_storage_dtype)

        weight = torch.from_numpy(weight).cuda().contiguous()

        # Apply an optional weight transformation if specified
        if self.weight_transform is not None:
            weight = self.weight_transform(weight.cpu()).cuda().contiguous()

        # Prepare the return list with the transformed weight and optionally include scale, zeros, and bias
        result = [weight]
        if scale is not None:
            result.append(scale)
        if zeros is not None:
            result.append(zeros)
        if bias is not None:
            result.append(bias)

        return next(iter(result), result)

    def transform_input(self, input_tensor):
        if self.propagate_a is not TransformKind.NonTransform:
            # check workspace size
            if input_tensor.numel() > WORKSPACE_SIZE:
                raise ValueError(
                    f"Input size {input_tensor.numel()} is larger than the workspace size {WORKSPACE_SIZE}, please increase the workspace size."
                )
            self.ladder_permutate_a._forward_from_prebuild_lib(input_tensor, self.workspace)
            return self.workspace
        return input_tensor

    def forward(self, A, W, scale=None, zeros=None, bias=None, output=None) -> Any:
        args = []
        args.append(self.transform_input(A))
        args.append(W)

        if self.lut is not None:
            args.append(self.lut)

        if output is None:
            output = torch.empty(
                (self.k_split,) + A.shape[:-1] + (self.N,),
                dtype=self.torch_output_dtype,
                device=A.device)
        if scale is not None:
            args.append(scale)
        if zeros is not None:
            args.append(zeros)
        if bias is not None:
            args.append(bias)
        args.append(output)

        if self.dynamic_range is not None:
            m = reduce(operator.mul, A.shape[:-1], 1)
            args.append(m)

        stream = torch.cuda.current_stream()

        if self.lib is None:
            self._forward_from_torch_func(*args)
        self._forward_from_prebuild_lib(*args, stream=stream.cuda_stream)
        output = torch.sum(output, dim=0)
        return output

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    @property
    def k_split(self):
        return self.config.k_split


__all__ = ["MatmulConfigWithSplitK", "MatmulWithSplitK"]
