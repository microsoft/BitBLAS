# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from logging import getLogger

import numpy as np
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

from bitblas.quantization.utils import general_compress, interleave_weight
from bitblas.ops.matmul import MatmulWeightOnlyDequantize


class QuantLinear(nn.Module):
    QUANT_TYPE = "bitblas"

    def __init__(
        self, bits, group_size, infeatures, outfeatures, bias, trainable=False, **kwargs
    ):
        super().__init__()
        if infeatures % 128 != 0 or outfeatures % 256 != 0:
            raise ValueError(
                "`infeatures` must be divisible by 128 and `outfeatures` by 256."
            )
        if bits not in [1, 2, 4]:
            raise NotImplementedError("Only 1/2/4 bits are supported.")
        if infeatures % group_size != 0:
            raise ValueError("`infeatures` must be divisible by `group_size`.")
        if trainable:
            raise NotImplementedError("Bitblas does not support train.")

        self.bits = bits
        storage_nbit = 8  # assume int8 storage
        n_float_per_elem = storage_nbit // bits
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.group_size = group_size if group_size != -1 else infeatures
        self.register_buffer(
            "qweight",
            torch.empty(
                (self.outfeatures, self.infeatures // storage_nbit * n_float_per_elem),
                dtype=torch.uint8,
            ),
        )
        self.register_buffer(
            "scales",
            torch.empty(
                (self.outfeatures, self.infeatures // self.group_size), dtype=torch.half
            ),
        )
        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=torch.half))
        else:
            self.bias = None

        self.fast_type_conversion = False
        self.weight_propagation = False

        # optimize target shapes for dynamic symbolic
        OPTIMIZE_M_RANGE = [1, 16, 32]
        self.bitblas_matmul = MatmulWeightOnlyDequantize(
            M=1,
            N=outfeatures,
            K=infeatures,
            in_dtype="float16",
            out_dtype="float16",
            accum_dtype="float16",
            propagate_b=self.weight_propagation,
            bit=self.bits,
            storage_dtype="uint8",
            source_format="int",
            with_scaling=True,
            group_size=self.group_size,
            fast_decoding=self.fast_type_conversion,
            with_bias=bias,
        )
        # self.bitblas_matmul.optimize(topk=20)

    def post_init(self):
        pass

    def pack(self, linear, scales):
        """Pack a fake-quantized linear layer into this actual Bitblas representation.
        @linear: fake-quantized `torch.nn.Linear` layer to convert (must be of type `torch.half`)
        @scales: corresponding quantization scales of shape `(infeatures, groups)`
        """
        if linear.weight.dtype != torch.half:
            raise ValueError("Only `torch.half` weights are supported.")

        # do permutation with (n, k) layout
        w = linear.weight.data
        # scales shape should be (n, k) as well.
        s = scales
        # do permutation on weight
        intweight = []
        for idx in range(self.infeatures):
            g_idx = idx // self.group_size
            intweight.append(
                torch.round((w[:, idx]) / scales[:, g_idx]).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.contiguous()
        intweight = intweight.cpu().numpy().astype(np.int8)
        print("bitblas dequantize weight is ")
        print(intweight)
        intweight = intweight + 7
        print("bitblas dequantize weight +7 is ")
        print(intweight)
        # quantize to 4bit
        qw_np = general_compress(
            intweight, source_bits=self.bits, storage_dtype=np.uint8
        )
        # do interleave for fast type conversion
        if self.fast_type_conversion:
            qw_np = interleave_weight(qw_np, nbits=self.bits, target_dtype="float16")
        if self.weight_propagation:
            # do permutation on weight
            pass

        q = torch.from_numpy(qw_np).to(w.device)
        self.qweight = q.to(self.qweight.device).contiguous()
        self.scales = s.to(self.scales.device).contiguous()

        if self.bias is not None:
            self.bias[:] = linear.bias.data.to(self.bias.device).contiguous()

    def forward(self, A):
        A = A.half()
        C = torch.empty(
            A.shape[:-1] + (self.qweight.shape[0],), dtype=A.dtype, device=A.device
        )
        args = [A, self.qweight, self.scales]
        if self.bias is not None:
            args.append(self.bias)
        args.append(C)

        self.bitblas_matmul(*args)

        return C


__all__ = ["QuantLinear"]
