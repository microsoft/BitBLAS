# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from enum import IntEnum


class OptimizeStrategy(IntEnum):
    SingleBatchDecodeOnly = 0
    ContigousBatching = 1

    def is_single_batch_decode_only(self):
        return self == OptimizeStrategy.SingleBatchDecodeOnly

    def is_contigous_batching(self):
        return self == OptimizeStrategy.ContigousBatching


class TransformKind(IntEnum):
    NonTransform = 0
    InterWarpTransform = 1
    IntraWarpTransform = 2
    LDMatrixTransform = 3

    def is_non_transform(self):
        return self == TransformKind.NonTransform

    def is_inter_warp_transform(self):
        return self == TransformKind.InterWarpTransform

    def is_intra_warp_transform(self):
        return self == TransformKind.IntraWarpTransform

    def is_ld_matrix_transform(self):
        return self == TransformKind.LDMatrixTransform


class BackendKind(IntEnum):
    TIR = 0
    TileLang = 1

    def is_tir_backend(self):
        return self == BackendKind.TIR

    def is_tilelang_backend(self):
        return self == BackendKind.TileLang


class QuantizationMemoryStage(IntEnum):
    # Represents in which stage the dequantize operation is performed
    #
    # 1. For devices without async copy, we can use a simple dequantize schedule
    # without shared memory prefetch.
    #     quantized weight
    #         |
    #         V
    #     dequantized in register
    #         |
    #         V
    #     save into shared memory
    #         |
    #         V
    #     compute
    #
    # 2. For A100 Like devices, the shared memory prefetch(async) is required
    # to achieve optimal performance.
    #     quantized weight
    #         |
    #         V
    #     shared memory prefetch (with async copy)
    #         |
    #         V
    #     dequantized into shared memory
    #         |
    #         V
    #     compute
    # 3. For A100 Like devices, the shared memory prefetch(async) is required
    # to achieve optimal performance.
    #     quantized weight
    #         |
    #         V
    #     shared memory prefetch (with async copy)
    #         |
    #         V
    #     LDMatrix into warp memory
    #         |
    #         V
    #     Dequantize
    #         |
    #         V
    #     Compute
    Local = 0
    Shared = 1
    Global = 2

    def is_quant_memory_in_local(self):
        return self == QuantizationMemoryStage.Local

    def is_quant_memory_in_shared(self):
        return self == QuantizationMemoryStage.Shared

    def is_quant_memory_in_global(self):
        return self == QuantizationMemoryStage.Global
