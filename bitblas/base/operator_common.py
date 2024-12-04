# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from enum import IntEnum


class OptimizeStrategy(IntEnum):
    SingleBatchDecodeOnly = 0
    ContigousBatching = 1


class TransformKind(IntEnum):
    NonTransform = 0
    InterWarpTransform = 1
    IntraWarpTransform = 2
    LDMatrixTransform = 3


class BackendKind(IntEnum):
    TIR = 0
    TileLang = 1

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


class DequantizeStage(IntEnum):
    Local = 0
    Shared = 1
    Global = 2
