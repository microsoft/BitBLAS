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
