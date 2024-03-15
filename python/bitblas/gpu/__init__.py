# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
GPU-generic schedule rules.
For CUDA/ROCm/Vulkan/Metal-specific rules, use `tvm.dlight.cuda/rocm/vulkan/metal` instead
"""
from .fallback import Fallback
from .element_wise import ElementWise
from .gemv import GEMV
from .gemv_dequantize import GEMVWithDequantizeInfo
from .general_reduction import GeneralReduction
from .matmul import (
    Matmul,
    MatmulTensorizationMMA,
    MatmulTensorizationWMMA,
    MatmulTensorizationLegacy,
)
from .matmul_mma_dequantize import MatmulTensorizationMMAWithDequantizeInfo

from .reduction import Reduction
from .transpose import Transpose
