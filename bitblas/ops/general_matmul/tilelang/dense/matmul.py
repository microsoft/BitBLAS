# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm as tvm
from tvm import DataType
import tvm.tl.language as T
from typing import Optional, List
from tvm.tir import PrimFunc
from bitblas.base.operator_common import TransformKind
from bitblas.base.base_scheduler import BaseScheduler
from bitblas.base.arch import TileDevice, auto_infer_current_arch, is_ampere_arch, is_volta_arch
from bitblas.base.roller.hint import Hint
from bitblas.base.roller.rasterization import NoRasterization
from bitblas.base.utils import get_roller_hints_from_func
from dataclasses import dataclass
from bitblas.ops.general_matmul.tirscript import (matmul_select_implementation)
from bitblas.tl.base_hint import BaseTLHint

from .base import MatmulBaseParams
from .gemv_simt import GemvFineGrainSIMTScheduler
from .matmul_simt import MatmulFineGrainSIMTScheduler
from .matmul_tensorcore import (
    MatmulBlockScheduler,
    MatmulFineGrainScheduler,
    MatmulWeightPropagationScheduler,
    MatmulINT4FineGrainScheduler,
    MatmulINT4WeightPropagationScheduler,
)

import logging
logger = logging.getLogger(__name__)

def is_tensorcore_precision_supported(in_dtype:str, accum_dtype:str, arch:TileDevice) -> bool:
    volta_tensorcore_supported = [
        ("float16", "float32"),
        ("float16", "float16"),
    ]
    ampere_tensorcore_supported = [
        ("float16", "float32"),
        ("float16", "float16"),
        ("int8", "int32"),
        ("int4", "int32"),
        ("int2", "int32"),
        ("int1", "int32"),
    ]
    
    if is_volta_arch(arch):
        return (in_dtype, accum_dtype) in volta_tensorcore_supported
    elif is_ampere_arch(arch):
        return (in_dtype, accum_dtype) in ampere_tensorcore_supported
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    

@dataclass
class MatmulScheduler(MatmulBaseParams):
    # Fine-grained matrix multiplication scheduler
    # Allows for more detailed configuration.
    
    gemv_scheduler: Optional[GemvFineGrainSIMTScheduler] = None
    matmul_simt_scheduler: Optional[MatmulFineGrainSIMTScheduler] = None
    matmul_block_scheduler: Optional[MatmulBlockScheduler] = None
    matmul_fine_grain_scheduler: Optional[MatmulFineGrainScheduler] = None
    matmul_weight_propagation_scheduler: Optional[MatmulWeightPropagationScheduler] = None
    matmul_int4_fine_grain_scheduler: Optional[MatmulINT4FineGrainScheduler] = None
    matmul_int4_weight_propagation_scheduler: Optional[MatmulINT4WeightPropagationScheduler] = None 
    

    def __init__(self, **kwargs):
        self.gemv_scheduler = GemvFineGrainSIMTScheduler(**kwargs)
        self.matmul_simt_scheduler = MatmulFineGrainSIMTScheduler(**kwargs)
        self.matmul_block_scheduler = MatmulBlockScheduler(**kwargs)
        self.matmul_fine_grain_scheduler = MatmulFineGrainScheduler(**kwargs)
        self.matmul_weight_propagation_scheduler = MatmulWeightPropagationScheduler(**kwargs)
        self.matmul_int4_fine_grain_scheduler = MatmulINT4FineGrainScheduler(**kwargs)
        self.matmul_int4_weight_propagation_scheduler = MatmulINT4WeightPropagationScheduler(**kwargs)
        super().__init__(**kwargs)

    def dispatch_ampere_scheduler(self, arch:TileDevice) -> BaseScheduler:
        M, N, K = self.M, self.N, self.K       
        is_dynamic = (
            M is None or N is None or K is None
        )
        in_dtype, accum_dtype = (
            self.in_dtype,
            self.accum_dtype,
        )
        if is_dynamic:
            # Dynamic Dispatcher
            if is_tensorcore_precision_supported(in_dtype, accum_dtype, arch):
                return self.matmul_fine_grain_scheduler
            else:
                return self.matmul_simt_scheduler
        else:
            minimal_tensorcore_threshold: List[int, int, int] = [8, 16, 32] if accum_dtype == "int32" else [8, 16, 16]
            if M < minimal_tensorcore_threshold[0] or N < minimal_tensorcore_threshold[1] or K < minimal_tensorcore_threshold[2]:
                return self.gemv_scheduler
            elif is_tensorcore_precision_supported(in_dtype, accum_dtype, arch):
                if self.weight_transform_kind != TransformKind.NonTransform:
                    return self.matmul_weight_propagation_scheduler
                else:
                    return self.matmul_fine_grain_scheduler
            else:
                return self.matmul_simt_scheduler
    
    def dispatch_volta_scheduler(self, arch:TileDevice) -> BaseScheduler:
        M, N, K = self.M, self.N, self.K       
        is_dynamic = (
            M is None or N is None or K is None
        )
        in_dtype, accum_dtype = (
            self.in_dtype,
            self.accum_dtype,
        )
        if self.weight_transform_kind != TransformKind.NonTransform:
            raise ValueError(f"Weight propagation {self.weight_transform_kind} is not supported for Volta")
        if in_dtype not in ["int8", "float16", "float32", "float64"]:
            raise ValueError(f"Unsupported input data type: {in_dtype}")

        if is_dynamic:
            # Dynamic Dispatcher
            if is_tensorcore_precision_supported(in_dtype, accum_dtype, arch):
                return self.matmul_fine_grain_scheduler
            else:
                return self.matmul_simt_scheduler
        else:
            minimal_tensorcore_threshold: List[int, int, int] = [8, 16, 16]
            if M < minimal_tensorcore_threshold[0] or N < minimal_tensorcore_threshold[1] or K < minimal_tensorcore_threshold[2]:
                return self.gemv_scheduler
            elif is_tensorcore_precision_supported(in_dtype, accum_dtype, arch):
                # Fine-grained scheduler (mma) is not supported for Volta
                return self.matmul_block_scheduler
            else:
                return self.matmul_simt_scheduler       

    def with_default_config(self, arch: Optional[TileDevice] = None) -> PrimFunc:
        if arch is None:
            arch = auto_infer_current_arch()
            logger.debug(f"arch is not specified in with_default_config, auto-infer to {arch}")

        dispatched_scheduler: Optional[BaseScheduler] = None
        if is_ampere_arch(arch):
            dispatched_scheduler = self.dispatch_ampere_scheduler(arch)
        elif is_volta_arch(arch):
            dispatched_scheduler = self.dispatch_volta_scheduler(arch)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        return dispatched_scheduler.with_default_config()

    def apply_config(
        self,
        hint: Optional[BaseTLHint] = None,
        arch: Optional[TileDevice] = None,
    ):
        if arch is None:
            arch = auto_infer_current_arch()
            logger.debug(f"arch is not specified in apply_config, auto-infer to {arch}")

        dispatched_scheduler: Optional[BaseScheduler] = None
        if is_ampere_arch(arch):
            dispatched_scheduler = self.dispatch_ampere_scheduler(arch)
        elif is_volta_arch(arch):
            dispatched_scheduler = self.dispatch_volta_scheduler(arch)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        
        return dispatched_scheduler.apply_config(
            block_size_x=block_size_x,
            block_size_y=block_size_y,
            thread_row_tiles=thread_row_tiles,
            thread_col_tiles=thread_col_tiles,
            chunk=chunk,
        )

    def __post_init__(self):
        # Validate the matrix transpose settings
        assert self.trans_A is False, "Currently only support Matrix A not transposed"
        assert self.trans_B is True, "Currently only support Matrix B transposed"
        assert self.with_bias is False, "Currently only support without bias"
        assert self.input_transform_kind == TransformKind.NonTransform, "Currently only support NonTransform for input"

        return

__all__ = ["MatmulScheduler"]