# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm as tvm
from typing import Optional, List, Dict
from tvm.tir import PrimFunc
from bitblas.base.operator_common import TransformKind
from bitblas.base.base_scheduler import BaseScheduler
from bitblas.base.arch import TileDevice, auto_infer_current_arch, is_ampere_arch, is_volta_arch
from dataclasses import dataclass
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


def is_tensorcore_precision_supported(in_dtype: str, accum_dtype: str, arch: TileDevice) -> bool:
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


@dataclass(repr=False)
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
        self.matmul_int4_weight_propagation_scheduler = MatmulINT4WeightPropagationScheduler(
            **kwargs)
        super().__init__(**kwargs)

    def dispatch_ampere_scheduler(self, arch: TileDevice) -> BaseScheduler:
        M, N, K = self.M, self.N, self.K
        is_dynamic = self.is_dynamic
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
            minimal_tensorcore_threshold: List[int, int,
                                               int] = [8, 16, 32
                                                      ] if accum_dtype == "int32" else [8, 16, 16]
            if minimal_tensorcore_threshold[0] > M or minimal_tensorcore_threshold[
                    1] > N or minimal_tensorcore_threshold[2] > K:
                return self.gemv_scheduler
            elif is_tensorcore_precision_supported(in_dtype, accum_dtype, arch):
                if self.weight_transform_kind != TransformKind.NonTransform:
                    return self.matmul_weight_propagation_scheduler
                else:
                    return self.matmul_fine_grain_scheduler
            else:
                return self.matmul_simt_scheduler

    def dispatch_volta_scheduler(self, arch: TileDevice) -> BaseScheduler:
        M, N, K = self.M, self.N, self.K
        is_dynamic = self.is_dynamic
        in_dtype, accum_dtype = (
            self.in_dtype,
            self.accum_dtype,
        )
        if self.weight_transform_kind != TransformKind.NonTransform:
            raise ValueError(
                f"Weight propagation {self.weight_transform_kind} is not supported for Volta")
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
            if minimal_tensorcore_threshold[0] > M or minimal_tensorcore_threshold[
                    1] > N or minimal_tensorcore_threshold[2] > K:
                return self.gemv_scheduler
            elif is_tensorcore_precision_supported(in_dtype, accum_dtype, arch):
                # Fine-grained scheduler (mma) is not supported for Volta
                return self.matmul_block_scheduler
            else:
                return self.matmul_simt_scheduler

    def dispatch_scheduler(self, arch: TileDevice) -> BaseScheduler:
        if is_ampere_arch(arch):
            return self.dispatch_ampere_scheduler(arch)
        elif is_volta_arch(arch):
            return self.dispatch_volta_scheduler(arch)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

    def detect_scheduler_from_hint(self, hint: BaseTLHint) -> BaseScheduler:
        for scheduler in [
                self.gemv_scheduler,
                self.matmul_simt_scheduler,
                self.matmul_block_scheduler,
                self.matmul_fine_grain_scheduler,
                self.matmul_weight_propagation_scheduler,
        ]:
            if isinstance(hint, scheduler.TLHint):
                return scheduler
        raise ValueError(f"Unsupported hint type: {type(hint)}")

    def with_default_config(self, arch: Optional[TileDevice] = None) -> PrimFunc:
        if arch is None:
            arch = auto_infer_current_arch()
            logger.debug(f"arch is not specified in with_default_config, auto-infer to {arch}")

        dispatched_scheduler = self.dispatch_scheduler(arch)

        return dispatched_scheduler.with_default_config()

    def get_hardware_aware_configs(self,
                                   arch: Optional[TileDevice] = None,
                                   topk: int = 10) -> List[PrimFunc]:
        if arch is None:
            arch = auto_infer_current_arch()
            logger.debug(
                f"arch is not specified in get_hardware_aware_configs, auto-infer to {arch}")

        dispatched_scheduler = self.dispatch_scheduler(arch)

        return dispatched_scheduler.get_hardware_aware_configs(arch, topk=topk)

    def apply_config(
        self,
        hint: Optional[BaseTLHint] = None,
        arch: Optional[TileDevice] = None,
    ):
        if hint is None:
            raise ValueError("hint is required for apply_config")

        if arch is None:
            arch = auto_infer_current_arch()
            logger.debug(f"arch is not specified in apply_config, auto-infer to {arch}")

        target_scheduler = self.detect_scheduler_from_hint(hint)

        return target_scheduler.apply_config(**hint.get_config_params())

    def specialize_from_dynamic_range(self,
                                      dynamic_range: Optional[Dict[str, int]] = None
                                     ) -> "MatmulScheduler":
        if dynamic_range is None:
            dynamic_range = self._dynamic_range

        assert (dynamic_range
                is not None), "dynamic_range is required for specialize_from_dynamic_range"

        class_attributes = self.params_as_dict()
        for symbol, value in dynamic_range.items():
            attribute_name = symbol.upper()
            if attribute_name not in class_attributes:
                raise ValueError(f"Unknown symbol: {symbol}")
            class_attributes[attribute_name] = value
        return MatmulScheduler(**class_attributes).set_dynamic_range(dynamic_range)

    def set_dynamic_range(self, dynamic_range: Dict[str, int]) -> "BaseScheduler":
        super().set_dynamic_range(dynamic_range)
        for scheduler in [
                self.gemv_scheduler,
                self.matmul_simt_scheduler,
                self.matmul_block_scheduler,
                self.matmul_fine_grain_scheduler,
                self.matmul_weight_propagation_scheduler,
        ]:
            scheduler.set_dynamic_range(dynamic_range)
        return self

    @property
    def is_dynamic(self) -> bool:
        M, N, K = self.M, self.N, self.K
        return ((not isinstance(M, int)) or (not isinstance(N, int)) or (not isinstance(K, int)))

    def __post_init__(self):
        # Validate the matrix transpose settings
        assert self.trans_A is False, "Currently only support Matrix A not transposed"
        assert self.trans_B is True, "Currently only support Matrix B transposed"
        assert self.with_bias is False, "Currently only support without bias"
        assert self.input_transform_kind == TransformKind.NonTransform, "Currently only support NonTransform for input"

        return


__all__ = ["MatmulScheduler"]