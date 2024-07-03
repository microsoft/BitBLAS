# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from abc import ABC, abstractmethod
import tvm
from tvm import IRModule
from tvm.target import Target
from tvm.tir import PrimFunc
from tvm.contrib.dlpack import to_pytorch_func
from tvm._ffi.base import _LIB, raise_last_ffi_error
from tvm._ffi._ctypes.types import TVMValue, ArgTypeCode
import bitblas
import ctypes
from typing import List, Dict, Any, Optional
import numpy as np
from ..base import fast_tune, fast_tune_with_dynamic_range
from copy import deepcopy
from bitblas.base.roller.arch import get_arch
from bitblas.utils.tensor_adapter import tvm_tensor_to_torch
from bitblas.wrapper import CUDASourceWrapper, CUDASourceWrapperWithDynamic
from dataclasses import dataclass
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)


class TransformKind(IntEnum):
    NonTransform = 0
    InterWarpTransform = 1
    IntraWarpTransform = 2


@dataclass
class OperatorConfig:
    """Base class for operator configurations. Used for typing."""

    pass


class Operator(ABC):

    def __init__(self, name, config: OperatorConfig, target: Target = None):
        if isinstance(target, str):
            target = Target(target)
        self.name = name
        self.config = config
        self.target = target
        self.prim_func_mod = self._select_implementation()
        self.optimized_func = None
        self.rt_mod = None
        self.time_evaluator = None
        self.profile_tensors = None
        self.arch = get_arch(target) if target else None
        self.dynamic_range = None
        self.pass_context: Dict = {}
        self.num_args = len(self.prim_func.params)
        self.function_handle = None
        self.num_output_args: int = (
            1  # todo(lei): should be analyzed from the prim_func.
        )
        self.wrapper = None
        self.src_name = None
        self.lib_name = None
        self.lib = None

    def get_source(self, target: Target = None) -> str:
        if target is None:
            target = self.target
        if self.rt_mod is None:
            self._build_runtime_module(target)
        return self.rt_mod.imported_modules[0].get_source() if self.rt_mod else None

    def _build_runtime_module(self, target: Target):
        """
        Builds the runtime module based on the architecture platform.

        This function attempts to build a runtime module (rt_mod) for the specified target.
        If the platform is CUDA and an optimized function is available, it tries to build
        using the optimized function with a specific pass context. Otherwise, it falls back
        to building with the primary function. After successful build, it initializes a
        time evaluator for performance measurement.

        Args:
            target (Target): The compilation target specification.

        Returns:
            The compiled runtime module or None if the build was unsuccessful.
        """

        # Initialize rt_mod as None to handle cases where build fails or is skipped
        rt_mod = None

        # Check if the platform is CUDA and we have an optimized function
        if self.arch.platform == "CUDA":
            if self.optimized_func is None:
                return None

            @tvm.register_func(func_name="tvm_callback_cuda_postproc", override=True)
            def tvm_callback_cuda_postproc(code, _):
                return self.post_process(code)

            try:
                # Use a specific TVM pass context for CUDA platforms
                with tvm.transform.PassContext(config={
                        "tir.use_async_copy": True,
                        **self.pass_context
                }):
                    rt_mod = tvm.build(self.optimized_func, target=target, name=self.name)
            except Exception as e:
                rt_build_error = e  # noqa
                logger.debug(
                    "Failed to build optimized function for CUDA target with default schedule, Please consider enable hardware aware tuning!"
                )
        else:
            # For non-CUDA platforms or when no optimized function is available, build with the primary function
            rt_mod = tvm.build(self.prim_func, target=target, name=self.name)

        # If the runtime module was successfully built, set up for evaluation
        if rt_mod:
            self.rt_mod = rt_mod
            # Initialize a time evaluator with the built module, specifying the device and the number of runs
            self.time_evaluator = rt_mod.time_evaluator(
                rt_mod.entry_name, self.arch.device, number=10)
            self.function_handle = rt_mod.get_function(rt_mod.entry_name).handle
            self.torch_func = to_pytorch_func(rt_mod)
            if self.arch.platform == "CUDA":
                try:
                    if (self.dynamic_range is not None and len(self.optimized_func.functions) > 1):
                        wrapper = CUDASourceWrapperWithDynamic(self.optimized_func,
                                                               self.get_source(target), self.arch)
                    else:
                        wrapper = CUDASourceWrapper(self.optimized_func, self.get_source(target),
                                                    self.arch)
                    wrapper.compile_lib()
                    self.wrapper = wrapper
                    self.src_name = self.wrapper.src_name
                    self.lib_name = self.wrapper.lib_name
                    self.lib = self.wrapper.load_lib()
                    self.lib.init()
                except Exception as e:
                    build_runtime_library_error = e
                    logger.debug(
                        "Failed to build runtime library {}".format(build_runtime_library_error))

        return rt_mod

    def apply_default_schedule(self, func_mod: IRModule, target: Target) -> IRModule:
        mod_for_opt = deepcopy(func_mod)
        with target:
            optimized_mod = (
                bitblas.ApplyDefaultSchedule(  # pylint: disable=not-callable
                    bitblas.gpu.Matmul(),
                    bitblas.gpu.GEMV(),
                    bitblas.gpu.Reduction(),
                    bitblas.gpu.GeneralReduction(),
                    bitblas.gpu.Fallback(),
                )(mod_for_opt))

        if optimized_mod is not None:
            return optimized_mod
        return None

    def post_process(self, code: str) -> str:
        return code

    def apply_fast_tuning(self,
                          func: PrimFunc,
                          target: Target,
                          topk: int = 20,
                          parallel_build=True) -> IRModule:
        _, best = fast_tune(func, target, topk=topk, parallel_build=parallel_build)
        if best is not None:
            return best.sch.mod
        self.pass_context = best.config.pass_context
        return None

    def apply_fast_tuning_with_dynamic_range(
        self,
        func: PrimFunc,
        target: Target,
        topk: int = 20,
        dynamic_range: Dict[str, List[int]] = None,
    ):
        optimized_mod = fast_tune_with_dynamic_range(
            func, target, topk=topk, parallel_build=True, dynamic_range=dynamic_range)
        if optimized_mod is not None:
            return optimized_mod
        return None

    def hardware_aware_finetune(self,
                                topk: int = 20,
                                target: tvm.target.Target = None,
                                parallel_build=True):
        if target is None:
            target = self.target
        dynamic_range = self.dynamic_range
        func = self.prim_func
        if dynamic_range is not None:
            self.optimized_func = self.apply_fast_tuning_with_dynamic_range(
                func, target, topk, dynamic_range)
        else:
            self.optimized_func = self.apply_fast_tuning(
                func, target, topk, parallel_build=parallel_build)
        self._build_runtime_module(self.target)

    def get_profile_tensors(self, dynamic_symbolic_constrains: Optional[Dict] = None):
        if dynamic_symbolic_constrains is None:
            dynamic_symbolic_constrains = {}
        func = self.prim_func
        device = self.arch.device

        def var_warpper(v):
            if isinstance(v, tvm.tir.Var):
                if v.name in dynamic_symbolic_constrains:
                    return dynamic_symbolic_constrains[v.name]
                assert "opt_shapes" in func.attrs
                assert v.name in func.attrs["opt_shapes"]
                return func.attrs["opt_shapes"][v.name].value
            elif isinstance(v, tvm.tir.IntImm):
                return v.value
            else:
                raise RuntimeError("Not supported type: ", type(v))

        def map_numpy_type(intype):
            typemap = {
                'e4m3_float8': 'float8_e4m3fn',
                'e5m2_float8': 'float8_e5m2',
            }
            if intype in typemap:
                return typemap[intype]
            else:
                return intype

        profile_tensors = []
        for param in func.params:
            if param not in func.buffer_map:
                # in case of dynamic symbolic may in params
                continue
            arg = func.buffer_map[param]
            numpy_dtype = map_numpy_type(arg.dtype)
            profile_tensors.append(
                tvm.nd.array(
                    np.random.uniform(0, 1,
                                      [var_warpper(i) for i in arg.shape]).astype(numpy_dtype),
                    device=device,
                ))
        self.profile_tensors = profile_tensors
        return profile_tensors

    def profile_latency(self, dynamic_symbolic_constrains: Optional[Dict] = None) -> str:
        if dynamic_symbolic_constrains is None:
            dynamic_symbolic_constrains = {}
        profile_tensors = self.get_profile_tensors(dynamic_symbolic_constrains)
        latency = self.time_evaluator(*profile_tensors).mean * 1e3
        return latency

    def _tensor_adapter(self, tensor, device):
        import torch
        from torch.utils.dlpack import to_dlpack

        if isinstance(tensor, tvm.te.Tensor):
            return tensor
        elif isinstance(tensor, torch.Tensor):
            return tvm.runtime.ndarray.from_dlpack(to_dlpack(tensor))
        elif isinstance(tensor, np.ndarray):
            return tvm.nd.array(tensor, device=device)
        else:
            raise RuntimeError("Not supported type: ", type(tensor))

    def _forward_from_tvm_args(self, *args):
        _tvm_args = [self._tensor_adapter(arg, self.arch.device) for arg in args]
        self.rt_mod(*_tvm_args)

    def _forward_from_tvm_nd_array(self, *args):
        self.rt_mod(*args)

    def _forward_from_torch_func(self, *args):
        # torch func is not reliable as some datatypes they don't support
        # like float8.
        self.torch_func(*args)
        return args[-1]

    def forward(self, *args):
        return self._forward_from_torch_func(*args)

    def _forward_from_prebuild_lib(self, *args, stream=0):
        ctypes_args = [
            ctypes.c_void_p(arr.data_ptr()) if not isinstance(arr, int) else arr for arr in args
        ]
        ctypes_args.append(ctypes.c_void_p(stream))
        self.lib.call(*ctypes_args)

    def call_lib(self, *args, stream=0):
        self.lib.call(*args, ctypes.c_void_p(stream))

    def _forward_from_tvm_lib_func(self, values):
        tcodes = (ctypes.c_int * self.num_args)()
        ret_val = TVMValue()
        ret_tcode = ctypes.c_int()
        for i in range(self.num_args):
            tcodes[i] = ArgTypeCode.NDARRAY_HANDLE
        if (_LIB.TVMFuncCall(
                self.function_handle,
                values,
                tcodes,
                ctypes.c_int(self.num_args),
                ctypes.byref(ret_val),
                ctypes.byref(ret_tcode),
        ) != 0):
            raise_last_ffi_error()

    def __call__(self, *args: Any) -> Any:
        return self.forward(*args)

    def update_func(self, func: PrimFunc):
        self.prim_func_mod["main"] = func

    def update_runtime_module(self, rt_mod, src_name=None, lib_name=None):
        self.rt_mod = rt_mod
        self.time_evaluator = rt_mod.time_evaluator(rt_mod.entry_name, self.arch.device, number=10)
        self.function_handle = rt_mod.get_function(rt_mod.entry_name).handle
        self.torch_func = to_pytorch_func(rt_mod)
        if src_name is not None:
            self.src_name = src_name
        if lib_name is not None:
            self.lib_name = lib_name
            self.lib = ctypes.CDLL(lib_name)
            self.lib.init()

    @abstractmethod
    def _select_implementation(self) -> IRModule:
        pass

    @property
    def prim_func(self):
        return self.prim_func_mod["main"]


class OPExecutorCPU:
    """
    A class to execute a sequence of operators on the CPU.
    """

    def __init__(self, operators: Optional[List[Operator]] = None):
        if operators is None:
            operators = []
        self.operators = operators

    def append(self, op):
        self.operators.append(op)

    def is_none(self):
        return len(self.operators) == 0

    def forward(self, weight):
        inputs = [weight]
        for op in self.operators:
            inputs.append(tvm_tensor_to_torch(op.get_profile_tensors()[-1]).cpu())
            inputs = [op.forward(*inputs)]
        return inputs[-1]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    @property
    def size(self):
        return len(self.operators)
