# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from abc import ABC, abstractmethod
from bitblas import tvm as tvm
from bitblas import tilelang as tilelang
from tvm import IRModule
from tvm.runtime.module import Module
from tvm.target import Target
from tvm.tir import PrimFunc
from tvm.contrib.dlpack import to_pytorch_func
import bitblas
import ctypes
from typing import List, Dict, Any, Optional, Tuple, Literal, Callable, Union
import numpy as np
from copy import deepcopy
from bitblas.base.base_scheduler import BaseScheduler
from bitblas.base.tuner import fast_tune, fast_tune_with_dynamic_range
from bitblas.base.arch import get_arch, TileDevice, is_cuda_arch, is_cdna_arch, is_cpu_arch
from bitblas.base.roller.hint import Hint
from bitblas.builder.wrapper import TIRWrapper, TLWrapper
from bitblas.builder.lib_generator import LibraryGenerator
from bitblas.common import MAX_ERROR_MESSAGE_LENGTH
from bitblas.utils import retrieve_func_from_module
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)

APPLY_SCHEDULE_FAILED_MESSAGE = ("Failed to apply default schedule for operator {} "
                                 "With target {} and hint {}. \n"
                                 "The error message: {} "
                                 "Please perform hardware-aware tuning manually.")

BUILD_RUNTIME_LIBRARY_FAILED_MESSAGE = ("Failed to build runtime library for operator {} "
                                        "With target {} and hint {}. \n"
                                        "The error message: '{}' \n "
                                        "Please perform hardware-aware tuning manually.")


@dataclass(frozen=True)
class OperatorConfig:
    """Base class for operator configurations. Used for typing."""

    pass


class BaseKernelNameGenerator(ABC):
    """Optional class for generating kernel names based on the config and hint"""

    def __init__(self, config: OperatorConfig):
        assert self.is_valid_config(config), (f"Invalid config for {self.__class__.__name__}: "
                                              f"{config}")
        self.config = config

    @abstractmethod
    def is_valid_config(self, config: OperatorConfig):
        pass

    @abstractmethod
    def generate(self, hint: Hint = None) -> str:
        """Generate the kernel name based on the config and hint"""
        pass

    def is_valid(self, kernel_name: str = None) -> bool:
        '''Validate kernel name after generation'''
        pattern = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
        return kernel_name.isidentifier() and pattern.match(kernel_name)


class DefaultKernelNameGenerator(BaseKernelNameGenerator):

    DEFAULT_PREFIX = "main"
    kernel_name = None

    def __init__(self, config: OperatorConfig, name: str):
        self.DEFAULT_PREFIX = name
        super().__init__(config)

    def generate(self, hint: Hint = None) -> str:
        # hint is not used
        assert hint is not None
        return self.DEFAULT_PREFIX

    def is_valid_config(self, config: OperatorConfig) -> bool:
        # config is not used
        assert config is not None
        return True


class Operator(object):

    def __init__(
        self,
        name,
        config: OperatorConfig,
        target: Target = None,
        backend: Literal["tir", "tl"] = "tir",
    ):
        if isinstance(target, str):
            target = Target(target)
        self.name = name
        self.config = config
        self.target = target
        self.backend = backend

        self.scheduled_ir_module: Optional[IRModule] = None
        self.rt_mod: Optional[Module] = None
        self.time_evaluator: Optional[Callable] = None
        self.dynamic_range: Optional[Dict] = None
        self.arch: Optional[TileDevice] = get_arch(target) if target else None

        # selector must be invoked after arch is initialized
        self.ir_module: Optional[IRModule] = (
            self._select_implementation() if self.is_tir_backend() else None)
        self.scheduler: Optional[BaseScheduler] = (
            self._select_scheduler().with_arch(self.arch) if self.is_tilelang_backend() else None)

        self.pass_context: Optional[Dict] = None

        self.kernel_name_generator: Optional[BaseKernelNameGenerator] = (
            self.get_kernel_name_generator())
        self.lib_generator = LibraryGenerator(self.arch)

        if self.is_tir_backend():
            self.wrapper = TIRWrapper(self.arch)
        elif self.is_tilelang_backend():
            self.wrapper = TLWrapper(self.arch)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        self.lib: Optional[ctypes.CDLL] = None

    def is_tir_backend(self):
        return self.backend == "tir"

    def is_tilelang_backend(self):
        return self.backend == "tl"

    def get_kernel_name_generator(self) -> Optional[BaseKernelNameGenerator]:
        return DefaultKernelNameGenerator(self.config, self.name)

    def get_source(self, target: Optional[Target] = None, kenrel_only=False) -> str:
        if target is None:
            target = self.target
        if self.lib_generator.lib_code is not None and not kenrel_only:
            return self.lib_generator.lib_code
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
        if is_cuda_arch(self.arch) or is_cdna_arch(self.arch):
            if self.scheduled_ir_module is None:
                raise ValueError(f"No optimized function available for platform {self.arch}")

            @tvm.register_func(func_name="tvm_callback_cuda_postproc", override=True)
            def tvm_callback_cuda_postproc(code, _):
                return self.post_process(code)

            @tvm.register_func(func_name="tvm_callback_hip_postproc", override=True)
            def tvm_callback_hip_postproc(code, _):
                return self.post_process(code)

            try:
                with tvm.transform.PassContext(
                        config={
                            "tir.use_async_copy": True,
                            "tir.disable_cse_tir": True,
                            **(self.pass_context if self.pass_context else {}),
                        }):
                    if self.is_tir_backend():
                        rt_mod = tvm.build(self.scheduled_ir_module, target=target)
                    elif self.is_tilelang_backend():
                        rt_mod = tilelang.lower(
                            self.scheduled_ir_module, target=target, runtime_only=True)
                    else:
                        raise ValueError(f"Unsupported backend: {self.backend}")
            except Exception as build_runtime_error:  # noqa: F841
                error_message = str(build_runtime_error)
                # Truncate only if the message exceeds the maximum length
                if len(error_message) > MAX_ERROR_MESSAGE_LENGTH:
                    truncated_message = f"{error_message[-MAX_ERROR_MESSAGE_LENGTH:]} [...]"
                else:
                    truncated_message = error_message

                logger.debug(
                    BUILD_RUNTIME_LIBRARY_FAILED_MESSAGE.format(
                        self.__class__.__name__,
                        target,
                        "optimized",
                        truncated_message,
                    ))
        else:
            # For non-CUDA and non-HIP platforms or when no optimized function is available, build with the primary function
            rt_mod = tvm.build(self.prim_func, target=target, name=self.name)

        # If the runtime module was successfully built, set up for evaluation
        if rt_mod is not None:
            self.rt_mod = rt_mod
            # Initialize a time evaluator with the built module, specifying the device and the number of runs
            self.time_evaluator = rt_mod.time_evaluator(
                rt_mod.entry_name, self.arch.device, number=10)
            self.torch_func = to_pytorch_func(rt_mod)
            if is_cuda_arch(self.arch) or is_cdna_arch(self.arch):
                is_dynamic = (
                    self.dynamic_range is not None and len(self.scheduled_ir_module.functions) > 1)
                self.wrapper.assign_optimized_module(self.scheduled_ir_module)
                wrapped_source = self.wrapper.wrap(
                    self.get_source(target, kenrel_only=True), is_dynamic)
                self.lib_generator.update_lib_code(wrapped_source)
                self.lib_generator.compile_lib(with_tl=self.is_tilelang_backend())
                self.lib = self.lib_generator.load_lib()
                self.lib.init()
            elif not is_cpu_arch(self.arch):
                raise ValueError(f"Unsupported target: {self.arch}")
        return rt_mod

    def scheduler_with_default(self, scheduler: BaseScheduler) -> Optional[IRModule]:
        scheduled_ir_module = IRModule.from_expr(scheduler.with_default_config())
        if scheduled_ir_module is not None:
            self.ir_module = scheduled_ir_module
            return scheduled_ir_module
        return None

    def apply_default_schedule(self, func_mod: IRModule, target: Target) -> IRModule:
        mod_for_opt = deepcopy(func_mod)
        with target:
            scheduled_ir_module = (
                bitblas.ApplyDefaultSchedule(  # pylint: disable=not-callable
                    bitblas.gpu.Matmul(),
                    bitblas.gpu.GEMV(),
                    bitblas.gpu.Reduction(),
                    bitblas.gpu.GeneralReduction(),
                    bitblas.gpu.Fallback(),
                )(mod_for_opt))

        if scheduled_ir_module is not None:
            return scheduled_ir_module
        return None

    def _update_optimized_mod(self, scheduled_ir_module: IRModule):
        self.scheduled_ir_module = scheduled_ir_module

    def _build_default_module(self, target: Target):
        try:
            if self.is_tir_backend():
                scheduled_mod = self.apply_default_schedule(self.ir_module, target)
            elif self.is_tilelang_backend():
                scheduled_mod = self.scheduler_with_default(self.scheduler)
            assert (
                len(scheduled_mod.get_global_vars()) == 1
            ), "The optimized module should only have one global variable for default schedule."
            global_symbol = scheduled_mod.get_global_vars()[0]
            default_kernal_name = self.kernel_name_generator.generate()
            func = scheduled_mod[global_symbol].with_attr("global_symbol", default_kernal_name)
            scheduled_ir_module = tvm.IRModule({default_kernal_name: func})
            self._update_optimized_mod(scheduled_ir_module)
        except Exception as apply_schedule_error:
            self.scheduled_ir_module = None
            logger.warning(
                APPLY_SCHEDULE_FAILED_MESSAGE.format(self.__class__.__name__, target, "default",
                                                     apply_schedule_error))

        self._build_runtime_module(target)

    def post_process(self, code: str) -> str:
        return code

    def get_tl_tuning_config(self, topk: int = 10):
        assert self.is_tilelang_backend(), "Only support tilelang backend"
        return self.scheduler.get_hardware_aware_configs(self.arch, topk)

    def apply_fast_tuning(
        self,
        func_or_scheduler: Union[PrimFunc, BaseScheduler],
        target: Target,
        topk: int = 20,
        parallel_build=True,
    ) -> Tuple[IRModule, Hint]:
        if self.is_tir_backend():
            _, best = fast_tune(func_or_scheduler, target, topk=topk, parallel_build=parallel_build)
            # annotate the best pass context
            # TODO(lei): actually we should remove this by enable pass through
            # annotation in the func's attribute.
            self.pass_context = best.config.pass_context
            return (best.sch.mod, best.config) if best is not None else (None, None)
        elif self.is_tilelang_backend():
            # Finetune the schedule
            _, best = fast_tune(
                func_or_scheduler,
                target,
                topk=topk,
                parallel_build=parallel_build,
            )
            # Return the best Config as Hint
            return (best.sch.mod, best.config) if best is not None else (None, None)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def apply_fast_tuning_with_dynamic_range(
        self,
        func_or_scheduler: Union[PrimFunc, BaseScheduler],
        target: Target,
        topk: int = 20,
        dynamic_range: Dict[str, List[int]] = None,
        parallel_build=True,
    ):
        if self.is_tir_backend() or self.is_tilelang_backend():
            scheduled_ir_module = fast_tune_with_dynamic_range(
                func_or_scheduler,
                target,
                topk=topk,
                parallel_build=parallel_build,
                dynamic_range=dynamic_range,
                kernel_name_generator=self.kernel_name_generator,
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        if scheduled_ir_module is not None:
            return scheduled_ir_module

        return None

    def hardware_aware_finetune(
        self,
        topk: int = 20,
        target: Optional[tvm.target.Target] = None,
        parallel_build=True,
    ):
        if target is None:
            target = self.target
        dynamic_range = self.dynamic_range
        if dynamic_range is not None:
            if self.is_tir_backend():
                func = self.prim_func
                self.scheduled_ir_module = self.apply_fast_tuning_with_dynamic_range(
                    func, target, topk, dynamic_range)
            elif self.is_tilelang_backend():
                scheduler = self.scheduler
                self.scheduled_ir_module = self.apply_fast_tuning_with_dynamic_range(
                    scheduler, target, topk, dynamic_range)
        else:
            func_or_scheduler = (self.prim_func if self.is_tir_backend() else self.scheduler)
            scheduled_mod, best_hint = self.apply_fast_tuning(
                func_or_scheduler, target, topk, parallel_build=parallel_build)

            if scheduled_mod is None:
                raise RuntimeError("Failed to apply fast tuning for operator {}.".format(self.name))

            assert (
                len(scheduled_mod.get_global_vars()) == 1
            ), "The optimized module should only have one global variable for default schedule."
            default_kernal_name = self.kernel_name_generator.generate(best_hint)
            func = retrieve_func_from_module(scheduled_mod).with_attr("global_symbol",
                                                                      default_kernal_name)
            scheduled_ir_module = tvm.IRModule({default_kernal_name: func})
            self._update_optimized_mod(scheduled_ir_module)

        self._build_runtime_module(self.target)

    def get_profile_tensors(self, dynamic_symbolic_constraints: Optional[Dict] = None):
        if dynamic_symbolic_constraints is None:
            dynamic_symbolic_constraints = {}
        func = self.prim_func or retrieve_func_from_module(self.scheduled_ir_module)
        device = self.arch.device

        def var_warpper(v):
            if isinstance(v, tvm.tir.Var):
                if v.name in dynamic_symbolic_constraints:
                    return dynamic_symbolic_constraints[v.name]
                assert "opt_shapes" in func.attrs
                assert v.name in func.attrs["opt_shapes"]
                if isinstance(func.attrs["opt_shapes"][v.name], tvm.tir.IntImm):
                    return func.attrs["opt_shapes"][v.name].value
                elif isinstance(func.attrs["opt_shapes"][v.name], tvm.ir.container.Array):
                    avg_shape: int = 0
                    for i in func.attrs["opt_shapes"][v.name]:
                        avg_shape += i.value
                    avg_shape = avg_shape // len(func.attrs["opt_shapes"][v.name])
                    _info_message = (
                        f"Doesn't provide dynamic symbolic constrains for {v.name} when do benchmarking, "
                        f"use average shape {avg_shape}")
                    logger.info(_info_message)
                    return avg_shape
                else:
                    raise RuntimeError("Not supported type: ",
                                       type(func.attrs["opt_shapes"][v.name]))

            elif isinstance(v, tvm.tir.IntImm):
                return v.value
            else:
                raise RuntimeError("Not supported type: ", type(v))

        def map_numpy_type(intype):
            typemap = {
                "e4m3_float8": "float8_e4m3fn",
                "e5m2_float8": "float8_e5m2",
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
        return profile_tensors

    def profile_latency(self, dynamic_symbolic_constraints: Optional[Dict] = None) -> str:
        if dynamic_symbolic_constraints is None:
            dynamic_symbolic_constraints = {}
        profile_tensors = self.get_profile_tensors(dynamic_symbolic_constraints)
        latency = self.time_evaluator(*profile_tensors).mean * 1e3
        # release the memory of profile tensors
        for tensor in profile_tensors:
            del tensor
        return latency

    def _forward_from_torch_func(self, *args):
        # Torch func is not reliable as the runtime overhead dlpack
        # is not negaliable, ref to https://discuss.tvm.apache.org/t/strange-overhead-of-tvm-runtime-ndarray-from-dlpack/16516
        self.torch_func(*args)
        return args[-1]

    def _forward_from_prebuild_lib(self, *args, stream=0):
        ctypes_args = [
            ctypes.c_void_p(arr.data_ptr()) if not isinstance(arr, int) else arr for arr in args
        ]
        ctypes_args.append(ctypes.c_void_p(stream))
        self.lib.call(*ctypes_args)

    def forward(self, *args):
        return self._forward_from_torch_func(*args)

    def __call__(self, *args: Any) -> Any:
        return self.forward(*args)

    def update_runtime_module(self, rt_mod=None, srcpath=None, libpath=None):
        if rt_mod is not None:
            self.rt_mod = rt_mod
            self.time_evaluator = rt_mod.time_evaluator(
                rt_mod.entry_name, self.arch.device, number=10)
            self.torch_func = to_pytorch_func(rt_mod)
        if srcpath is not None:
            assert self.lib_generator is not None, "lib_generator is not initialized"
            self.lib_generator.set_src_path(srcpath)
            # TODO(lei): update the lib code from srcpath
        if libpath is not None:
            assert self.lib_generator is not None, "lib_generator is not initialized"
            self.lib_generator.set_lib_path(libpath)
            self.lib = ctypes.CDLL(libpath)
            self.lib.init()

    def cleanup(self):
        raise NotImplementedError

    def check_only_tir_backend(self):
        assert self.is_tir_backend(), "Only support tir backend"

    def check_only_tilelang_backend(self):
        assert self.is_tilelang_backend(), "Only support tilelang backend"

    def _select_implementation(self) -> Optional[IRModule]:
        # only roller based template schedule
        raise NotImplementedError

    def _select_scheduler(self) -> Optional[BaseScheduler]:
        # only tilelang based template schedule
        raise NotImplementedError

    @property
    def prim_func(self) -> Optional[PrimFunc]:
        if self.ir_module is None:
            return None

        if len(self.ir_module.get_global_vars()) == 1:
            return self.ir_module[self.ir_module.get_global_vars()[0]]
        elif "main" in self.ir_module:
            return self.ir_module["main"]
        else:
            raise ValueError("Unable to determine primary function.")

    @property
    def srcpath(self):
        return self.lib_generator.get_source_path()

    @property
    def libpath(self):
        return self.lib_generator.get_lib_path()

    @property
    def wrapped_source(self):
        return self.lib_generator.lib_code


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
            inputs = [op.forward(*inputs)]
        return inputs[-1]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    @property
    def size(self):
        return len(self.operators)
