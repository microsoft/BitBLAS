# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from abc import ABC, abstractmethod
import tvm
from tvm import IRModule
from tvm.target import Target
from tvm.tir import PrimFunc
import bitblas
from typing import List, Dict, Any
import numpy as np
from ..base import fast_tune, fast_tune_with_dynamic_range
from copy import deepcopy
from bitblas.base.roller.arch import get_arch
from bitblas.utils.tensor_adapter import tvm_tensor_to_torch


class Operator(ABC):
    def __init__(self, name, target: Target = None):
        self.name = name
        self.prim_func_mod = None
        self.optimized_func = None
        self.rt_mod = None
        self.time_evaluator = None
        self.profile_tensors = None
        self.arch = get_arch(target) if target else None
        self.dynamic_range = None

    def codegen(self, target: Target = None) -> str:
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
                with tvm.transform.PassContext(config={"tir.use_async_copy": True}):
                    rt_mod = tvm.build(
                        self.optimized_func, target=target, name=self.name
                    )
            except Exception as e:
                # Log the exception for debugging purposes. Replace 'print' with logging if necessary.
                print(f"Failed to build optimized function for CUDA target")
        else:
            # For non-CUDA platforms or when no optimized function is available, build with the primary function
            rt_mod = tvm.build(self.prim_func, target=target, name=self.name)

        # If the runtime module was successfully built, set up for evaluation
        if rt_mod:
            self.rt_mod = rt_mod
            # Initialize a time evaluator with the built module, specifying the device and the number of runs
            self.time_evaluator = rt_mod.time_evaluator(
                rt_mod.entry_name, self.arch.device, number=10
            )

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
                )(mod_for_opt)
            )

        if optimized_mod is not None:
            return optimized_mod
        return None

    def post_process(self, code: str) -> str:
        return code

    def apply_fast_tuning(
        self, func: PrimFunc, target: Target, topk: int = 20, parallel_build=True
    ) -> IRModule:
        _, best = fast_tune(func, target, topk=topk, parallel_build=parallel_build)
        if best is not None:
            return best.sch.mod
        return None

    def apply_fast_tuning_with_dynamic_range(
        self,
        func: PrimFunc,
        target: Target,
        topk: int = 20,
        dynamic_range: Dict[str, List[int]] = None,
    ):
        optimized_mod = fast_tune_with_dynamic_range(
            func, target, topk=topk, parallel_build=True, dynamic_range=dynamic_range
        )
        if optimized_mod is not None:
            return optimized_mod
        return None

    def hardware_aware_finetune(
        self, topk: int = 20, target: tvm.target.Target = None, parallel_build=True
    ):
        if target is None:
            target = self.target
        dynamic_range = self.dynamic_range
        func = self.prim_func
        if dynamic_range is not None:
            self.optimized_func = self.apply_fast_tuning_with_dynamic_range(
                func, target, topk, dynamic_range
            )
        else:
            self.optimized_func = self.apply_fast_tuning(
                func, target, topk, parallel_build=parallel_build
            )
        self._build_runtime_module(self.target)

    def get_profile_tensors(self):
        func = self.prim_func
        device = self.arch.device

        def var_warpper(v):
            if isinstance(v, tvm.tir.Var):
                assert "opt_shapes" in func.attrs
                assert v.name in func.attrs["opt_shapes"]
                return func.attrs["opt_shapes"][v.name].value
            elif isinstance(v, tvm.tir.IntImm):
                return v.value
            else:
                raise RuntimeError("Not supported type: ", type(v))

        profile_tensors = []
        for param in func.params:
            if param not in func.buffer_map:
                # in case of dynamic symbolic may in params
                continue
            arg = func.buffer_map[param]
            profile_tensors.append(
                tvm.nd.array(
                    np.random.uniform(0, 1, [var_warpper(i) for i in arg.shape]).astype(
                        arg.dtype
                    ),
                    device=device,
                )
            )
        self.profile_tensors = profile_tensors
        return profile_tensors

    def profile_latency(self) -> str:
        if self.dynamic_range is not None:
            return self._profile_latency_with_dynamic_range()

        profile_tensors = self.get_profile_tensors()
        latency = self.time_evaluator(*profile_tensors).mean * 1e3
        return latency

    def _profile_latency_with_dynamic_range(self) -> List:
        raise NotImplementedError

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

    def forward_from_torch(self, *args):
        # convert tensor from torch to tvm
        _tvm_args = [self._tensor_adapter(arg, self.arch.device) for arg in args]
        self.rt_mod(*_tvm_args)
        return tvm_tensor_to_torch(_tvm_args[-1])

    def forward(self, *args):
        # "Currently only support forward from torch tensor"
        return self.forward_from_torch(*args)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def update_func(self, func: PrimFunc):
        self.prim_func_mod["main"] = func

    @abstractmethod
    def _select_implementation(self) -> IRModule:
        pass

    @property
    def prim_func(self):
        return self.prim_func_mod["main"]
