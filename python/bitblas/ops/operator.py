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

class Operator(ABC):
    def __init__(self, name):
        self.name = name
        self.prim_func_mod = None
        self.optimized_func = None
        self.rt_mod = None
        self.time_evaluator = None
        self.profile_tensors = None
        self.arch = None

    def codegen(self, target: Target) -> str:
        if self.rt_mod is None:
            self._build_runtime_module(target)
        return (
            self.post_process(self.rt_mod.imported_modules[0].get_source())
            if self.rt_mod
            else None
        )

    def _build_runtime_module(self, target: Target):
        if self.optimized_func:
            try:
                with tvm.transform.PassContext(config={"tir.use_async_copy": True}):
                    rt_mod = tvm.build(self.optimized_func, target=target)
            except:
                rt_mod = None
        if rt_mod:
            self.rt_mod = rt_mod
            self.time_evaluator = rt_mod.time_evaluator(
                rt_mod.entry_name, self.arch.device, number=10
            )
        return rt_mod

    def _optimize_default(self, func_mod: IRModule, target: Target) -> IRModule:
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

    def _optimize_fast_tune(
        self, func: PrimFunc, target: Target, topk: int = 20
    ) -> IRModule:
        _, best = fast_tune(func, target, topk=topk, parallel_build=True)
        if best is not None:
            return best.sch.mod
        return None

    def _optimize_fast_tune_with_dynamic_range(
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

    def profile_latency(self) -> str:
        if self.dynamic_range is not None:
            return self._profile_latency_with_dynamic_range()
        func = self.prim_func_mod["main"]
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
                    np.random.uniform(
                        0, 1, [var_warpper(i) for i in arg.shape]
                    ).astype(arg.dtype),
                    device=device,
                )
            )
        self.profile_tensors = profile_tensors
        latency = self.time_evaluator(*profile_tensors).mean * 1e3
        return latency

    def _profile_latency_with_dynamic_range(self) -> List:
        func = self.prim_func_mod["main"]
        device = self.arch.device

        def var_warpper(v, m):
            if isinstance(v, tvm.tir.Var):
                assert "opt_shapes" in func.attrs
                assert v.name in func.attrs["opt_shapes"]
                return m
            elif isinstance(v, tvm.tir.IntImm):
                return v.value
            else:
                raise RuntimeError("Not supported type: ", type(v))

        benchmark_latencies = []
        for m in self.dynamic_range["m"]:
            profile_tensors = []
            for param in func.params:
                if param not in func.buffer_map:
                    # in case of dynamic symbolic may in params
                    continue
                arg = func.buffer_map[param]
                if arg.dtype == "int8":
                    profile_tensors.append(
                        tvm.nd.array(
                            np.random.randint(
                                -127, 128, [var_warpper(i, m) for i in arg.shape]
                            ).astype(arg.dtype),
                            device=device,
                        )
                    )
                else:
                    profile_tensors.append(
                        tvm.nd.array(
                            np.random.uniform(
                                0, 1, [var_warpper(i, m) for i in arg.shape]
                            ).astype(arg.dtype),
                            device=device,
                        )
                    )
            self.profile_tensors = profile_tensors
            latency = self.time_evaluator(*profile_tensors).mean * 1e3
            benchmark_latencies.append({"m": m, "latency": latency})
        # ms
        return benchmark_latencies

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

    def _tensor_to_torch(self, tensor):
        import torch
        from torch.utils.dlpack import from_dlpack

        if isinstance(tensor, tvm.te.Tensor):
            return torch.from_numpy(tensor.numpy())
        elif isinstance(tensor, tvm.nd.NDArray):
            return from_dlpack(tensor)
        else:
            raise RuntimeError("Not supported type: ", type(tensor))

    def forward_from_torch(self, *args):
        # convert tensor from torch to tvm
        _tvm_args = [self._tensor_adapter(arg, self.arch.device) for arg in args]
        self.rt_mod(*_tvm_args)
        return self._tensor_to_torch(_tvm_args[-1])

    def forward(self, *args):
        return self.forward_from_torch(*args)
        # inp = args[0]
        # if isinstance(inp, torch.Tensor):
        #     return self.forward_from_torch(*args)
        # else:
        #     raise ValueError("Currently only support forward from torch tensor")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
