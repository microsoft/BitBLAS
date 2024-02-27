# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tvm
import numpy as np
from tvm.target import Target
from bitblas.base.roller.arch.cuda import CUDA
from typing import List, Union
from .operator import Operator
from .impl.matmul_impl import select_implementation
from ..base.utils import get_rasterization_code
from bitblas.utils import match_global_kernel
from dataclasses import dataclass
from .ladder_permutate import LadderPermutate, LadderPermutateConfig


@dataclass
class MatmulConfig:
    M: Union[int, List]
    N: int
    K: int
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float16"
    with_bias: bool = False
    layout: str = "nt"
    propagate_a: bool = False
    propagate_b: bool = False


class Matmul(Operator):
    def __init__(
        self,
        config: MatmulConfig,
        name: str = "matmul",
        target: Target = tvm.target.Target("cuda"),
    ):
        super().__init__(name)
        self.config = config

        if target.kind.name != "cuda":
            raise ValueError("Currently only support cuda target")
        self.arch = CUDA(target)
        assert self.propagate_a is False, "Currently only support propagate_a=False"

        prim_func_mod = self._select_implementation()
        self.prim_func_mod = prim_func_mod
        self.optimized_func = self.apply_default_schedule(prim_func_mod, target)

        if isinstance(self.M, List):
            self.dynamic_range = {"m": self.M}
            self.update_func(
                self.prim_func.with_attrs({"opt_shapes": self.dynamic_range})
            )
        else:
            self.dynamic_range = None
        self.target = target
        self._build_runtime_module(target)

        if self.propagate_a:
            ladder_permutate_config = LadderPermutateConfig(
                M=self.M,
                N=self.K,
                datatype=self.in_dtype,
                storage_dtype=self.in_dtype,
                propagate_kind="A",
                transpose_matrix=False,
                transform_kind=2,
            )
            self.ladder_permutate_a = LadderPermutate(
                config=ladder_permutate_config,
                target=tvm.target.Target("llvm"),
            )
        else:
            self.ladder_permutate_a = None

        if self.propagate_b:
            ladder_permutate_config = LadderPermutateConfig(
                M=self.N,
                N=self.K,
                datatype=self.in_dtype,
                storage_dtype=self.in_dtype,
                propagate_kind="B",
                transpose_matrix=True if self.layout == "nt" else False,
                transform_kind=2,
            )
            self.ladder_permutate_b = LadderPermutate(
                config=ladder_permutate_config,
                target=tvm.target.Target("llvm"),
            )
        else:
            self.ladder_permutate_b = None

    def _select_implementation(self):
        return select_implementation(
            M=self.M,
            N=self.N,
            K=self.K,
            in_dtype=self.in_dtype,
            out_dtype=self.out_dtype,
            accum_dtype=self.accum_dtype,
            with_bias=self.with_bias,
            layout=self.layout,
            propagate_a=self.propagate_a,
            propagate_b=self.propagate_b,
        )

    def post_process(self, code: str) -> str:
        index = code.index("{", match_global_kernel(code))
        # some tricky judge to decide whether to insert rasterization code
        if self.N * self.K > 10**6:
            rasterization_code = get_rasterization_code(10)
            code = code[: index + 2] + rasterization_code + code[index + 2 :]
        return code

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

    @property
    def M(self):
        return self.config.M

    @property
    def N(self):
        return self.config.N

    @property
    def K(self):
        return self.config.K

    @property
    def in_dtype(self):
        return self.config.in_dtype

    @property
    def out_dtype(self):
        return self.config.out_dtype

    @property
    def accum_dtype(self):
        return self.config.accum_dtype

    @property
    def layout(self):
        return self.config.layout

    @property
    def with_bias(self):
        return self.config.with_bias

    @property
    def propagate_a(self):
        return self.config.propagate_a

    @property
    def propagate_b(self):
        return self.config.propagate_b

    @property
    def input_transform(self):
        if self.ladder_permutate_a is not None:
            return self.ladder_permutate_a
        return None

    @property
    def weight_transform(self):
        if self.ladder_permutate_b is not None:
            return self.ladder_permutate_b
        return None


__all__ = ["Matmul", "MatmulConfig"]