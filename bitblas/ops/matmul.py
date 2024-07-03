# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tvm
import numpy as np
from tvm.target import Target
from bitblas.utils.tensor_adapter import tvm_tensor_to_torch
from typing import List, Union, Optional, Any, Tuple
from .operator import Operator, TransformKind
from .impl.matmul_impl import select_implementation
from bitblas.utils import tensor_replace_dp4a, tensor_remove_make_int4, tensor_remove_make_int2
from dataclasses import dataclass
from .ladder_permutate import LadderPermutate, LadderPermutateConfig
import logging

logger = logging.getLogger(__name__)


class TransformExecutorCPU:

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


@dataclass(frozen=True)
class MatmulConfig:
    M: Union[int, Tuple[int]]
    N: int
    K: int
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float16"
    with_bias: bool = False
    # layout of matrix A and B
    # "nn": C[i, j] = A[i, k] * B[k, j]
    # "nt": C[i, j] = A[i, k] * B[j, k]
    layout: str = "nt"
    # weight transformation kind of matrix A
    propagate_a: TransformKind = TransformKind.NonTransform
    # weight transformation kind of matrix B
    propagate_b: TransformKind = TransformKind.NonTransform

    def __post_init__(self):
        # set M to tuple if it is list
        # otherwise, M is not hashable
        object.__setattr__(self, "M", tuple(self.M) if isinstance(self.M, list) else self.M)
        if isinstance(self.propagate_a, bool):
            object.__setattr__(
                self,
                "propagate_a",
                (TransformKind.IntraWarpTransform
                 if self.propagate_a else TransformKind.NonTransform),
            )
        elif isinstance(self.propagate_a, int):
            object.__setattr__(self, "propagate_a", TransformKind(self.propagate_a))

        if isinstance(self.propagate_b, bool):
            object.__setattr__(
                self,
                "propagate_b",
                (TransformKind.IntraWarpTransform
                 if self.propagate_b else TransformKind.NonTransform),
            )
        elif isinstance(self.propagate_b, int):
            object.__setattr__(self, "propagate_b", TransformKind(self.propagate_b))


class Matmul(Operator):

    def __init__(
        self,
        config: MatmulConfig,
        name: str = "matmul",
        target: Union[str, Target] = "cuda",
        enable_tuning: bool = False,
        from_database: bool = False,
    ):
        super().__init__(name, config, target)
        target = self.target
        if target.kind.name != "cuda":
            raise ValueError("Currently only support cuda target")

        if isinstance(self.M, Tuple):
            self.dynamic_range = {"m": self.M}
            self.update_func(self.prim_func.with_attrs({"opt_shapes": self.dynamic_range}))
        else:
            self.dynamic_range = None

        if not from_database:
            self._build_default_module(target)

        if self.propagate_a:
            assert (self.propagate_a is
                    TransformKind.NonTransform), "Currently only support NonTransform for input"
            ladder_permutate_config = LadderPermutateConfig(
                M=self.M,
                N=self.K,
                datatype=self.in_dtype,
                storage_dtype=self.in_dtype,
                propagate_kind="A",
                transpose_matrix=False,
                transform_kind=self.propagate_a,
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
                transpose_matrix=(self.layout == "nt"),
                transform_kind=self.propagate_b,
            )
            self.ladder_permutate_b = LadderPermutate(
                config=ladder_permutate_config,
                target=tvm.target.Target("llvm"),
            )
        else:
            self.ladder_permutate_b = None

        input_executors = TransformExecutorCPU()
        if self.ladder_permutate_a is not None:
            input_executors.append(self.ladder_permutate_b)

        self.input_executors = input_executors

        weight_executors = TransformExecutorCPU()
        if self.ladder_permutate_b is not None:
            weight_executors.append(self.ladder_permutate_b)

        self.weight_executors = weight_executors

        if enable_tuning:
            self.hardware_aware_finetune()

    def _build_default_module(self, target: Target):
        try:
            self.optimized_func = self.apply_default_schedule(self.prim_func_mod, target)
        except Exception:
            self.optimized_func = None
            logger.warning(
                "[BitBLAS][Warning] Apply default schedule failed, should do hardware-aware optimization manually."
            )

        self._build_runtime_module(target)

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
        code = tensor_replace_dp4a(code)
        code = tensor_remove_make_int4(code)
        code = tensor_remove_make_int2(code)
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
                        np.random.uniform(0, 1,
                                          [var_warpper(i, m) for i in arg.shape]).astype(arg.dtype),
                        device=device,
                    ))
            self.profile_tensors = profile_tensors
            latency = self.time_evaluator(*profile_tensors).mean * 1e3
            benchmark_latencies.append({"m": m, "latency": latency})
        # ms
        return benchmark_latencies

    def forward(self, *args) -> Any:
        if self.lib is None:
            self._forward_from_torch_func(*args)
        dynamic_symbolic = []
        if self.dynamic_range is not None:
            # assume we only have one dynamic range
            m = args[0].shape[0]
            dynamic_symbolic.append(m)
        self._forward_from_prebuild_lib(*args, *dynamic_symbolic)

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
        return self.input_executors if self.input_executors.size else None

    @property
    def weight_transform(self):
        return self.weight_executors if self.weight_executors.size else None


__all__ = ["Matmul", "MatmulConfig"]
