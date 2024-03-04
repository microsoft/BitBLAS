# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tvm
from tvm.target import Target
from bitblas.base.roller.arch.cuda import CUDA
from typing import Any, List, Literal, Optional
from .operator import Operator
from .impl.matmul_dequantize_impl import select_implementation
from ..base.utils import get_rasterization_code, tensor_replace_dp4a
from bitblas.utils.tensor_adapter import tvm_tensor_to_torch
from dataclasses import dataclass
from bitblas.utils import match_global_kernel
from .ladder_permutate import LadderPermutate, LadderPermutateConfig
from .lop3_permutate import LOP3Permutate, LOP3PermutateConfig


class WeightExecutorCPU:
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


@dataclass
class MatmulWeightOnlyDequantizeConfig:
    M: int
    N: int
    K: int
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float16"
    bit: int = 4
    storage_dtype: str = "int8"
    source_format: str = "int"
    with_scaling: bool = False
    with_zeros: bool = False
    group_size: int = -1
    fast_decoding: bool = False
    with_bias: bool = False
    propagate_a: bool = False
    propagate_b: bool = False
    layout: str = "nt"
    # documents for zeros_type:
    # original: target = (dequantize_weight - zero_point) * scale
    # rescale: target = dequantize_weight * scale - zero_point
    # quantzied: target = (dequantize_weight - dequantize_zeros) * scale
    # Notice: only support "original" and "rescale" now
    # The auto-gptq framework prefer "original" for alignment with cuda.
    zeros_type: Literal["original", "rescale", "quantzied"] = "original"


class MatmulWeightOnlyDequantize(Operator):

    def __init__(
        self,
        config: MatmulWeightOnlyDequantizeConfig,
        name: str = "matmul_weight_only_dequantize",
        target: Target = tvm.target.Target("cuda"),
    ):
        super().__init__(name, target)
        self.config = config

        target = self.target
        if target.kind.name != "cuda":
            raise ValueError("Currently only support cuda target")

        self.arch = CUDA(target)

        self.prim_func_mod = self._select_implementation()
        try:
            self.optimized_func = self.apply_default_schedule(
                self.prim_func_mod, target
            )
        except Exception:
            self.optimized_func = None
            print(
                f"[BitBLAS][Warning] Apply default schedule failed, should do hardware-aware optimization manually."
            )

        if isinstance(self.M, List):
            self.dynamic_range = {"m": self.M}
            self.prim_func_mod["main"] = self.prim_func_mod["main"].with_attrs(
                {"opt_shapes": self.dynamic_range}
            )
        else:
            self.dynamic_range = None

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
                dequantize_bits=self.bit,
                storage_dtype=self.storage_dtype,
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

        if self.fast_decoding:
            lop3_permutate_config = LOP3PermutateConfig(
                M=self.N,
                N=self.K,
                datatype=self.in_dtype,
                dequantize_bits=self.bit,
                storage_dtype=self.storage_dtype,
            )
            self.lop3_permutate = LOP3Permutate(
                config=lop3_permutate_config,
                target=tvm.target.Target("llvm"),
            )
        else:
            self.lop3_permutate = None

        weight_executors = WeightExecutorCPU()
        if self.lop3_permutate is not None:
            weight_executors.append(self.lop3_permutate)

        if self.ladder_permutate_b is not None:
            weight_executors.append(self.ladder_permutate_b)

        self.weight_executors = weight_executors

    def _select_implementation(self):
        return select_implementation(
            M=self.M,
            N=self.N,
            K=self.K,
            in_dtype=self.in_dtype,
            out_dtype=self.out_dtype,
            accum_dtype=self.accum_dtype,
            bit=self.bit,
            storage_dtype=self.storage_dtype,
            source_format=self.source_format,
            with_scaling=self.with_scaling,
            with_zeros=self.with_zeros,
            group_size=self.group_size,
            fast_decoding=self.fast_decoding,
            with_bias=self.with_bias,
            layout=self.layout,
            zeros_type=self.zeros_type,
            propagate_a=self.propagate_a,
            propagate_b=self.propagate_b,
        )

    def post_process(self, code: str) -> str:
        index = code.index("{", match_global_kernel(code))
        code = tensor_replace_dp4a(code)
        # some tricky judge to decide whether to insert rasterization code
        if self.M == 1:
            return code
        if self.N * self.K > 10**6:
            rasterization_code = get_rasterization_code(10)
            code = code[: index + 2] + rasterization_code + code[index + 2 :]
        return code

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
    def bit(self):
        return self.config.bit

    @property
    def storage_dtype(self):
        return self.config.storage_dtype

    @property
    def source_format(self):
        return self.config.source_format

    @property
    def with_scaling(self):
        return self.config.with_scaling

    @property
    def with_zeros(self):
        return self.config.with_zeros

    @property
    def group_size(self):
        return self.config.group_size

    @property
    def fast_decoding(self):
        return self.config.fast_decoding

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
    def layout(self):
        return self.config.layout

    @property
    def zeros_type(self):
        return self.config.zeros_type

    @property
    def input_transform(self):
        if self.ladder_permutate_a is not None:
            return self.ladder_permutate_a
        return None

    @property
    def weight_transform(self):
        return self.weight_executors if self.weight_executors.size else None