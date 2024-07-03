# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tvm
from tvm.target import Target
from bitblas.base.roller.arch.cuda import CUDA
from typing import Any, List, Literal, Optional, Tuple, Union
from .operator import Operator, TransformKind
from .impl.matmul_dequantize_impl import select_implementation
from ..base.utils import tensor_replace_dp4a, tensor_remove_make_int4, tensor_remove_make_int2
from bitblas.utils.tensor_adapter import tvm_tensor_to_torch
from dataclasses import dataclass
from .ladder_permutate import LadderPermutate, LadderPermutateConfig
from .lop3_permutate import LOP3Permutate, LOP3PermutateConfig
import logging

logger = logging.getLogger(__name__)


class OPExecutorCPU:

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
class MatmulWeightOnlyDequantizeConfig:
    M: Union[int, Tuple[int]]
    N: int
    K: int
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float16"
    bit: int = 4
    storage_dtype: str = "int8"
    # documents for source_format:
    # the format of the source data, which can be "int", "uint", "fp", "nf"
    # "int": dequantize_weight = (target)((int)(quantize_weight - fixed_zero_point)) * scale
    #        where the fixed_zero_point is 2^(bit - 1) - 1
    # "uint": dequantize_weight = (target)((uint)(quantize_weight - zero_point)) * scale
    #        where the zero_point is manually set by zeros tensor
    # "fp": dequantize_weight = (quantize_weight - zero_point) * scale
    # "nf": dequantize_weight = (lut[quantize_weight] - zero_point) * scale
    source_format: Literal["int", "uint", "fp", "nf"] = "int"
    with_scaling: bool = False
    with_zeros: bool = False
    group_size: int = -1
    fast_decoding: bool = False
    with_bias: bool = False
    propagate_a: TransformKind = TransformKind.NonTransform
    propagate_b: TransformKind = TransformKind.NonTransform
    layout: str = "nt"
    # documents for zeros_mode:
    # original: target = (dequantize_weight - zero_point) * scale
    # rescale: target = dequantize_weight * scale - zero_point
    # quantized: target = (dequantize_weight - dequantize_zeros) * scale
    # The auto-gptq framework prefer "quantized" and "original" for alignment with cuda.
    zeros_mode: Literal["original", "rescale", "quantized"] = "original"

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


class MatmulWeightOnlyDequantize(Operator):

    def __init__(
        self,
        config: MatmulWeightOnlyDequantizeConfig,
        name: str = "matmul_weight_only_dequantize",
        target: Target = "cuda",
        enable_tuning: bool = False,
        from_database: bool = False,
    ):
        super().__init__(name, config, target)

        target = self.target
        if target.kind.name != "cuda":
            raise ValueError("Currently only support cuda target")

        self.arch = CUDA(target)

        if isinstance(self.M, Tuple):
            self.dynamic_range = {"m": self.M}
            self.prim_func_mod["main"] = self.prim_func_mod["main"].with_attrs(
                {"opt_shapes": self.dynamic_range})
        else:
            self.dynamic_range = None

        if not from_database:
            self._build_default_module(target)

        if self.propagate_a:
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
                dequantize_bits=self.bit,
                storage_dtype=self.storage_dtype,
                propagate_kind="B",
                transpose_matrix=self.layout == "nt",
                transform_kind=self.propagate_b,
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

        input_executors = OPExecutorCPU()
        if self.ladder_permutate_a is not None:
            input_executors.append(self.ladder_permutate_a)
        self.input_executors = input_executors

        weight_executors = OPExecutorCPU()
        if self.lop3_permutate is not None:
            weight_executors.append(self.lop3_permutate)

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
            bit=self.bit,
            storage_dtype=self.storage_dtype,
            source_format=self.source_format,
            with_scaling=self.with_scaling,
            with_zeros=self.with_zeros,
            group_size=self.group_size,
            fast_decoding=self.fast_decoding,
            with_bias=self.with_bias,
            layout=self.layout,
            zeros_mode=self.zeros_mode,
            propagate_a=self.propagate_a,
            propagate_b=self.propagate_b,
        )

    def post_process(self, code: str) -> str:
        code = tensor_replace_dp4a(code)
        code = tensor_remove_make_int4(code)
        code = tensor_remove_make_int2(code)
        return code

    def retrieve_weight_shape(self):
        return [int(i) for i in self.prim_func.buffer_map[self.prim_func.params[1]].shape]

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
    def zeros_mode(self):
        return self.config.zeros_mode

    @property
    def input_transform(self):
        return self.input_executors if self.input_executors.size else None

    @property
    def weight_transform(self):
        return self.weight_executors if self.weight_executors.size else None
