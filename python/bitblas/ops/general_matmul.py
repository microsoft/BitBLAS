# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tvm
from tvm.target import Target
import operator
from functools import reduce
from bitblas.base.roller.arch.cuda import CUDA
from typing import Any, Literal, Optional, Tuple, Union
from .operator import Operator, TransformKind, OPExecutorCPU
from .impl.matmul_dequantize_impl import (
    select_implementation as weight_dequantize_implementation,)
from .impl.matmul_impl import select_implementation as consistent_implementation
from ..base.utils import tensor_replace_dp4a, tensor_remove_make_int4, tensor_remove_make_int2
from bitblas.utils.target_detector import auto_detect_nvidia_target
from dataclasses import dataclass
from .ladder_permutate import LadderPermutate, LadderPermutateConfig
from .lop3_permutate import LOP3Permutate, LOP3PermutateConfig
import logging
import torch

logger = logging.getLogger(__name__)

WORKSPACE_SIZE = 1024 * 1024 * 256

# TODO(lei): This should be improved into a general
# Method to get the consistent compute patterns.
NATIVE_COMPUTE_PATTERNS = [
    # A_dtype, W_dtype
    ("float64", "float64"),
    ("float32", "float32"),
    ("float16", "float16"),
    ("int8", "int8"),
    ("e4m3_float8", "e4m3_float8"),
    ("e4m3_float8", "e5m2_float8"),
    ("e5m2_float8", "e4m3_float8"),
    ("e5m2_float8", "e5m2_float8"),
]


def is_native_compute(A_dtype, W_dtype) -> bool:
    return (A_dtype, W_dtype) in NATIVE_COMPUTE_PATTERNS


@dataclass(frozen=True)
class MatmulConfig:
    M: Union[int, Tuple[int]] = None
    N: int = None
    K: int = None
    A_dtype: str = "float16"
    # is a wrapper for source_format and bit
    W_dtype: str = A_dtype  # W_dtype is the same as A_dtype by default
    out_dtype: str = "float16"
    accum_dtype: str = "float16"
    layout: Literal["nn", "nt", "tn", "tt"] = "nt"
    with_bias: bool = False
    group_size: int = -1
    with_scaling: bool = False
    with_zeros: bool = False
    # documents for zeros_mode:
    # original: target = (dequantize_weight - zero_point) * scale
    # rescale: target = dequantize_weight * scale - zero_point
    # quantized: target = (dequantize_weight - dequantize_zeros) * scale
    # The auto-gptq framework prefer "quantized" and "original" for alignment with cuda.
    zeros_mode: Literal["original", "rescale", "quantized"] = "original"
    storage_dtype: str = "int8"

    # weight transform related flags
    fast_decoding: Optional[bool] = (
        None  # enable fast decoding by default, if not specified, it is enabled by a rule.
    )
    propagate_a: Optional[TransformKind] = (
        None  # propagate_a is a flag to control the ladder permutation.
    )
    propagate_b: Optional[TransformKind] = (
        None  # propagate_b is a flag to control the ladder permutation
    )

    def __legalize_dynamic_symbolic(self, M):
        return tuple(self.M) if isinstance(self.M, list) else self.M

    def __legalize_propagate(self, propagate):
        if isinstance(propagate, bool):
            return (TransformKind.IntraWarpTransform if propagate else TransformKind.NonTransform)
        elif isinstance(propagate, int):
            return TransformKind(propagate)

        return propagate

    def __initialize_propagate(self, propagate_a: Optional[TransformKind],
                               propagate_b: Optional[TransformKind]):
        MICRO_KERNEL_SIZE = 16
        if (isinstance(self.M, int) and (self.M % MICRO_KERNEL_SIZE) == 0 and
            (self.K % MICRO_KERNEL_SIZE) == 0):
            object.__setattr__(self, "propagate_a", TransformKind.IntraWarpTransform)
        else:
            object.__setattr__(self, "propagate_a", TransformKind.NonTransform)

        if (self.M == 1 or (self.N % MICRO_KERNEL_SIZE) != 0 or (self.K % MICRO_KERNEL_SIZE) != 0 or
                isinstance(self.M, Tuple) or (self.with_zeros and self.zeros_mode == "quantized")):
            object.__setattr__(self, "propagate_a", TransformKind.NonTransform)
            object.__setattr__(self, "propagate_b", TransformKind.NonTransform)
        else:
            object.__setattr__(self, "propagate_b", TransformKind.IntraWarpTransform)

        # set a and b value if is not None
        if propagate_a is not None:
            object.__setattr__(self, "propagate_a", propagate_a)
        if propagate_b is not None:
            object.__setattr__(self, "propagate_b", propagate_b)

        # TODO(lei): This is a limitation arose by pytorch and llvm
        # Should be removed in the future.
        if self.A_dtype in ["e4m3_float8", "e5m2_float8"]:
            object.__setattr__(self, "propagate_a", TransformKind.NonTransform)
            object.__setattr__(self, "propagate_b", TransformKind.NonTransform)

    def __initialize_zeros_mode(self, zeros_mode: Optional[str]):
        if zeros_mode is None:
            object.__setattr__(self, "zeros_mode", "original")

    def __initialize_fast_decoding(self, fast_decoding: Optional[bool]):

        def is_not_fast_decoding_supported():
            conditions = []
            conditions.append("int" not in self.W_dtype)
            conditions.append(self.W_dtype == self.A_dtype)
            # int8,uint8 also do not implement and also do not require fast decoding
            conditions.append(self.W_dtype in ["int8", "uint8"])
            return any(conditions)

        if fast_decoding is not None:
            object.__setattr__(self, "fast_decoding", fast_decoding)
        elif is_not_fast_decoding_supported():
            object.__setattr__(self, "fast_decoding", False)
        else:
            object.__setattr__(self, "fast_decoding", True)

    def __post_init__(self):
        # set M to default dynamic range if it is None
        if self.M is None:
            object.__setattr__(self, "M", [1, 16, 32, 64, 128, 256, 512, 1024])
        if self.N is None:
            raise ValueError("N should be specified currently.")
        if self.K is None:
            raise ValueError("K should be specified currently.")

        # set M to tuple if it is list
        # otherwise, M is not hashable
        object.__setattr__(self, "M", self.__legalize_dynamic_symbolic(self.M))

        # set propagate_a and propagate_b to default value if it is None
        object.__setattr__(self, "propagate_a", self.__legalize_propagate(self.propagate_a))
        object.__setattr__(self, "propagate_b", self.__legalize_propagate(self.propagate_b))

        # This is hack to legalize propagate_a and b
        # TODO(lei): should be removed in the future when tc+br template is ready.
        self.__initialize_propagate(self.propagate_a, self.propagate_b)

        self.__initialize_zeros_mode(self.zeros_mode)

        self.__initialize_fast_decoding(self.fast_decoding)

        if self.with_bias is None:
            object.__setattr__(self, "with_bias", False)

        if self.group_size is None:
            object.__setattr__(self, "group_size", -1)

        if self.with_scaling is None:
            object.__setattr__(self, "with_scaling", False)

        if self.with_zeros is None:
            object.__setattr__(self, "with_zeros", False)

        if self.A_dtype == self.W_dtype and self.W_dtype in [
                "float16",
                "int8",
                "e4m3_float8",
                "e5m2_float8",
        ]:
            object.__setattr__(self, "storage_dtype", self.W_dtype)


class Matmul(Operator):

    # TODO(lei): This should be improved into a general datatype.
    BITBLAS_TRICK_DTYPE_MAP = {
        "float64": ("fp", 64),
        "float32": ("fp", 32),
        "float16": ("fp", 16),
        "int32": ("int", 32),
        "uint32": ("uint", 32),
        "int16": ("int", 16),
        "uint16": ("uint", 16),
        "int8": ("int", 8),
        "uint8": ("uint", 8),
        "int4": ("int", 4),
        "uint4": ("uint", 4),
        "int2": ("int", 2),
        "uint2": ("uint", 2),
        "int1": ("int", 1),
        "uint1": ("uint", 1),
        "nf4": ("nf", 4),
        "fp4_e2m1": ("fp", 4),
        "e4m3_float8": ("fp_e4m3", 8),  # "e4m3_float8" is a trick for "float8_e4m3fn"
        "e5m2_float8": ("fp_e5m2", 8),
    }

    def __init__(
        self,
        config: MatmulConfig,
        name: str = "matmul",
        target: Optional[Union[str, Target]] = None,
        enable_tuning: bool = True,
        from_database: bool = False,
    ):
        # if from database, we should disable default schedule
        # to save compilation time
        if target is None:
            target = auto_detect_nvidia_target()
            logger.info(f"Auto detected target: {target}")
        assert (config.A_dtype
                in self.BITBLAS_TRICK_DTYPE_MAP), f"Unsupported input dtype {config.A_dtype}"
        source_format, bit = self.BITBLAS_TRICK_DTYPE_MAP[config.W_dtype]

        self.source_format = source_format
        self.bit = bit
        super().__init__(name, config, target)

        if source_format == "int" and self.with_zeros:
            logger.warning(
                "[BitBLAS][Warning] with_zeros is not supported for int source format as int has a constant zeropoints already."
            )

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

        self.workspace = None
        if self.propagate_a:
            # for general purpose, we use propagate_a to control the ladder permutation.
            ladder_permutate_config = LadderPermutateConfig(
                M=self.M,
                N=self.K,
                datatype=self.A_dtype,
                storage_dtype=self.A_dtype,
                propagate_kind="A",
                transpose_matrix=False,
                transform_kind=self.propagate_a,
            )
            self.ladder_permutate_a = LadderPermutate(
                config=ladder_permutate_config,
                target=target,
                enable_tuning=enable_tuning,
            )
            self.workspace = torch.empty(WORKSPACE_SIZE, dtype=torch.float16).cuda()
        else:
            self.ladder_permutate_a = None

        if self.propagate_b:
            ladder_permutate_config = LadderPermutateConfig(
                M=self.N,
                N=self.K,
                datatype=self.A_dtype,
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
            assert self.source_format in ["int", "uint"]
            lop3_permutate_config = LOP3PermutateConfig(
                M=self.N,
                N=self.K,
                datatype=self.A_dtype,
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

        if source_format == "nf":
            self.lut = torch.tensor(
                [
                    -1.0,
                    -0.6961928009986877,
                    -0.5250730514526367,
                    -0.39491748809814453,
                    -0.28444138169288635,
                    -0.18477343022823334,
                    -0.09105003625154495,
                    0.0,
                    0.07958029955625534,
                    0.16093020141124725,
                    0.24611230194568634,
                    0.33791524171829224,
                    0.44070982933044434,
                    0.5626170039176941,
                    0.7229568362236023,
                    1.0,
                ],
                dtype=getattr(torch, self.A_dtype),
            ).cuda()
        else:
            self.lut = None

        # output data type
        self.torch_output_dtype = getattr(torch, self.out_dtype)

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
        if is_native_compute(self.A_dtype, self.W_dtype):
            return consistent_implementation(
                M=self.M,
                N=self.N,
                K=self.K,
                in_dtype=self.A_dtype,
                out_dtype=self.out_dtype,
                accum_dtype=self.accum_dtype,
                with_bias=self.with_bias,
                layout=self.layout,
                propagate_a=self.propagate_a,
                propagate_b=self.propagate_b,
            )
        else:
            return weight_dequantize_implementation(
                M=self.M,
                N=self.N,
                K=self.K,
                in_dtype=self.A_dtype,
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

    def transform_weight(self, weight, scale=None, zeros=None, bias=None):
        """
        Transforms the given weight tensor based on the specified quantization parameters and
        returns the transformed weight along with optional scale, zeros, and bias.

        Parameters:
        - weight: The input weight tensor to be transformed.
        - scale: Optional scaling factor for the weight tensor.
        - zeros: Optional zero-point adjustment for the weight tensor.
        - bias: Optional bias to be added to the weight tensor.

        Returns:
        A list containing the transformed weight tensor and optionally the scale, zeros, and bias.
        """
        weight = weight.contiguous()
        if self.W_dtype == self.A_dtype:
            if self.weight_transform is not None:
                return self.weight_transform(weight.cpu()).cuda().contiguous()
            return weight

        from bitblas.quantization import general_compress
        import torch
        import numpy as np

        source_format, bit = self.source_format, self.bit

        # Process integer source format
        if source_format == "int" and bit < 8:
            assert not self.with_scaling, "scale should be False for int source format"
            assert not self.with_zeros, "zeros should be False for int source format"
            maxq = 2**(bit - 1)
            # Clamp weight values to be within the quantizable range and adjust
            weight = torch.clamp(weight, -maxq, maxq).int() + maxq
        elif source_format in ["fp_e5m2", "fp_e4m3"]:
            weight = weight.view(torch.int8)
            weight = weight.int()
        else:
            # For non-integer formats, simply convert weights to integers
            weight = weight.int()

        np_storage_dtype = getattr(np, self.storage_dtype)

        weight = general_compress(
            weight.cpu().numpy(), source_bits=bit, storage_dtype=np_storage_dtype)

        weight = torch.from_numpy(weight).cuda().contiguous()

        # Apply an optional weight transformation if specified
        if self.weight_transform is not None:
            weight = self.weight_transform(weight.cpu()).cuda().contiguous()

        # Prepare the return list with the transformed weight and optionally include scale, zeros, and bias
        result = [weight]
        if scale is not None:
            result.append(scale)
        if zeros is not None:
            result.append(zeros)
        if bias is not None:
            result.append(bias)

        return next(iter(result), result)

    def transform_input(self, input_tensor):
        if self.propagate_a is not TransformKind.NonTransform:
            # check workspace size
            if input_tensor.numel() > WORKSPACE_SIZE:
                raise ValueError(
                    f"Input size {input_tensor.numel()} is larger than the workspace size {WORKSPACE_SIZE}, please increase the workspace size."
                )
            self.ladder_permutate_a._forward_from_prebuild_lib(input_tensor, self.workspace)
            return self.workspace
        return input_tensor

    def forward(self, A, W, scale=None, zeros=None, bias=None, output=None) -> Any:
        args = []
        args.append(self.transform_input(A))
        args.append(W)

        if self.lut is not None:
            args.append(self.lut)

        if output is None:
            output = torch.empty(
                A.shape[:-1] + (self.N,), dtype=self.torch_output_dtype, device=A.device)
        if scale is not None:
            args.append(scale)
        if zeros is not None:
            args.append(zeros)
        if bias is not None:
            args.append(bias)
        args.append(output)

        if self.dynamic_range is not None:
            m = reduce(operator.mul, A.shape[:-1], 1)
            args.append(m)

        stream = torch.cuda.current_stream()

        if self.lib is None:
            self._forward_from_torch_func(*args)
        self._forward_from_prebuild_lib(*args, stream=stream.cuda_stream)

        return output

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

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
    def A_dtype(self):
        return self.config.A_dtype

    @property
    def W_dtype(self):
        return self.config.W_dtype

    @property
    def out_dtype(self):
        return self.config.out_dtype

    @property
    def accum_dtype(self):
        return self.config.accum_dtype

    @property
    def storage_dtype(self):
        return self.config.storage_dtype

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
