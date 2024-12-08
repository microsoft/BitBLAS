# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas.base.roller.hint import Hint
from tvm.target import Target
from .tilelang import select_scheduler as consistent_scheduler
from bitblas.base.base_scheduler import BaseScheduler
from ..operator import OperatorConfig, Operator, BaseKernelNameGenerator
from ...base.arch.cuda import CUDA
from ...utils import auto_detect_nvidia_target
from dataclasses import dataclass
from typing import Union, Tuple, Literal, Optional, Any
import logging
import torch

logger = logging.getLogger(__name__)

WORKSPACE_SIZE = 1024 * 1024 * 256


def is_native_compute(Q_dtype, K_dtype, V_dtype) -> bool:
    return Q_dtype == K_dtype and K_dtype == V_dtype


@dataclass(frozen=True)
class FlashAttenConfig(OperatorConfig):
    batch: Union[int, Tuple[int]] = None
    # TODO should distinguish from q_heads and kv_heads
    heads: Optional[int] = None
    kv_heads: Optional[int] = None
    seq_len: Optional[int] = None
    dim: Optional[int] = None
    Q_dtype: str = "float16"
    K_dtype: str = Q_dtype  # for default
    V_dtype: str = Q_dtype
    Accu_dtype: str = "float32"
    Out_dtype: str = "float16"
    layout: Literal["nnn", "ntn"] = "nnn"
    is_causal: bool = False


class FlashAttenKernelNameGenerator(BaseKernelNameGenerator):

    KERNEL_PREFIX = "flashatten"

    def is_valid_config(self, config: OperatorConfig) -> bool:
        return isinstance(config, FlashAttenConfig)

    @staticmethod
    def simplify_dtype(dtype: str) -> str:
        if dtype.startswith("float"):
            return f"f{dtype[5:]}"
        elif dtype.startswith("bfloat"):
            return f"bf{dtype[6:]}"
        elif dtype.startswith("int"):
            return f"i{dtype[3:]}"
        elif dtype.startswith("uint"):
            return f"u{dtype[4:]}"
        else:
            raise ValueError("Currently only support float, bfloat, int, uint")

    def generate(self, hint: Hint = None) -> str:
        config = self.config
        kernel_name = self.KERNEL_PREFIX
        shape_str = f"batch{self.config.batch}heads{self.config.heads}seqlen{self.config.seq_len}dim{self.config.dim}"
        Q_dtype = self.simplify_dtype(config.Q_dtype)
        K_dtype = self.simplify_dtype(config.K_dtype)
        V_dtype = self.simplify_dtype(config.V_dtype)
        Accu_dtype = self.simplify_dtype(config.Accu_dtype)
        Out_dtype = self.simplify_dtype(config.Out_dtype)
        precision_str = f"Q{Q_dtype}_K{K_dtype}_V{V_dtype}_Accu{Accu_dtype}_Out{Out_dtype}"
        kernel_name = "_".join([kernel_name, shape_str, precision_str])
        # TODO need to add hint
        assert self.is_valid(kernel_name), "Kernel name invalid"
        return kernel_name


class FlashAtten(Operator):

    BITBLAS_TRICK_DTYPE_MAP = {
        "float32": ("fp", 32),
        "float16": ("fp", 16),
        "int8": ("int", 8),
        "int4": ("int", 4),
    }

    def __init__(
        self,
        config: FlashAttenConfig,
        name: str = "flashatten",
        target: Optional[Union[str, Target]] = None,
        enable_tuning: bool = False,
        from_database: bool = False,
        backend: str = "tl",
    ):
        if target is None:
            target = auto_detect_nvidia_target()
            logger.info(f"Auto detected target: {target}")

        assert (config.Q_dtype
                in self.BITBLAS_TRICK_DTYPE_MAP), f"Unsupported input dtype {config.Q_dtype}"
        assert (config.K_dtype
                in self.BITBLAS_TRICK_DTYPE_MAP), f"Unsupported input dtype {config.K_dtype}"
        assert (config.V_dtype
                in self.BITBLAS_TRICK_DTYPE_MAP), f"Unsupported input dtype {config.V_dtype}"
        assert backend == "tl", "FlashAttention only support TL compiler"

        source_format, bit = self.BITBLAS_TRICK_DTYPE_MAP[config.Q_dtype]

        self.source_format = source_format
        self.bit = bit
        self.backend = backend
        super().__init__(name, config, target, backend)

        target = self.target
        if target.kind.name != "cuda":
            raise ValueError("Currently only support cuda target")

        self.dispatch_tl(target, from_database, source_format, enable_tuning)

    def dispatch_tl(self,
                    target: Target,
                    from_database: bool = False,
                    source_format: str = "fp16",
                    enable_tuning: bool = True):
        self.arch = CUDA(target)
        if not from_database:
            self._build_default_module(target)
        self.workspace = None
        if enable_tuning:
            self.hardware_aware_finetune()
        self.torch_output_dtype = getattr(torch, self.Out_dtype)

    def get_kernel_name_generator(self):
        return FlashAttenKernelNameGenerator(self.config)

    def _alloc_workspace(self):
        return torch.empty(WORKSPACE_SIZE, dtype=torch.float16).cuda()

    def _free_workspace(self):
        # release the workspace if it is None
        if self.workspace is not None:
            self.workspace = None

    def _select_scheduler(self) -> Optional[BaseScheduler]:
        if is_native_compute(self.Q_dtype, self.K_dtype, self.V_dtype):
            return consistent_scheduler(
                batch=self.batch,
                heads=self.heads,
                seq_len=self.seq_len,
                dim=self.dim,
                layout=self.layout,
                dtype_QKV=self.Q_dtype,
                dtype_Out=self.Out_dtype,
                dtype_Accu=self.Accu_dtype,
                is_causal=self.is_causal,
            )
        else:
            raise ValueError("Currently only support native compute for scheduler")

    def forward(self, Q, K, V, output=None) -> Any:
        args = []
        args.append(Q)
        args.append(K)
        args.append(V)
        args.append(output)
        if self.lib is None:
            self._forward_from_torch_func(*args)
        else:
            stream = torch.cuda.current_stream(device=Q.device)
            self._forward_from_prebuild_lib(*args, stream=stream.cuda_stream)

    def cleanup(self):
        self._free_workspace()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    @property
    def batch(self):
        return self.config.batch

    @property
    def heads(self):
        return self.config.heads

    @property
    def seq_len(self):
        return self.config.seq_len

    @property
    def dim(self):
        return self.config.dim

    @property
    def Q_dtype(self):
        return self.config.Q_dtype

    @property
    def K_dtype(self):
        return self.config.K_dtype

    @property
    def V_dtype(self):
        return self.config.V_dtype

    @property
    def Accu_dtype(self):
        return self.config.Accu_dtype

    @property
    def Out_dtype(self):
        return self.config.Out_dtype

    @property
    def layout(self):
        return self.config.layout

    @property
    def is_causal(self):
        return self.config.is_causal
