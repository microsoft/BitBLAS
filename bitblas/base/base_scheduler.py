# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm as tvm
from bitblas import tilelang as tilelang
from tvm import te
from tvm import IRModule
from tvm.tir import PrimFunc
from typing import Optional, Union, Callable, List, Dict
from dataclasses import dataclass, field
from tilelang.transform import Simplify
from abc import ABC, abstractmethod
from bitblas.base.arch import TileDevice, is_volta_arch, is_ampere_arch, is_cdna_arch, auto_infer_current_arch
from bitblas.base.roller.hint import Hint
from bitblas.tl.base_hint import BaseTLHint


# Decorator to simplify the output of a function
def maybe_simplify(self, func: Callable) -> Callable:

    def wrapper(*args, **kwargs):
        stmt: Union[PrimFunc, IRModule] = (func)(*args, **kwargs)
        if self._enable_simplify:
            return self.Simplify(stmt)
        return stmt

    return wrapper


@dataclass
class BaseScheduler(ABC):

    _arch: TileDevice = field(default=auto_infer_current_arch(), init=False, repr=False)

    _enable_simplify: bool = field(default=True, init=False, repr=False)

    _dynamic_range: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    @staticmethod
    def Simplify(stmt: Union[PrimFunc, IRModule]) -> Union[PrimFunc, IRModule]:
        if isinstance(stmt, PrimFunc):
            mod = Simplify()(IRModule.from_expr(stmt))
            assert len(mod.functions) == 1, "Simplify should return a single function"
            return list(mod.functions.values()).pop()
        elif isinstance(stmt, IRModule):
            return Simplify()(stmt)
        else:
            raise ValueError(f"Unsupported type: {type(stmt)}")

    def get_hardware_aware_configs(self,
                                   arch: TileDevice = None,
                                   topk: int = 10) -> List[BaseTLHint]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support hardware-aware tuning for {arch} with topk={topk}"
        )

    def activate_simplify(self) -> "BaseScheduler":
        self._enable_simplify = True
        return self

    def deactivate_simplify(self) -> "BaseScheduler":
        self._enable_simplify = False
        return self

    def maybe_simplify(self, stmt: Union[PrimFunc, IRModule]) -> Union[PrimFunc, IRModule]:
        if self._enable_simplify:
            return self.Simplify(stmt)
        return stmt

    def with_self_attrs(self, func: PrimFunc) -> PrimFunc:
        if self._dynamic_range:
            func = func.with_attr("opt_shapes", self._dynamic_range)
        return func

    def post_process(self, func: PrimFunc) -> PrimFunc:
        func = self.with_self_attrs(func)
        func = self.maybe_simplify(func)
        return func

    def set_dynamic_range(self, dynamic_range: Dict[str, int]) -> "BaseScheduler":
        self._dynamic_range = dynamic_range
        return self

    def has_dynamic_range(self) -> bool:
        return bool(self._dynamic_range)

    def with_arch(self, arch: TileDevice) -> "BaseScheduler":
        self._arch = arch
        return self

    def has_arch(self) -> bool:
        return self._arch is not None

    def is_volta_arch(self) -> bool:
        return is_volta_arch(self._arch) if self._arch is not None else False

    def is_ampere_arch(self) -> bool:
        return is_ampere_arch(self._arch) if self._arch is not None else False

    def is_cdna_arch(self) -> bool:
        return is_cdna_arch(self._arch) if self._arch is not None else False

    @staticmethod
    def maybe_dynamic(arg: Union[int, List[int]], dynamic_symbol: str = "m") -> PrimFunc:
        if isinstance(arg, int):
            return arg
        return te.var(dynamic_symbol)

    @abstractmethod
    def with_default_config(self, *args, **kwargs) -> PrimFunc:
        pass

    @abstractmethod
    def apply_config(
        self,
        *args,
        **kwargs,
    ) -> PrimFunc:
        pass

    def get_hint_type(self) -> str:
        raise NotImplementedError("Get Hint type is not implemented")

    def serialize_hints_to_configs(self, hints: List[Hint]) -> List[BaseTLHint]:
        # Convert Roller Hints to TileLang Hints
        raise NotImplementedError("Serialization of hints to configs is not implemented")

    def specialize_from_dynamic_range(self,
                                      dynamic_range: Optional[Dict[str,
                                                                   int]] = None) -> "BaseScheduler":
        raise NotImplementedError("Specialization from dynamic range is not implemented")

    @property
    def common_header(self) -> str:
        # TODO(lei): For HIP Backend it should be different
        common_header = "#include <tl_templates/cuda/common.h>\n"
        return common_header

    @property
    def global_symbol(self):
        # For kernel name generation
        return "default"

    @property
    def arch(self) -> TileDevice:
        return self._arch


# Decorator to simplify the output of a function
def simplify_prim_func(func: Callable) -> Callable:

    def wrapper(*args, **kwargs):
        stmt: Union[PrimFunc, IRModule] = (func)(*args, **kwargs)
        return BaseScheduler.Simplify(stmt)

    return wrapper
