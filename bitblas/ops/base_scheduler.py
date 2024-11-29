from tvm import IRModule
from tvm.tir import PrimFunc
from typing import Union, Callable, List
from dataclasses import dataclass, field
from tvm.tl.transform import Simplify
from abc import ABC, abstractmethod
from bitblas.base.arch import TileDevice
from bitblas.base.roller.hint import Hint
from bitblas.tl.base_hint import BaseTLHint


# Decorator to simplify the output of a function
def maybe_simplify(self, func: Callable):

    def wrapper(*args, **kwargs):
        stmt: Union[PrimFunc, IRModule] = (func)(*args, **kwargs)
        if self._enable_simplify:
            return self.Simplify(stmt)
        return stmt

    return wrapper


@dataclass
class BaseScheduler(ABC):

    _enable_simplify: bool = field(default=True, init=False, repr=False)

    @staticmethod
    def Simplify(stmt: Union[PrimFunc, IRModule]):
        if isinstance(stmt, PrimFunc):
            mod = Simplify()(IRModule.from_expr(stmt))
            assert len(mod.functions) == 1, "Simplify should return a single function"
            return list(mod.functions.values()).pop()
        elif isinstance(stmt, IRModule):
            return Simplify()(stmt)
        else:
            raise ValueError(f"Unsupported type: {type(stmt)}")

    def get_hardware_aware_configs(self, arch: TileDevice = None, topk: int = 10):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support hardware-aware tuning for {arch} with topk={topk}"
        )

    def activate_simplify(self):
        self._enable_simplify = True
        return self

    def deactivate_simplify(self):
        self._enable_simplify = False
        return self

    def maybe_simplify(self, stmt: Union[PrimFunc, IRModule]):
        if self._enable_simplify:
            return self.Simplify(stmt)
        return stmt

    @abstractmethod
    def with_default_config(self) -> PrimFunc:
        pass

    @abstractmethod
    def apply_config(
        self,
        *args,
        **kwargs,
    ):
        pass

    def serialze_hints_to_configs(self, hints: List[Hint]) -> List[BaseTLHint]:
        # Convert Roller Hints to TileLang Hints
        raise NotImplementedError

    @property
    def common_header(self):
        # TODO(lei): For HIP Backend it should be different
        common_header = "#include <tl_templates/cuda/common.h>\n"
        return common_header


# Decorator to simplify the output of a function
def simplify_prim_func(func: Callable):

    def wrapper(*args, **kwargs):
        stmt: Union[PrimFunc, IRModule] = (func)(*args, **kwargs)
        return BaseScheduler.Simplify(stmt)

    return wrapper
