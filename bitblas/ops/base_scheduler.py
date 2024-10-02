from tvm import IRModule
from tvm.tir import PrimFunc
from typing import Union
from dataclasses import dataclass, field
from tvm.tir.transform import Simplify
from abc import ABC, abstractmethod
from bitblas.base.arch import TileDevice


@dataclass
class BaseScheduler(ABC):

    _enable_simplify: bool = field(default=True, init=False, repr=False)

    @staticmethod
    def Simplify(stmt: Union[PrimFunc, IRModule]):
        if isinstance(stmt, PrimFunc):
            return Simplify()(IRModule.from_expr(stmt))["main"]
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
    def with_default_config(self):
        pass

    @abstractmethod
    def apply_config(
        self,
        *args,
        **kwargs,
    ):
        pass
