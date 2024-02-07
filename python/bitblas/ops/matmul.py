import tvm
from tvm.target import Target
from bitblas.base.roller.arch.arch_base import Arch
from bitblas.base.roller.arch.cuda import CUDA
from bitblas.base.utils import fast_tune, fast_tune_with_dynamic_range
from typing import List, Dict

from .operator import Operator
from .matmul_impl import matmul_impl_factory
from ..base.utils import match_global_kernel, get_rasterization_code


class Matmul(Operator):
    def __init__(
        self,
        M,
        N,
        K,
        a_dtype="float16",
        b_dtype="float16",
        c_dtype="float16",
        propagate_a=False,
        propagate_b=False,
        layout="nt",
        name="matmul",
        target: Target = tvm.target.Target("cuda"),
    ):
        super().__init__(name)

        if target.kind.name != "cuda":
            raise ValueError("Currently only support cuda target")
        self.arch = CUDA(target)
        assert propagate_a is False, "Currently only support propagate_a=False"

        self.M = M
        self.N = N
        self.K = K
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.c_dtype = c_dtype
        self.propagate_a = propagate_a
        self.propagate_b = propagate_b
        self.layout = layout
        self.prim_func_mod = self._select_impl()
        self.optimized_func = self._optimize_default(self.prim_func_mod, target)
        if isinstance(M, List):
            self.dynamic_range = {"m": M}
            self.prim_func_mod["main"] = self.prim_func_mod["main"].with_attrs(
                {"opt_shapes": self.dynamic_range}
            )
        else:
            self.dynamic_range = None
        self.target = target

    def _select_impl(self):
        _impl_key = f"matmul_{self.layout}"
        if isinstance(self.M, List):
            _impl_key += "_dyn_m"
        if self.propagate_a:
            _impl_key += "_pa"
        if self.propagate_b:
            _impl_key += "_pb"
        if isinstance(self.M, int):
            args = (self.M, self.N, self.K, self.a_dtype, self.c_dtype)
        else:
            args = (self.N, self.K, self.a_dtype, self.c_dtype)
        impl_handler = matmul_impl_factory[_impl_key]
        return impl_handler(*args)

    def optimize(self, topk: int = 20):
        dynamic_range = self.dynamic_range
        if dynamic_range is not None:
            self.optimized_func = self._optimize_fast_tune_with_dynamic_range(
                self.prim_func_mod["main"], self.target, topk, dynamic_range
            )
        else:
            self.optimized_func = self._optimize_fast_tune(
                self.prim_func_mod["main"], self.target, topk
            )

    def post_process(self, code: str) -> str:
        index = code.index("{", match_global_kernel(code))
        # some tricky judge to decide whether to insert rasterization code
        if self.N * self.K > 10**6:
            rasterization_code = get_rasterization_code(10)
            code = code[: index + 2] + rasterization_code + code[index + 2 :]
        return code

    def forward(self, a, b, c):
        adapater_a = self._tensor_adapter(a, self.arch.device)
        adapater_b = self._tensor_adapter(b, self.arch.device)
        adapater_c = self._tensor_adapter(c, self.arch.device)
        self.rt_mod(adapater_a, adapater_b, adapater_c)
        return adapater_c
