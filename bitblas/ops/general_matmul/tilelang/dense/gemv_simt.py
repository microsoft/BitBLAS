# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm as tvm
from functools import reduce
from typing import Optional, List
from bitblas.base.base_scheduler import BaseScheduler
import tvm.tl.language as T
from tvm import DataType
from tvm.tir import PrimFunc

from dataclasses import dataclass
from bitblas.base.utils import get_roller_hints_from_func
from bitblas.ops.general_matmul.tirscript import (matmul_select_implementation)
from bitblas.base.arch import TileDevice
from bitblas.tl.base_hint import BaseTLHint
from bitblas.base.roller.hint import Hint
from .matmul_simt import MatmulSIMTBaseScheduler


@dataclass
class GemvFineGrainSIMTScheduler(MatmulSIMTBaseScheduler):
    # Fine-grained matrix multiplication scheduler
    # Allows for more detailed configuration.

    # Default Hint Configuration
    n_partition: int = 8
    reduce_thread: int = 16

    class TLHint(BaseTLHint):

        def __init__(self):
            super().__init__()

        @classmethod
        def from_roller_hint(cls, hint: Hint):
            tl_hint = cls()
            for key, value in hint.__dict__.items():
                setattr(tl_hint, key, value)

            def prod(iterable):
                return reduce(lambda x, y: x * y, iterable, 1)

            n_partition = int(prod(hint.thread))
            reduce_thread = int(prod(hint.reduce_thread))

            tl_hint.n_partition = n_partition
            tl_hint.reduce_thread = reduce_thread

            return tl_hint

        def get_config_params(self):
            return {
                "n_partition": self.n_partition,
                "reduce_thread": self.reduce_thread,
            }

        def __repr__(self):
            return ("{"
                    f"n_partition: {self.n_partition}, "
                    f"reduce_thread: {self.reduce_thread}, "
                    "}")

    def serialize_hints_to_configs(self, hints: List[Hint]):
        configs = []
        for hint in hints:
            config = self.TLHint.from_roller_hint(hint)
            configs.append(config)
        return configs

    def with_default_config(self) -> PrimFunc:
        n_partition = getattr(self, "n_partition", 8)
        reduce_thread = getattr(self, "reduce_thread", 16)

        return self.apply_config(
            n_partition=n_partition,
            reduce_thread=reduce_thread,
        )

    def apply_config(
        self,
        n_partition: Optional[int] = None,
        reduce_thread: Optional[int] = None,
    ):
        assert n_partition is not None, "n_partition must be provided"
        assert reduce_thread is not None, (
            "reduce_thread must be provided currently, as related bitblas.gpu.gemv.GEMV"
            "sch_outer_reduction_with_config is not implemented")

        M, N, K = self.M, self.N, self.K
        if not isinstance(M, int):
            M = tvm.te.var("m")

        in_dtype, out_dtype, accum_dtype = (
            self.in_dtype,
            self.out_dtype,
            self.accum_dtype,
        )

        vec_size = 128 // DataType(in_dtype).bits

        block_K = reduce_thread * vec_size

        A_shape = (M, K)
        B_shape = (N, K)
        C_shape = (M, N)

        dp4a_size = 4
        use_dp4a = in_dtype == "int8" and accum_dtype == "int32"

        @T.prim_func
        def main(
                A: T.Buffer(A_shape, in_dtype),
                B: T.Buffer(B_shape, in_dtype),
                C: T.Buffer(C_shape, out_dtype),
        ):
            with T.Kernel(
                    T.ceildiv(N, n_partition), M, threads=(reduce_thread, n_partition)) as (
                        bx,
                        by,
                    ):
                A_local = T.alloc_local((vec_size,), in_dtype)
                B_local = T.alloc_local((vec_size,), in_dtype)
                accum_res = T.alloc_local((1,), accum_dtype)
                reduced_accum_res = T.alloc_local((1,), accum_dtype)

                kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
                ni = T.thread_binding(0, n_partition, thread="threadIdx.y")

                T.clear(accum_res)
                for ko in T.serial(T.ceildiv(K, block_K)):
                    for v in T.vectorized(vec_size):
                        A_local[v] = A[by, ko * block_K + kr * vec_size + v]

                    for v in T.vectorized(vec_size):
                        B_local[v] = B[bx * n_partition + ni, ko * block_K + kr * vec_size + v]
                        
                    if use_dp4a:
                        for ki in T.serial(vec_size // dp4a_size):
                            T.dp4a(
                                A_local[ki * dp4a_size],
                                B_local[ki * dp4a_size],
                                accum_res[0],
                            )
                    else:
                        for ki in T.serial(vec_size):
                            accum_res[0] += A_local[ki] * B_local[ki]

                with T.attr(
                        T.comm_reducer(lambda x, y: x + y, [T.float16(0)]),
                        "reduce_scope",
                        T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1),
                            accum_res[0],
                            True,
                            reduced_accum_res[0],
                            kr,
                            dtype="handle",
                        ))
                if kr == 0:
                    C[by, bx * n_partition + ni] = reduced_accum_res[0]

        return self.post_process(main)

    def __post_init__(self):
        # Validate the matrix transpose settings
        assert self.trans_A is False, "Currently only support Matrix A not transposed"
        assert self.trans_B is True, "Currently only support Matrix B transposed"
        assert self.with_bias is False, "Currently only support without bias"
        return
