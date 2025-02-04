# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm as tvm
from bitblas import tilelang as tilelang
from typing import Optional, List
import tilelang.language as T
from tvm import DataType
from tvm.tir import PrimFunc

from dataclasses import dataclass
from bitblas.base.utils import get_roller_hints_from_func
from bitblas.ops.general_matmul.tirscript import (matmul_select_implementation)
from bitblas.base.arch import TileDevice
from bitblas.tl.base_hint import BaseTLHint
from bitblas.base.roller.hint import Hint
from .base import MatmulBaseParams


@dataclass
class MatmulSIMTBaseScheduler(MatmulBaseParams):
    # Base class for matrix multiplication scheduler
    # Contains the basic configuration for matrix multiplication

    def get_roller_configs(self, arch: TileDevice = None, topk: int = 10):
        layout = f"{'t' if self.trans_A else 'n'}{'t' if self.trans_B else 'n'}"

        # Simple TIR Compute Expression
        ir_module = matmul_select_implementation(
            M=self.M,
            N=self.N,
            K=self.K,
            in_dtype=self.in_dtype,
            out_dtype=self.out_dtype,
            accum_dtype=self.accum_dtype,
            layout=layout,
        )

        roller_hints = get_roller_hints_from_func(
            ir_module,
            arch,
            topk,
            tensorcore_only=False,
        )

        if roller_hints is None:
            raise ValueError("No Roller Hints Found for TensorCore Scheduling")

        return self.serialize_hints_to_configs(roller_hints)

    def get_hardware_aware_configs(self, arch: TileDevice = None, topk=10):
        if arch is None:
            arch = self.arch
        return self.get_roller_configs(arch, topk)

    # check if required shared memory cache
    def check_require_cache(self) -> bool:
        with_bias = self.with_bias

        conditions: List[bool] = []
        conditions.append(False)
        # Bias Add should be performed in shared memory
        conditions.append(with_bias)
        return any(conditions)  # Always set to False Currently


@dataclass
class MatmulFineGrainSIMTScheduler(MatmulSIMTBaseScheduler):
    # Fine-grained matrix multiplication scheduler
    # Allows for more detailed configuration.

    # SIMT Warp Configuration
    block_size_x: int = 8
    block_size_y: int = 8
    thread_row_tiles: int = 16
    thread_col_tiles: int = 16
    chunk: int = 16  # Usually determines the K-dimension split size

    class TLHint(BaseTLHint):

        hint_type: str = "MatmulFineGrainSIMTScheduler"

        def __init__(self):
            super().__init__()

        @classmethod
        def from_roller_hint(cls, hint: Hint):
            tl_hint = cls()
            for key, value in hint.__dict__.items():
                setattr(tl_hint, key, value)

            block_row_warps = hint.block[0] // (hint.thread[0] * hint.step[0])
            block_col_warps = hint.block[1] // (hint.thread[1] * hint.step[1])
            thread_row_tiles = hint.thread[0] // (hint.step[0] * 2)
            thread_col_tiles = hint.thread[1] // (hint.step[1] * 2)
            vthread_row_tiles = (hint.step[0] * 2)  # expand vtrhead to avoid load band conflict
            vthread_col_tiles = (hint.step[1] * 2)  # expand vtrhead to avoid load band conflict
            chunk = hint.rstep[0]

            tl_hint.block_size_x = block_row_warps
            tl_hint.block_size_y = block_col_warps
            tl_hint.thread_row_tiles = thread_row_tiles
            tl_hint.thread_col_tiles = thread_col_tiles
            tl_hint.vthread_row_tiles = vthread_row_tiles
            tl_hint.vthread_col_tiles = vthread_col_tiles
            tl_hint.chunk = chunk

            return tl_hint

        def get_config_params(self):
            return {
                "block_size_x": self.block_size_x,
                "block_size_y": self.block_size_y,
                "thread_row_tiles": self.thread_row_tiles,
                "thread_col_tiles": self.thread_col_tiles,
                "chunk": self.chunk,
            }

        def __repr__(self):
            return ("{"
                    f"block_size_x: {self.block_size_x}, "
                    f"block_size_y: {self.block_size_y}, "
                    f"thread_row_tiles: {self.thread_row_tiles}, "
                    f"thread_col_tiles: {self.thread_col_tiles}, "
                    f"chunk: {self.chunk}"
                    "}")

    def get_hint_type(self):
        return self.TLHint.hint_type

    def serialize_hints_to_configs(self, hints: List[Hint]):
        configs = []
        for hint in hints:
            config = self.TLHint.from_roller_hint(hint)
            configs.append(config)
        return configs

    def with_default_config(self) -> PrimFunc:
        block_size_x = getattr(self, "block_size_x", 2)
        block_size_y = getattr(self, "block_size_y", 2)
        thread_row_tiles = getattr(self, "thread_row_tiles", 16)
        thread_col_tiles = getattr(self, "thread_col_tiles", 16)
        chunk = getattr(self, "chunk", 16)

        return self.apply_config(
            block_size_x=block_size_x,
            block_size_y=block_size_y,
            thread_row_tiles=thread_row_tiles,
            thread_col_tiles=thread_col_tiles,
            chunk=chunk,
        )

    def apply_config(
        self,
        block_size_x: Optional[int] = None,
        block_size_y: Optional[int] = None,
        thread_row_tiles: Optional[int] = None,
        thread_col_tiles: Optional[int] = None,
        chunk: Optional[int] = None,
    ):
        assert block_size_x is not None, "block_size_x must be provided"
        assert block_size_y is not None, "block_size_y must be provided"
        assert thread_row_tiles is not None, "thread_row_tiles must be provided"
        assert thread_col_tiles is not None, "thread_col_tiles must be provided"
        assert chunk is not None, "chunk must be provided"

        M = self.maybe_dynamic(self.M, "m")
        N, K = self.N, self.K
        assert isinstance(N, int) and isinstance(K, int), "Do not support dynamic N and K Currently"

        in_dtype, out_dtype, accum_dtype = (
            self.in_dtype,
            self.out_dtype,
            self.accum_dtype,
        )

        with_bias = self.with_bias

        shared_scope = "shared.dyn"

        block_M = block_size_x * thread_row_tiles
        block_N = block_size_y * thread_col_tiles
        block_K = chunk

        A_shape = (M, K)
        B_shape = (N, K)
        C_shape = (M, N)
        A_shared_shape = (block_M, block_K)
        B_shared_shape = (block_N, block_K)
        Bias_shape = (N,)

        threads = thread_row_tiles * thread_col_tiles
        local_size_a = block_M // thread_row_tiles
        local_size_b = block_N // thread_col_tiles
        local_size_c = (block_M // thread_row_tiles) * (block_N // thread_col_tiles)

        micro_size_k = 128 // DataType(in_dtype).bits

        dp4a_size = 4
        use_dp4a = in_dtype == "int8" and accum_dtype == "int32"

        @T.prim_func
        def main(
                A: T.Buffer(A_shape, in_dtype),
                B: T.Buffer(B_shape, in_dtype),
                Bias: T.Buffer(Bias_shape, out_dtype),
                C: T.Buffer(C_shape, out_dtype),
        ):
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):

                A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
                B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope=shared_scope)

                A_local = T.alloc_local((local_size_a, micro_size_k), in_dtype)
                B_local = T.alloc_local((local_size_b, micro_size_k), in_dtype)
                C_local = T.alloc_local((local_size_c,), accum_dtype)

                thread_binding = T.thread_binding(threads, "threadIdx.x")

                warp_m = thread_binding % thread_row_tiles
                warp_n = thread_binding // thread_row_tiles

                T.clear(C_local)

                for ko in T.serial(K // block_K):

                    # Load A into shared memory
                    for i, k in T.Parallel(block_M, block_K):
                        A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                    # Load B into shared memory
                    for j, k in T.Parallel(block_N, block_K):
                        B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

                    for ki in T.serial((block_K // micro_size_k)):
                        for i in T.serial(local_size_a):
                            for mk in T.vectorized(micro_size_k):
                                A_local[i, mk] = A_shared[warp_m * local_size_a + i,
                                                          ki * micro_size_k + mk]

                        for i in T.serial(local_size_b):
                            for mk in T.vectorized(micro_size_k):
                                B_local[i, mk] = B_shared[warp_n * local_size_b + i,
                                                          ki * micro_size_k + mk]

                        for i, j in T.grid(local_size_a, local_size_b):
                            for mk in T.serial(micro_size_k // dp4a_size):
                                if use_dp4a:
                                    T.dp4a(
                                        A_local[i, mk * dp4a_size],
                                        B_local[j, mk * dp4a_size],
                                        C_local[i * local_size_b + j],
                                    )
                                else:
                                    for dp4a_idx in T.serial(dp4a_size):
                                        C_local[i * local_size_b + j] += (
                                            A_local[i,
                                                    mk * dp4a_size + dp4a_idx].astype(accum_dtype) *
                                            B_local[j,
                                                    mk * dp4a_size + dp4a_idx].astype(accum_dtype))

                if with_bias:
                    for i, j in T.grid(local_size_a, local_size_b):
                        C_local[i * local_size_b + j] += Bias[bx * block_N + warp_n * local_size_b +
                                                              j]

                for i in T.serial(local_size_a):
                    for j in T.vectorized(local_size_b):
                        C[
                            by * block_M + warp_m * local_size_a + i,
                            bx * block_N + warp_n * local_size_b + j,
                        ] = C_local[i * local_size_b + j]

        return self.post_process(main)

    def __post_init__(self):
        # Validate the matrix transpose settings
        assert self.trans_A is False, "Currently only support Matrix A not transposed"
        assert self.trans_B is True, "Currently only support Matrix B transposed"

        return
