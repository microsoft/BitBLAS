# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas import tvm as tvm
from bitblas.base.base_scheduler import BaseScheduler
from bitblas import tilelang as tilelang
import tilelang.language as T
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.Logger(__name__)


@dataclass
class FlashAttenScheduler(BaseScheduler):
    # flashattention config
    batch: Optional[int] = None
    heads: Optional[int] = None
    seq_len: Optional[int] = None
    dim: Optional[int] = None
    trans_K: bool = False
    dtype_QKV: str = "float16"
    dtype_Out: str = "float16"
    dtype_Accu: str = "float32"
    is_causal: bool = False
    # block config
    block_M: int = 64
    block_N: int = 64
    num_stages: int = 1
    threads: int = 128
    enable_rasterization: bool = False

    def choose_pipeline(
        self,
        iterable,
        num_stages,
    ):
        enable_pipeline = num_stages > 1
        if enable_pipeline:
            return T.Pipelined(iterable, num_stages=num_stages)
        else:
            return T.serial(iterable)

    def with_default_config(self):
        block_M = getattr(self, "block_M", 64)
        block_N = getattr(self, "block_N", 64)
        num_stages = getattr(self, "num_stages", 1)
        threads = getattr(self, "threads", 128)
        enable_rasterization = getattr(self, "rasterization", False)
        return self.apply_config(
            block_M=block_M,
            block_N=block_N,
            num_stages=num_stages,
            threads=threads,
            enable_rasterization=enable_rasterization)

    def apply_config(
        self,
        block_M=64,
        block_N=64,
        num_stages=2,
        threads=128,
        enable_rasterization: bool = False,
    ):
        batch, heads, seq_len, dim = self.batch, self.heads, self.seq_len, self.dim
        trans_K = self.trans_K
        dtypeQKV, dtypeAccu, dtypeOut = self.dtype_QKV, self.dtype_Accu, self.dtype_Out
        is_causal = self.is_causal

        Q_shape = (batch, seq_len, heads, dim)
        K_shape = (batch, seq_len, heads, dim) if not trans_K else (batch, dim, heads, seq_len)
        V_shape = (batch, seq_len, heads, dim)
        Output_shape = (batch, seq_len, heads, dim)

        Q_shared_shape = (block_M, dim)
        K_shared_shape = (block_N, dim) if not trans_K else (dim, block_N)
        V_shared_shape = (block_N, dim)

        Q_local_shape = (block_M, dim)

        @T.prim_func
        def main(
                Q: T.Buffer(Q_shape, dtypeQKV),
                K: T.Buffer(K_shape, dtypeQKV),
                V: T.Buffer(V_shape, dtypeQKV),
                Output: T.Buffer(Output_shape, dtypeOut),
        ):
            scale = (1.0 / dim)**0.5 * 1.44269504
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), batch, heads, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared(Q_shared_shape, dtypeQKV)
                Q_local = T.alloc_fragment(Q_local_shape, dtypeQKV)
                K_shared = T.alloc_shared(K_shared_shape, dtypeQKV)
                V_shared = T.alloc_shared(V_shared_shape, dtypeQKV)
                score_QK = T.alloc_fragment((block_M, block_N), dtypeAccu)
                score_QK_sum = T.alloc_fragment((block_M), dtypeAccu)
                score_QK_qkvtype = T.alloc_fragment((block_M, block_N), dtypeQKV)
                score_scale = T.alloc_fragment((block_M), dtypeAccu)
                local_rowmax = T.alloc_fragment((block_M), dtypeAccu)
                prev_rowmax = T.alloc_fragment((block_M), dtypeAccu)
                global_l = T.alloc_fragment((block_M), dtypeAccu)
                block_output = T.alloc_fragment((block_M, dim), dtypeOut)

                T.use_swizzle(10, enable=enable_rasterization)

                T.copy(Q[by, bx * block_M:(bx + 1) * block_M, bz, :], Q_shared)
                T.copy(Q_shared, Q_local)
                for i, j in T.Parallel(block_M, dim):
                    Q_local[i, j] *= scale
                T.fill(block_output, 0)
                T.fill(global_l, 0)
                T.fill(local_rowmax, -T.infinity(dtypeAccu))

                for k in self.choose_pipeline(
                        T.ceildiv((bx + 1) *
                                  block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N),
                        num_stages=num_stages):
                    if trans_K:
                        T.copy(K[by, :, bz, k * block_N:(k + 1) * block_N], K_shared)
                    else:
                        T.copy(K[by, k * block_N:(k + 1) * block_N, bz, :], K_shared)
                    T.copy(V[by, k * block_N:(k + 1) * block_N, bz, :], V_shared)
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            score_QK[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0,
                                                            -T.infinity(dtypeAccu))
                    else:
                        T.fill(score_QK, 0)
                    T.gemm(
                        Q_local,
                        K_shared,
                        score_QK,
                        transpose_B=(not trans_K),
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.copy(local_rowmax, prev_rowmax)
                    T.reduce_max(score_QK, local_rowmax, dim=1, clear=False)
                    for i, j in T.Parallel(block_M, block_N):
                        score_QK[i, j] = T.exp2(score_QK[i, j] - local_rowmax[i])
                    for i in T.Parallel(block_M):
                        score_scale[i] = T.exp2(prev_rowmax[i] - local_rowmax[i])
                    for i, j in T.Parallel(block_M, dim):
                        block_output[i, j] *= score_scale[i]
                    T.copy(score_QK, score_QK_qkvtype)
                    T.gemm(
                        score_QK_qkvtype,
                        V_shared,
                        block_output,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.reduce_sum(score_QK, score_QK_sum, dim=1)
                    for i in T.Parallel(block_M):
                        global_l[i] = global_l[i] * score_scale[i] + score_QK_sum[i]
                for i, j in T.Parallel(block_M, dim):
                    block_output[i, j] /= global_l[i]
                T.copy(block_output, Output[by, bx * block_M:(bx + 1) * block_M, bz, :])

        return self.maybe_simplify(main)


def maybe_pipeline(
    iterable,
    num_stages,
):
    enable_pipeline = num_stages > 1
    if enable_pipeline:
        return T.Pipelined(iterable, num_stages=num_stages)
    else:
        return T.serial(iterable)


def flashatten_blocked(
        batch,
        seq_len,
        heads,
        dim,
        block_M_seq=64,
        block_N_seq=64,
        trans_Q=False,  # (batch, seq_len, heads, dim) for default, (batch, dim, heads, seq_len) for trans
        trans_K=False,  # (batch, seq_len, heads, dim) for default, (batch, dim, heads, seq_len) for trans
        trans_V=False,  # (batch, seq_len, heads, dim) for default, (batch, dim, heads, seq_len) for trans
        dtypeQKV="float16",
        dtypeAccu="float32",
        dtypeOut="float16",
        num_stages=2,
        threads=128,
        is_causal=False,
        enable_rasterization: bool = False,  # Enhance L2 Locality
):
    Q_shape = (batch, seq_len, heads, dim) if not trans_Q else (batch, dim, heads, seq_len)
    K_shape = (batch, seq_len, heads, dim) if not trans_K else (batch, dim, heads, seq_len)
    V_shape = (batch, seq_len, heads, dim) if not trans_V else (batch, dim, heads, seq_len)
    Output_shape = (batch, seq_len, heads, dim)

    Q_shared_shape = (block_M_seq, dim) if not trans_Q else (dim, block_M_seq)
    K_shared_shape = (block_N_seq, dim) if not trans_K else (dim, block_N_seq)
    V_shared_shape = (block_N_seq, dim) if not trans_V else (dim, block_N_seq)

    Q_local_shape = (block_M_seq, dim) if not trans_Q else (dim, block_M_seq)

    @T.prim_func
    def main(
            Q: T.Buffer(Q_shape, dtypeQKV),
            K: T.Buffer(K_shape, dtypeQKV),
            V: T.Buffer(V_shape, dtypeQKV),
            Output: T.Buffer(Output_shape, dtypeOut),
    ):
        scale = (1.0 / dim)**0.5 * 1.44269504
        with T.Kernel(
                T.ceildiv(seq_len, block_M_seq), batch, heads, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared(Q_shared_shape, dtypeQKV)
            Q_local = T.alloc_fragment(Q_local_shape, dtypeQKV)
            K_shared = T.alloc_shared(K_shared_shape, dtypeQKV)
            V_shared = T.alloc_shared(V_shared_shape, dtypeQKV)
            score_QK = T.alloc_fragment((block_M_seq, block_N_seq), dtypeAccu)
            score_QK_sum = T.alloc_fragment((block_M_seq), dtypeAccu)
            score_QK_qkvtype = T.alloc_fragment((block_M_seq, block_N_seq), dtypeQKV)
            score_scale = T.alloc_fragment((block_M_seq), dtypeAccu)
            local_rowmax = T.alloc_fragment((block_M_seq), dtypeAccu)
            prev_rowmax = T.alloc_fragment((block_M_seq), dtypeAccu)
            global_l = T.alloc_fragment((block_M_seq), dtypeAccu)
            block_output = T.alloc_fragment((block_M_seq, dim), dtypeOut)

            T.use_swizzle(10, enable=enable_rasterization)

            if trans_Q:
                T.copy(Q[by, :, bz, bx * block_M_seq:(bx + 1) * block_M_seq], Q_shared)
            else:
                T.copy(Q[by, bx * block_M_seq:(bx + 1) * block_M_seq, bz, :], Q_shared)
            T.copy(Q_shared, Q_local)
            for i, j in T.Parallel(block_M_seq, dim):
                Q_local[i, j] *= scale
            T.fill(block_output, 0)
            T.fill(global_l, 0)
            T.fill(local_rowmax, -T.infinity(dtypeAccu))

            for k in maybe_pipeline(
                    T.ceildiv(
                        (bx + 1) *
                        block_M_seq, block_N_seq) if is_causal else T.ceildiv(seq_len, block_N_seq),
                    num_stages=num_stages):
                if trans_K:
                    T.copy(K[by, :, bz, k * block_N_seq:(k + 1) * block_N_seq], K_shared)
                else:
                    T.copy(K[by, k * block_N_seq:(k + 1) * block_N_seq, bz, :], K_shared)
                if trans_V:
                    T.copy(V[by, :, bz, k * block_N_seq:(k + 1) * block_N_seq], V_shared)
                else:
                    T.copy(V[by, k * block_N_seq:(k + 1) * block_N_seq, bz, :], V_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M_seq, block_N_seq):
                        score_QK[i, j] = T.if_then_else(bx * block_M_seq + i >= k * block_N_seq + j,
                                                        0, -T.infinity(dtypeAccu))
                else:
                    T.fill(score_QK, 0)
                T.gemm(
                    Q_local,
                    K_shared,
                    score_QK,
                    transpose_A=trans_Q,
                    transpose_B=(not trans_K),
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(local_rowmax, prev_rowmax)
                T.reduce_max(score_QK, local_rowmax, dim=1, clear=False)
                for i, j in T.Parallel(block_M_seq, block_N_seq):
                    score_QK[i, j] = T.exp2(score_QK[i, j] - local_rowmax[i])
                for i in T.Parallel(block_M_seq):
                    score_scale[i] = T.exp2(prev_rowmax[i] - local_rowmax[i])
                T.reduce_sum(score_QK, score_QK_sum, dim=1)
                for i in T.Parallel(block_M_seq):
                    global_l[i] = global_l[i] * score_scale[i] + score_QK_sum[i]
                for i, j in T.Parallel(block_M_seq, dim):
                    block_output[i, j] *= score_scale[i]
                T.copy(score_QK, score_QK_qkvtype)
                T.gemm(
                    score_QK_qkvtype,
                    V_shared,
                    block_output,
                    transpose_B=trans_V,
                    policy=T.GemmWarpPolicy.FullRow,
                )
            for i, j in T.Parallel(block_M_seq, dim):
                block_output[i, j] /= global_l[i]
            T.copy(block_output, Output[by, bx * block_M_seq:(bx + 1) * block_M_seq, bz, :])

    return main
