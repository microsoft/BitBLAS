# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import bitblas
import bitblas.testing
from bitblas import tvm as tvm
from bitblas import tilelang as tilelang
import tilelang.language as T
from tilelang.autotuner import *
from functools import partial
import itertools
import torch
import logging
from bitblas import set_log_level
from bitblas.ops.general_flashatten.tilelang.flashatten import flashatten_blocked

set_log_level(logging.DEBUG)


def get_configs():
    block_M = [32, 64, 128]
    block_N = [32, 64, 128]
    num_stages = [1, 2]
    thread_num = [128, 256]
    _configs = list(itertools.product(block_M, block_N, num_stages, thread_num))

    configs = [{
        "block_M": c[0],
        "block_N": c[1],
        "num_stages": c[2],
        "thread_num": c[3],
    } for c in _configs]
    return configs


def ref_program(Q, K, V, causal):
    from flash_attn.flash_attn_interface import flash_attn_func

    return flash_attn_func(Q, K, V, causal=causal)


def ref_flashattn_result(batch, heads, seq_len, dim, is_casual, dtype="float16"):
    q_shape = (batch, seq_len, heads, dim)
    k_shape = (batch, seq_len, heads, dim)
    v_shape = (batch, seq_len, heads, dim)
    typemap = {"float16": torch.float16}
    Q = torch.rand(batch * seq_len * heads * dim).uniform_(-1, 1).reshape(q_shape).type(
        typemap[dtype]).cuda()
    K = torch.rand(batch * seq_len * heads * dim).uniform_(-1, 1).reshape(k_shape).type(
        typemap[dtype]).cuda()
    V = torch.rand(batch * seq_len * heads * dim).uniform_(-1, 1).reshape(v_shape).type(
        typemap[dtype]).cuda()
    res = ref_program(Q, K, V, is_casual)
    return res


def flashattn_tilelang(batch, heads, seq_len, dim, trans_K, dtypeQKV, dtypeAccu, num_stages,
                       is_causal):
    tl_prim_func = flashatten_blocked(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        dim=dim,
        trans_K=trans_K,
        dtypeQKV=dtypeQKV,
        dtypeAccu=dtypeAccu,
        num_stages=num_stages,
        is_causal=is_causal,
    )
    mod, params = tilelang.lower(tl_prim_func)
    mod = tilelang.Profiler(mod, params, [3], tilelang.TensorSupplyType.Normal)
    from flash_attn.flash_attn_interface import flash_attn_func
    # TODO Now hack to internal function get the same input, may need to modify 3rdparty:tvm.tl.utils
    ins = mod._get_inputs()
    tilelang_res = mod(*ins)
    Q, K, V = ins[0], ins[1], ins[2]
    if trans_K:
        K = K.transpose(1, 3).contiguous()
    ref_res = flash_attn_func(Q, K, V, causal=is_causal)
    torch.testing.assert_close(tilelang_res, ref_res, rtol=0.1, atol=0.1)


def test_flashattn_blocked():
    can_import_flash_attn = True
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        can_import_flash_attn = False
        print("flash_attn is not installed, skipping test")

    if can_import_flash_attn:
        flashattn_tilelang(1, 4, 256, 256, False, "float16", "float32", 1, False)
        flashattn_tilelang(1, 4, 512, 256, False, "float16", "float32", 1, False)
        flashattn_tilelang(1, 4, 512, 256, True, "float16", "float32", 1, False)


def flashattn_ref(batch, heads, seq_len, dim, is_causal):

    def kernel(block_M=64, block_N=64, num_stages=1, thread_num=128):
        scale = (1.0 / dim)**0.5 * 1.44269504
        shape = [batch, seq_len, heads, dim]
        dtype = "float16"
        accum_dtype = "float"

        @T.prim_func
        def main(
                Q: T.Buffer(shape, dtype),
                K: T.Buffer(shape, dtype),
                V: T.Buffer(shape, dtype),
                Output: T.Buffer(shape, dtype),
        ):
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), heads, batch, threads=thread_num) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                Q_local = T.alloc_fragment([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                T.annotate_layout({Q_shared: tilelang.layout.make_swizzled_layout(Q_shared)})
                T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.copy(Q_shared, Q_local)
                for i, j in T.Parallel(block_M, dim):
                    Q_local[i, j] *= scale
                loop_range = (
                    T.ceildiv(
                        (bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N))
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(
                                bx * block_M + i >= k * block_N + j,
                                0,
                                -T.infinity(acc_s.dtype),
                            )
                    else:
                        T.clear(acc_s)
                    T.gemm(
                        Q_local,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] - scores_max[i])
                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.exp2(scores_max_prev[i] - scores_max[i])
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scores_scale[i]
                    T.copy(acc_s, acc_s_cast)
                    T.gemm(
                        acc_s_cast,
                        V_shared,
                        acc_o,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])

        return main

    mod, params = tilelang.lower(kernel())
    mod = tilelang.Profiler(mod, params, [3], tilelang.TensorSupplyType.Normal)
    mod.assert_allclose(partial(ref_program, causal=is_causal), rtol=0.01, atol=0.01)


def test_flashattn_ref():
    can_import_flash_attn = True
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        can_import_flash_attn = False
        print("flash_attn is not installed, skipping test")

    if can_import_flash_attn:
        flashattn_ref(1, 8, 256, 256, False)
        flashattn_ref(1, 8, 256, 256, True)
        flashattn_ref(4, 8, 256, 256, True)


def flashattn_autotune(batch, heads, seq_len, dim, is_causal):

    @autotune(
        configs=get_configs(),
        keys=["block_M", "block_N", "num_stages", "thread_num"],
        warmup=10,
        rep=5,
    )
    @jit(
        out_idx=[3],
        supply_type=tilelang.TensorSupplyType.Normal,
        ref_prog=partial(ref_program, causal=is_causal),
        rtol=0.01,
        atol=0.01,
    )
    def kernel(block_M=None, block_N=None, num_stages=None, thread_num=None):
        scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
        shape = [batch, seq_len, heads, dim]
        dtype = "float16"
        accum_dtype = "float"

        @T.prim_func
        def main(
                Q: T.Buffer(shape, dtype),  # type: ignore
                K: T.Buffer(shape, dtype),  # type: ignore
                V: T.Buffer(shape, dtype),  # type: ignore
                Output: T.Buffer(shape, dtype),  # type: ignore
        ):
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), heads, batch, threads=thread_num) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                Q_local = T.alloc_fragment([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                T.annotate_layout({Q_shared: tilelang.layout.make_swizzled_layout(Q_shared)})
                T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.copy(Q_shared, Q_local)
                for i, j in T.Parallel(block_M, dim):
                    Q_local[i, j] *= scale
                loop_range = (
                    T.ceildiv(
                        (bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N))
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(
                                bx * block_M + i >= k * block_N + j,
                                0,
                                -T.infinity(acc_s.dtype),
                            )
                    else:
                        T.clear(acc_s)
                    T.gemm(
                        Q_local,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.exp2(scores_max_prev[i] - scores_max[i])
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scores_scale[i]
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] - scores_max[i])
                    T.copy(acc_s, acc_s_cast)
                    T.gemm(
                        acc_s_cast,
                        V_shared,
                        acc_o,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])

        return main

    return kernel()


@bitblas.testing.requires_cuda_compute_version(8, 9)
def test_flashattn_autotune():
    can_import_flash_attn = True
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        can_import_flash_attn = False
        print("flash_attn is not installed, skipping test")

    if can_import_flash_attn:
        flashattn_autotune(1, 4, 256, 256, True)
        flashattn_autotune(1, 8, 256, 256, True)
        flashattn_autotune(4, 4, 256, 256, True)
        flashattn_autotune(4, 8, 256, 256, True)


def flashattn(batch, heads, seq_len, dim, is_causal):

    def kernel(block_M=64, block_N=64, num_stages=1, thread_num=128):
        scale = (1.0 / dim)**0.5 * 1.44269504
        shape = [batch, seq_len, heads, dim]
        dtype = "float16"
        accum_dtype = "float"

        @T.prim_func
        def main(
                Q: T.Buffer(shape, dtype),
                K: T.Buffer(shape, dtype),
                V: T.Buffer(shape, dtype),
                Output: T.Buffer(shape, dtype),
        ):
            print(type(seq_len), seq_len)
            print(type(block_M), block_M)
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), heads, batch, threads=thread_num) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                Q_local = T.alloc_fragment([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                T.annotate_layout({Q_shared: tilelang.layout.make_swizzled_layout(Q_shared)})
                T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.copy(Q_shared, Q_local)
                for i, j in T.Parallel(block_M, dim):
                    Q_local[i, j] *= scale
                loop_range = (
                    T.ceildiv(
                        (bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N))
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(
                                bx * block_M + i >= k * block_N + j,
                                0,
                                -T.infinity(acc_s.dtype),
                            )
                    else:
                        T.clear(acc_s)
                    T.gemm(
                        Q_local,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] - scores_max[i])
                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.exp2(scores_max_prev[i] - scores_max[i])
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scores_scale[i]
                    T.copy(acc_s, acc_s_cast)
                    T.gemm(
                        acc_s_cast,
                        V_shared,
                        acc_o,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])

        return main

    mod, params = tilelang.lower(kernel())
    mod = tilelang.Profiler(mod, params, [3], tilelang.TensorSupplyType.Normal)
    mod.assert_allclose(partial(ref_program, causal=is_causal), rtol=0.1, atol=0.1)


@bitblas.testing.requires_cuda_compute_version(8, 9)
def test_flashattn():
    can_import_flash_attn = True
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        can_import_flash_attn = False

    if can_import_flash_attn:
        flashattn(1, 4, 256, 256, True)
        flashattn(1, 8, 256, 256, True)
        flashattn(4, 4, 256, 256, True)
        flashattn(4, 8, 256, 256, True)


if __name__ == "__main__":
    can_import_flash_attn = True
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        can_import_flash_attn = False

    if can_import_flash_attn:
        bitblas.testing.main()
