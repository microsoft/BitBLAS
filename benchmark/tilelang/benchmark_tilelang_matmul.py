import argparse
from tvm import tl
import tvm.tl.language as T
from tvm.tl.autotuner import *
import itertools


def ref_program(A, B):
    return A @ B.T


def get_configs():
    block_M = [64, 128, 256]
    block_N = [64, 128, 256]
    block_K = [64, 128, 256]
    num_stages = [0, 1, 2, 3, 4]
    thread_num = [128, 256]
    enable_rasteration = [True, False]
    _configs = list(
        itertools.product(block_M, block_N, block_K, num_stages, thread_num, enable_rasteration))

    configs = [{
        'block_M': c[0],
        'block_N': c[1],
        'block_K': c[2],
        'num_stages': c[3],
        'thread_num': c[4],
        'enable_rasteration': c[5]
    } for c in _configs]
    return configs


def matmul(M, N, K):

    @autotune(
        configs=get_configs(),
        keys=['block_M', 'block_N', 'block_K', 'num_stages', 'thread_num'],
        warmup=3,
        rep=5)
    @jit(
        out_idx=[2],
        supply_type=tl.TensorSupplyType.Integer,
        ref_prog=ref_program,
        skip_check=True,
        profiler="tvm",
        target="hip")
    def kernel(block_M=None,
               block_N=None,
               block_K=None,
               num_stages=None,
               thread_num=None,
               enable_rasteration=None):
        dtype = "float16"
        accum_dtype = "float"

        @T.prim_func
        def main(A: T.Buffer((M, K), dtype), B: T.Buffer((N, K), dtype), C: T.Buffer((M, N),
                                                                                     dtype)):
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                T.use_swizzle(panel_size=10, enable=enable_rasteration)

                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    return kernel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=8192, help='M')
    parser.add_argument('--n', type=int, default=8192, help='N')
    parser.add_argument('--k', type=int, default=8192, help='K')
    args = parser.parse_args()
    M, N, K = args.m, args.n, args.k
    total_flops = 2 * M * N * K
    best_latency, best_config, ref_latency = matmul(M, N, K)
    print(f"Best latency: {best_latency}")
    print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
    print(f"Best config: {best_config}")
    print(f"Ref TFlops: {total_flops / ref_latency * 1e-9}")
