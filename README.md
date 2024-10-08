<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

<div align="center">

# Tile Language (tile-lang)

</div>

Tile Language (tile-lang) is an extension of the Apache tvm designed to facilitate the development of simple yet high-performance GPU kernels. The project tile-lang currently supports CUDA devices with architectures including Ampere (sm_80+), Turing (sm_75), and Volta (sm_70).

This project is co-authored by [nox-410](https://github.com/nox-410) and [chengyupku](https://github.com/chengyupku) and [LeiWang1999](https://github.com/LeiWang1999).

Let's get started with a simple GEMM example.

```python
import tilelang
import tilelang.language as T

def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype = "float"):
    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
        bias: T.Buffer([N], dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            bias_local = T.alloc_fragment((block_N,), dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(bias[bx * block_N], bias_local)
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] += bias_local[j]
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main

func = matmul(1024, 1024, 1024, 32, 32, 32)

print(func)

rt_mod, _ = tilelang.lower(func)

# CUDA Source
print(rt_mod.imported_modules[0].get_source())

```
Despite this simple examples, tilalang can be used to write more complicated examples including convolutions, flash-attention-v2 (fwd & bwd), normalizations, these examples can be found under folder tl_scripts.

The performance of our flash-attention is comparable to the manually implementation. (see [Link](./tl_doc/flash_perf.md)).

## Install

Install is similar to tvm. First, fill in USE_CUDA and USE_LLVM in cmake/config.cmake, like this:
```bash
git clone --recursive https://github.com/TileLang/tile-lang
```

Then install

```bash
./install.sh
```

Note 1: It is recommeneded to use the latest cuda toolkit, because we requires nvcc to jit compile the generated CUDA code.

Note 2: Don't forget to clone the submodules.

## Operator Example
### Flash Attention
```python
@T.prim_func
def flash_atten_v3(
    Q: T.Buffer(shape, dtype), # type: ignore
    K: T.Buffer(shape, dtype), # type: ignore
    V: T.Buffer(shape, dtype), # type: ignore
    Output: T.Buffer(shape, dtype), # type: ignore
):
    with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=thread_num) as (bx, by, bz):
        Q_shared = T.alloc_shared([block_M, dim], dtype)
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

        T.annotate_layout({Q_shared: tl.layout.make_swizzled_layout(Q_shared)})
        T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
        T.fill(acc_o, 0)
        T.fill(logsum, 0)
        T.fill(scores_max, -T.infinity(accum_dtype))
        loop_range = (
            T.ceildiv((bx + 1) * block_M, block_N) if is_casual else T.ceildiv(seq_len, block_N)
        )
        for k in T.Pipelined(loop_range, num_stages=num_stages):
            T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)
            if is_casual:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(
                        bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype)
                    )
            else:
                T.clear(acc_s)
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
            for i, j in T.Parallel(block_M, dim):
                acc_s[i, j] *= scale
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] - scores_max[i])
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.exp2(acc_s[i, j] - scores_max[i])
            T.copy(acc_s, acc_s_cast)
            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] /= logsum[i]
        T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])
```

### Dequant GEMM

```python
@T.prim_func
def dequant_matmul(
    A: T.Buffer(A_shape, in_dtype),
    B: T.Buffer(B_shape, storage_dtype),
    Ct: T.Buffer((N, M), out_dtype),
):
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        A_shared = T.alloc_shared(A_shared_shape, in_dtype)
        B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
        B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
        B_dequantize_local = T.alloc_fragment(B_dequantize_shared_shape, in_dtype)
        Ct_local = T.alloc_fragment((block_N, block_M), accum_dtype)

        T.clear(Ct_local)
        for k in T.Pipelined(
        T.ceildiv(K, block_K), 
        num_stages=num_stages
        ):
        T.copy(A[by * block_M, k * block_K], A_shared)
        T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
        T.copy(B_shared, B_local)
        for i, j in T.Parallel(block_N, block_K):
            B_dequantize_local[i, j] = _tir_packed_to_unsigned_convert("int", 8)(
            num_bits,
            B_local[i, j // 2],
            j % 2,
            dtype=in_dtype,
            )
        T.gemm(B_dequantize_local, A_shared, Ct_local, transpose_B=True)
        T.copy(Ct_local, Ct[bx * block_N, by * block_M])

```
