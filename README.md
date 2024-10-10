<div align="center">

# Tile Language (tile-lang)

</div>

Tile Language (**tile-lang**) is an extension of [Apache TVM](https://tvm.apache.org/) designed to facilitate the development of simple yet high-performance GPU kernels. Currently, **tile-lang** supports CUDA devices with architectures including Ampere (sm_80+), Turing (sm_75), and Volta (sm_70).

## Features

- **Simplified Syntax**: Write GPU kernels with a more straightforward and expressive syntax.
- **High Performance**: Achieve performance comparable to manually optimized implementations.
- **Advanced Operations**: Support for complex operations like convolutions, flash-attention, and normalizations.
- **Compatibility**: Works with modern CUDA architectures.

## OP Examples

- [Matrix Multiplication](#quick-start)
- [Flash Attention](#flash-attention)
- [Dequantization GEMM](#dequantization-gemm)
- [RetNet](#retina-net)
- [MAMBA](#mamba)

## Installation

We currently provide three ways to install **tile-lang**:
 - [Install from Source (using your own TVM installation)](#install-from-source-with-your-own-tvm-installation)
 - [Install from Source (using the bundled TVM submodule)](#install-from-source-with-our-tvm-submodule)
 - [Install Using the Provided Script](#install-with-provided-script)


### Method 1: Install from Source (using your own TVM installation)

If you already have a compatible TVM installation, follow these steps:

1. **Clone the Repository:**

    ```bash
    git clone --recursive https://github.com/TileLang/tile-lang
    cd tile-lang
    ```

   > **Note**: Use the `--recursive` flag to include necessary submodules.

2. **Configure Build Options:**

    Create a build directory and specify your existing TVM path:

    ```bash
    mkdir build
    cd build
    cmake .. -DTVM_PREBUILD_PATH=/your/path/to/tvm/build  # e.g., /workspace/tvm/build
    make -j 16
    ```

3. **Set Environment Variables:**

    Update `PYTHONPATH` to include the `tile-lang` Python module:

    ```bash
    export PYTHONPATH=/your/path/to/tile-lang/python:$PYTHONPATH
    # TVM_IMPORT_PYTHON_PATH is used by 3rdparty framework to import tvm
    export TVM_IMPORT_PYTHON_PATH=/your/path/to/tvm/python
    ```

### Method 2: Install from Source (using the bundled TVM submodule)

If you prefer to use the built-in TVM version, follow these instructions:

1. **Clone the Repository:**

    ```bash
    git clone --recursive https://github.com/TileLang/tile-lang
    cd tile-lang
    ```

   > **Note**: Ensure the `--recursive` flag is included to fetch submodules.

2. **Configure Build Options:**

    Copy the configuration file and enable the desired backends (e.g., LLVM and CUDA):

    ```bash
    mkdir build
    cp 3rdparty/tvm/cmake/config.cmake build
    cd build
    echo "set(USE_LLVM ON)" >> config.cmake
    echo "set(USE_CUDA ON)" >> config.cmake
    cmake ..
    make -j 16
    ```

   The build outputs (e.g., `libtilelang.so`, `libtvm.so`, `libtvm_runtime.so`) will be generated in the `build` directory.

3. **Set Environment Variables:**

    Ensure the `tile-lang` Python package is in your `PYTHONPATH`:

    ```bash
    export PYTHONPATH=/your/path/to/tile-lang/python:$PYTHONPATH
    ```

### Method 3: Install Using the Provided Script

For a simplified installation, use the provided script:

1. **Clone the Repository:**

    ```bash
    git clone --recursive https://github.com/TileLang/tile-lang
    cd tile-lang
    ```

2. **Run the Installation Script:**

    ```bash
    bash install.sh
    ```

This script automates the setup, including submodule initialization and configuration.

### Quick Start

Here's how you can get started with a simple GEMM (General Matrix Multiplication) example:

```python
import tilelang
from tilelang import Profiler
import tilelang.language as T

def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main

func = matmul(1024, 1024, 1024, 128, 128, 32)

print(func)

rt_mod, params = tilelang.lower(func)

profiler = Profiler(rt_mod, params, result_idx=[2])

import torch
a = torch.randn(1024, 1024).cuda().half()
b = torch.randn(1024, 1024).cuda().half()

c = profiler(a, b)

ref_c = a @ b

print(c)
print(ref_c)

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

# Get CUDA Source
print(rt_mod.imported_modules[0].get_source())
```

TL also provide interface for users to manupulate the memory layout, pipeline and enable rasterization for better L2 Cache Locality. Here is an example of how to use the memory layout and rasterization:

```python
import tilelang.language as T
from bitblas.tl.utils import (
    make_swizzle_layout,
)

def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            
            # Apply memory layout optimizations
            # Or you can define your own memory layout
            T.annotate_layout({
                A_shared: make_swizzle_layout(A_shared),
                B_shared: make_swizzle_layout(B_shared),
            })

            # Enable rasterization for better L2 Cache Locality
            T.use_swizzle(panel_size=10, enable=enable_rasterization)

            # Clear the local buffer
            T.clear(C_local)

            # Auto pipeline the computation
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)

                # Instead of using
                # T.copy(B[k * block_K, bx * block_N], B_shared)
                # we can also use Parallel to auto map the thread
                # bindings and vectorize the copy operation.
                for k, j in T.Parallel(block_K, block_N):
                    B_shared[k, j] = B[ko * block_K + k, bx * block_N + j]

                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main
```

Even though this is a simple example, **tile-lang** can be used to write more complex operations, including convolutions, flash-attention-v2 (forward & backward), and normalizations. These examples can be found under the `tl_scripts` folder.

The performance of our flash-attention implementation is comparable to manually optimized versions. See the [performance comparison](./tl_doc/flash_perf.md) for more details.

## Operator Examples

### Flash Attention

Below is an example of implementing Flash Attention using **tile-lang**:

```python
@T.prim_func
def flash_attention_v3(
    Q: T.Buffer(shape, dtype),
    K: T.Buffer(shape, dtype),
    V: T.Buffer(shape, dtype),
    Output: T.Buffer(shape, dtype),
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

### Dequantization GEMM

An example of implementing a dequantization GEMM:

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

## Roadmap
- [ ] **Seperate TVM Library and Tile Language**.
- [ ] **Transform BitBLAS 3rdparty tvm into tl_core branch and tilelang**.

---


TileLang has now been used in project [BitBLAS](https://github.com/microsoft/BitBLAS).

Feel free to explore the repository and contribute to the project. If you have any questions or suggestions, please open an issue or contact the authors. This project is co-authored by [nox-410](https://github.com/nox-410), [chengyupku](https://github.com/chengyupku), and [LeiWang1999](https://github.com/LeiWang1999).
