# Quick Start

We provide two primary ways to do the code generation: using a high-level DSL (TensorIR Script), or using packed Operators. We recommend using the packed Operators for most users, as it is more user-friendly and has better performance. This document will guide you on how to use BitBLAS directly from packed operators in Python, and provide two examples of a module wrapper in PyTorch (Dense Linear and QuantLinear).

## Brief Introduction of Packed Operators

We have packaged common operators with configurations in `bitblas/ops/impl`. You can use these operators directly with Python and Pytorch. For instance, if you want to use the Matmul operator, you can do it as follows:

```python
from bitblas.ops.matmul import Matmul, MatmulConfig
from bitblas.utils import auto_detect_nvidia_target

target = auto_detect_nvidia_target()
# or use brief target = "cuda"
# or use detailed target = "nvidia/nvidia-a100" 
# to let the tuner aware more hardware information

matmul_config = MatmulConfig(
    M=1024,
    N=1024,
    K=1024,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    layout="nt",
)
matmul = Matmul(
    config=matmul_config,
    target=target,
)
```

More details of the configuration can be found in the [Operator Configuration Documentation](OperatorConfig.md).

By default, BitBLAS will apply a default schedule into the operator (the performance is not optimal), you can also get code generation result by calling matmul.get_source().

```python
print(matmul.get_source())
```

The code generation result will be printed out. Here is an example of the code generation result:

<details>
<summary> The code output of `matmul.get_source()` </summary>

```cpp
extern "C" __global__ void __launch_bounds__(128) main_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C) {
  extern __shared__ uchar buf_dyn_shmem[];
  half C_reindex_shared_dyn_warp[128];
  half A_reindex_shared_dyn_warp[32];
  half B_reindex_shared_dyn_warp[32];
  half A_reindex_shared_dyn_warp_1[32];
  half B_reindex_shared_dyn_warp_1[32];
  for (int ax1_0_3_init = 0; ax1_0_3_init < 4; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 4; ++ax2_0_3_init) {
      for (int i = 0; i < 8; ++i) {
C_reindex_shared_dyn_warp[((ax1_0_3_init * 32) + (ax2_0_3_init * 8)) + i] = 0.0;}
;
    }
  }
  for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0) {

  {
        unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(buf_dyn_shmem + ((((((ax0_ax1_fused_0 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 16)) + 32768))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(buf_dyn_shmem + ((((((ax0_ax1_fused_0 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 16)) + 32768)))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + (((((((((int)blockIdx.x) >> 3) * 131072) + (ax0_ax1_fused_0 * 32768)) + (((int)threadIdx.z) * 16384)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8)))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 4; ++ax0_ax1_fused_0_1) {

  {
        unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(buf_dyn_shmem + (((((ax0_ax1_fused_0_1 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 16)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(buf_dyn_shmem + (((((ax0_ax1_fused_0_1 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 16))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + (((((((((int)blockIdx.x) & 7) * 131072) + (ax0_ax1_fused_0_1 * 32768)) + (((int)threadIdx.z) * 16384)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8)))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int ax0_ax1_fused_0_2 = 0; ax0_ax1_fused_0_2 < 4; ++ax0_ax1_fused_0_2) {

  {
        unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(buf_dyn_shmem + ((((((ax0_ax1_fused_0_2 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 16)) + 40960))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(buf_dyn_shmem + ((((((ax0_ax1_fused_0_2 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 16)) + 40960)))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 3) * 131072) + (ax0_ax1_fused_0_2 * 32768)) + (((int)threadIdx.z) * 16384)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8)) + 32))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_3 = 0; ax0_ax1_fused_0_3 < 4; ++ax0_ax1_fused_0_3) {

  {
        unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(buf_dyn_shmem + ((((((ax0_ax1_fused_0_3 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 16)) + 8192))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(buf_dyn_shmem + ((((((ax0_ax1_fused_0_3 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 16)) + 8192)))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 7) * 131072) + (ax0_ax1_fused_0_3 * 32768)) + (((int)threadIdx.z) * 16384)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8)) + 32))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int ax0_ax1_fused_0_4 = 0; ax0_ax1_fused_0_4 < 4; ++ax0_ax1_fused_0_4) {

  {
        unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(buf_dyn_shmem + ((((((ax0_ax1_fused_0_4 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 16)) + 49152))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(buf_dyn_shmem + ((((((ax0_ax1_fused_0_4 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 16)) + 49152)))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.x) >> 3) * 131072) + (ax0_ax1_fused_0_4 * 32768)) + (((int)threadIdx.z) * 16384)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8)) + 64))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_fused_0_5 = 0; ax0_ax1_fused_0_5 < 4; ++ax0_ax1_fused_0_5) {

  {
        unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(buf_dyn_shmem + ((((((ax0_ax1_fused_0_5 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 16)) + 16384))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(buf_dyn_shmem + ((((((ax0_ax1_fused_0_5 * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 16)) + 16384)))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((((int)blockIdx.x) & 7) * 131072) + (ax0_ax1_fused_0_5 * 32768)) + (((int)threadIdx.z) * 16384)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8)) + 64))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int ax3_0_0 = 0; ax3_0_0 < 29; ++ax3_0_0) {
    __syncthreads();
    for (int ax0_ax1_fused_0_6 = 0; ax0_ax1_fused_0_6 < 4; ++ax0_ax1_fused_0_6) {

  {
        unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(buf_dyn_shmem + (((((((((ax3_0_0 + 3) & 3) * 8192) + (ax0_ax1_fused_0_6 * 2048)) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 16)) + 32768))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(buf_dyn_shmem + (((((((((ax3_0_0 + 3) & 3) * 8192) + (ax0_ax1_fused_0_6 * 2048)) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 16)) + 32768)))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + (((((((((((int)blockIdx.x) >> 3) * 131072) + (ax0_ax1_fused_0_6 * 32768)) + (((int)threadIdx.z) * 16384)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 2) * 1024)) + (ax3_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 96))), "n"(16)
    );
  }
    }
    for (int ax0_ax1_fused_0_7 = 0; ax0_ax1_fused_0_7 < 4; ++ax0_ax1_fused_0_7) {

  {
        unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(buf_dyn_shmem + ((((((((ax3_0_0 + 3) & 3) * 8192) + (ax0_ax1_fused_0_7 * 2048)) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 16)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(buf_dyn_shmem + ((((((((ax3_0_0 + 3) & 3) * 8192) + (ax0_ax1_fused_0_7 * 2048)) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 16))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + (((((((((((int)blockIdx.x) & 7) * 131072) + (ax0_ax1_fused_0_7 * 32768)) + (((int)threadIdx.z) * 16384)) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 2) * 1024)) + (ax3_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 96))), "n"(16)
    );
  }
    }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 3;");

    __syncthreads();
    for (int ax3_0_1 = 0; ax3_0_1 < 2; ++ax3_0_1) {
      for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(((half*)buf_dyn_shmem)[(((((((ax3_0_0 & 3) * 4096) + (((int)threadIdx.z) * 2048)) + (ax0_0 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + ((((ax3_0_1 * 2) + (((int)threadIdx.x) >> 4)) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)) + 16384)])) + 0)));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(((half*)buf_dyn_shmem)[(((((((ax3_0_0 & 3) * 4096) + (((int)threadIdx.z) * 2048)) + (ax0_0 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + ((((ax3_0_1 * 2) + (((int)threadIdx.x) >> 4)) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)) + 16384)])) + 0))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax0_0 * 8)))[3])
      : "r"(addr)
    );
  }
      }
      for (int ax0_0_1 = 0; ax0_0_1 < 4; ++ax0_0_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(((half*)buf_dyn_shmem)[(((((((ax3_0_0 & 3) * 4096) + (((int)threadIdx.y) * 2048)) + (ax0_0_1 * 512)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + ((((ax3_0_1 * 2) + ((((int)threadIdx.x) & 15) >> 3)) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8))])) + 0)));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(((half*)buf_dyn_shmem)[(((((((ax3_0_0 & 3) * 4096) + (((int)threadIdx.y) * 2048)) + (ax0_0_1 * 512)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + ((((ax3_0_1 * 2) + ((((int)threadIdx.x) & 15) >> 3)) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8))])) + 0))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_reindex_shared_dyn_warp + (ax0_0_1 * 8)))[0]), "=r"(((unsigned *)(B_reindex_shared_dyn_warp + (ax0_0_1 * 8)))[1]), "=r"(((unsigned *)(B_reindex_shared_dyn_warp + (ax0_0_1 * 8)))[2]), "=r"(((unsigned *)(B_reindex_shared_dyn_warp + (ax0_0_1 * 8)))[3])
      : "r"(addr)
    );
  }
      }
      for (int ax1_0_3 = 0; ax1_0_3 < 4; ++ax1_0_3) {
        for (int ax2_0_3 = 0; ax2_0_3 < 4; ++ax2_0_3) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(C_reindex_shared_dyn_warp + ((ax1_0_3 * 32) + (ax2_0_3 * 8))))[0]), "=r"(((unsigned *)(C_reindex_shared_dyn_warp + ((ax1_0_3 * 32) + (ax2_0_3 * 8))))[1])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3 * 8)))[3]), "r"(((unsigned *)(B_reindex_shared_dyn_warp + (ax2_0_3 * 8)))[0]), "r"(((unsigned *)(B_reindex_shared_dyn_warp + (ax2_0_3 * 8)))[1]), "r"(((unsigned *)(C_reindex_shared_dyn_warp + ((ax1_0_3 * 32) + (ax2_0_3 * 8))))[0]), "r"(((unsigned *)(C_reindex_shared_dyn_warp + ((ax1_0_3 * 32) + (ax2_0_3 * 8))))[1]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(C_reindex_shared_dyn_warp + (((ax1_0_3 * 32) + (ax2_0_3 * 8)) + 4)))[0]), "=r"(((unsigned *)(C_reindex_shared_dyn_warp + (((ax1_0_3 * 32) + (ax2_0_3 * 8)) + 4)))[1])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp + (ax1_0_3 * 8)))[3]), "r"(((unsigned *)(B_reindex_shared_dyn_warp + ((ax2_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(B_reindex_shared_dyn_warp + ((ax2_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(C_reindex_shared_dyn_warp + (((ax1_0_3 * 32) + (ax2_0_3 * 8)) + 4)))[0]), "r"(((unsigned *)(C_reindex_shared_dyn_warp + (((ax1_0_3 * 32) + (ax2_0_3 * 8)) + 4)))[1]));
  }
        }
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 2;");

  __syncthreads();
  for (int ax3_0_1_1 = 0; ax3_0_1_1 < 2; ++ax3_0_1_1) {
    for (int ax0_0_2 = 0; ax0_0_2 < 4; ++ax0_0_2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(((half*)buf_dyn_shmem)[(((((((int)threadIdx.z) * 2048) + (ax0_0_2 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + ((((ax3_0_1_1 * 2) + (((int)threadIdx.x) >> 4)) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)) + 20480)])) + 0)));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(((half*)buf_dyn_shmem)[(((((((int)threadIdx.z) * 2048) + (ax0_0_2 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + ((((ax3_0_1_1 * 2) + (((int)threadIdx.x) >> 4)) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)) + 20480)])) + 0))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_2 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_2 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_2 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_2 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_3 = 0; ax0_0_3 < 4; ++ax0_0_3) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(((half*)buf_dyn_shmem)[((((((((int)threadIdx.y) * 2048) + (ax0_0_3 * 512)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + ((((ax3_0_1_1 * 2) + ((((int)threadIdx.x) & 15) >> 3)) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)) + 4096)])) + 0)));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(((half*)buf_dyn_shmem)[((((((((int)threadIdx.y) * 2048) + (ax0_0_3 * 512)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + ((((ax3_0_1_1 * 2) + ((((int)threadIdx.x) & 15) >> 3)) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)) + 4096)])) + 0))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + (ax0_0_3 * 8)))[0]), "=r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + (ax0_0_3 * 8)))[1]), "=r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + (ax0_0_3 * 8)))[2]), "=r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + (ax0_0_3 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_1 = 0; ax1_0_3_1 < 4; ++ax1_0_3_1) {
      for (int ax2_0_3_1 = 0; ax2_0_3_1 < 4; ++ax2_0_3_1) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(C_reindex_shared_dyn_warp + ((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8))))[0]), "=r"(((unsigned *)(C_reindex_shared_dyn_warp + ((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8))))[1])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_1 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_1 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_1 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_1 * 8)))[3]), "r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + (ax2_0_3_1 * 8)))[0]), "r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + (ax2_0_3_1 * 8)))[1]), "r"(((unsigned *)(C_reindex_shared_dyn_warp + ((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8))))[0]), "r"(((unsigned *)(C_reindex_shared_dyn_warp + ((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8))))[1]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(C_reindex_shared_dyn_warp + (((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8)) + 4)))[0]), "=r"(((unsigned *)(C_reindex_shared_dyn_warp + (((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8)) + 4)))[1])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_1 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_1 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_1 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_1 * 8)))[3]), "r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + ((ax2_0_3_1 * 8) + 4)))[0]), "r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + ((ax2_0_3_1 * 8) + 4)))[1]), "r"(((unsigned *)(C_reindex_shared_dyn_warp + (((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8)) + 4)))[0]), "r"(((unsigned *)(C_reindex_shared_dyn_warp + (((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8)) + 4)))[1]));
  }
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 1;");

  __syncthreads();
  for (int ax3_0_1_2 = 0; ax3_0_1_2 < 2; ++ax3_0_1_2) {
    for (int ax0_0_4 = 0; ax0_0_4 < 4; ++ax0_0_4) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(((half*)buf_dyn_shmem)[(((((((int)threadIdx.z) * 2048) + (ax0_0_4 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + ((((ax3_0_1_2 * 2) + (((int)threadIdx.x) >> 4)) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)) + 24576)])) + 0)));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(((half*)buf_dyn_shmem)[(((((((int)threadIdx.z) * 2048) + (ax0_0_4 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + ((((ax3_0_1_2 * 2) + (((int)threadIdx.x) >> 4)) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)) + 24576)])) + 0))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_4 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_4 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_4 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_4 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_5 = 0; ax0_0_5 < 4; ++ax0_0_5) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(((half*)buf_dyn_shmem)[((((((((int)threadIdx.y) * 2048) + (ax0_0_5 * 512)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + ((((ax3_0_1_2 * 2) + ((((int)threadIdx.x) & 15) >> 3)) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)) + 8192)])) + 0)));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(((half*)buf_dyn_shmem)[((((((((int)threadIdx.y) * 2048) + (ax0_0_5 * 512)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + ((((ax3_0_1_2 * 2) + ((((int)threadIdx.x) & 15) >> 3)) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)) + 8192)])) + 0))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + (ax0_0_5 * 8)))[0]), "=r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + (ax0_0_5 * 8)))[1]), "=r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + (ax0_0_5 * 8)))[2]), "=r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + (ax0_0_5 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_2 = 0; ax1_0_3_2 < 4; ++ax1_0_3_2) {
      for (int ax2_0_3_2 = 0; ax2_0_3_2 < 4; ++ax2_0_3_2) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(C_reindex_shared_dyn_warp + ((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8))))[0]), "=r"(((unsigned *)(C_reindex_shared_dyn_warp + ((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8))))[1])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_2 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_2 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_2 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_2 * 8)))[3]), "r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + (ax2_0_3_2 * 8)))[0]), "r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + (ax2_0_3_2 * 8)))[1]), "r"(((unsigned *)(C_reindex_shared_dyn_warp + ((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8))))[0]), "r"(((unsigned *)(C_reindex_shared_dyn_warp + ((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8))))[1]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(C_reindex_shared_dyn_warp + (((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8)) + 4)))[0]), "=r"(((unsigned *)(C_reindex_shared_dyn_warp + (((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8)) + 4)))[1])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_2 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_2 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_2 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_2 * 8)))[3]), "r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + ((ax2_0_3_2 * 8) + 4)))[0]), "r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + ((ax2_0_3_2 * 8) + 4)))[1]), "r"(((unsigned *)(C_reindex_shared_dyn_warp + (((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8)) + 4)))[0]), "r"(((unsigned *)(C_reindex_shared_dyn_warp + (((ax1_0_3_2 * 32) + (ax2_0_3_2 * 8)) + 4)))[1]));
  }
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int ax3_0_1_3 = 0; ax3_0_1_3 < 2; ++ax3_0_1_3) {
    for (int ax0_0_6 = 0; ax0_0_6 < 4; ++ax0_0_6) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(((half*)buf_dyn_shmem)[(((((((int)threadIdx.z) * 2048) + (ax0_0_6 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + ((((ax3_0_1_3 * 2) + (((int)threadIdx.x) >> 4)) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)) + 28672)])) + 0)));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(((half*)buf_dyn_shmem)[(((((((int)threadIdx.z) * 2048) + (ax0_0_6 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + ((((ax3_0_1_3 * 2) + (((int)threadIdx.x) >> 4)) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)) + 28672)])) + 0))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_6 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_6 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_6 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax0_0_6 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_0_7 = 0; ax0_0_7 < 4; ++ax0_0_7) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(((half*)buf_dyn_shmem)[((((((((int)threadIdx.y) * 2048) + (ax0_0_7 * 512)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + ((((ax3_0_1_3 * 2) + ((((int)threadIdx.x) & 15) >> 3)) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)) + 12288)])) + 0)));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(((half*)buf_dyn_shmem)[((((((((int)threadIdx.y) * 2048) + (ax0_0_7 * 512)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + ((((ax3_0_1_3 * 2) + ((((int)threadIdx.x) & 15) >> 3)) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)) + 12288)])) + 0))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + (ax0_0_7 * 8)))[0]), "=r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + (ax0_0_7 * 8)))[1]), "=r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + (ax0_0_7 * 8)))[2]), "=r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + (ax0_0_7 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_3 = 0; ax1_0_3_3 < 4; ++ax1_0_3_3) {
      for (int ax2_0_3_3 = 0; ax2_0_3_3 < 4; ++ax2_0_3_3) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(C_reindex_shared_dyn_warp + ((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8))))[0]), "=r"(((unsigned *)(C_reindex_shared_dyn_warp + ((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8))))[1])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_3 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_3 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_3 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_3 * 8)))[3]), "r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + (ax2_0_3_3 * 8)))[0]), "r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + (ax2_0_3_3 * 8)))[1]), "r"(((unsigned *)(C_reindex_shared_dyn_warp + ((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8))))[0]), "r"(((unsigned *)(C_reindex_shared_dyn_warp + ((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8))))[1]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(C_reindex_shared_dyn_warp + (((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8)) + 4)))[0]), "=r"(((unsigned *)(C_reindex_shared_dyn_warp + (((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8)) + 4)))[1])
      : "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_3 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_3 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_3 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_dyn_warp_1 + (ax1_0_3_3 * 8)))[3]), "r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + ((ax2_0_3_3 * 8) + 4)))[0]), "r"(((unsigned *)(B_reindex_shared_dyn_warp_1 + ((ax2_0_3_3 * 8) + 4)))[1]), "r"(((unsigned *)(C_reindex_shared_dyn_warp + (((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8)) + 4)))[0]), "r"(((unsigned *)(C_reindex_shared_dyn_warp + (((ax1_0_3_3 * 32) + (ax2_0_3_3 * 8)) + 4)))[1]));
  }
      }
    }
  }
  __syncthreads();
  for (int ax1_1 = 0; ax1_1 < 4; ++ax1_1) {
    for (int ax2_1 = 0; ax2_1 < 4; ++ax2_1) {
      for (int local_id = 0; local_id < 8; local_id+=2) {
*((uint *)&(&(((half*)buf_dyn_shmem)[(((((((int)threadIdx.z) * 8192) + (ax1_1 * 2048)) + (((int)threadIdx.y) * 64)) + (ax2_1 * 16)) + 16384)]))[((((((local_id % 4) / 2) * 8) + (threadIdx.x / 4)) * 128) + ((((local_id / 4) * 8) + ((threadIdx.x % 4) * 2)) + (local_id % 2)))]) = *((uint *)&C_reindex_shared_dyn_warp[((ax1_1 * 32) + (ax2_1 * 8)) + local_id]);
}
;
    }
  }
  __syncthreads();
  for (int ax0_ax1_fused_0_8 = 0; ax0_ax1_fused_0_8 < 64; ++ax0_ax1_fused_0_8) {
    *(uint4*)(C + ((((((((int)blockIdx.x) >> 3) * 131072) + (ax0_ax1_fused_0_8 * 2048)) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.x) & 7) * 128)) + ((((int)threadIdx.x) & 15) * 8))) = *(uint4*)(((half*)buf_dyn_shmem) + (((ax0_ax1_fused_0_8 * 256) + (((int)threadIdx.x) * 8)) + 16384));
  }
}

```
</details>

To use the operator, you can simply call the operator with the pytorch input tensors.

```python
inputs = []
inp = torch.rand((1024, 1024), dtype=torch.float16).cuda()
weight = torch.rand((1024, 1024), dtype=torch.float16).cuda()
output = torch.rand((1024, 1024), dtype=torch.float16).cuda()
matmul(inp, weight, output)
```

## Better Performance with Hardware Aware tuning

### Tune with Hardware Information

If you want to tune the operator to get better performance, you can use the api `hardware_aware_finetune`.

```python
print(matmul.profile_latency())
matmul.hardware_aware_finetune(topk=20)
print(matmul.profile_latency())
```

The latency will be reduced after tuning. We re-implement OSDI'22 paper Roller to do fast tuning with hardware information. Typically, the 20 candidates is good enough.

More details about Performance improvements can be found in the relevant document [Benchmark](../benchmark/README.md).

**Notice**: The tuning process uses hardware information to optimize performance. By default, the target is set to `cuda`. However, you can change the target to `nvidia/nvidia-a100` or any other [available targets](3rdparty/tvm/src/target/tag.cc) by setting the `TVM_TARGET` environment variable, like so: `export TVM_TARGET="nvidia/nvidia-a100"`. This allows the tuning process to utilize more specific hardware information. We also provide `auto_detect_nvidia_target` to automatically detect the target from `nvidia-smi`.

### Tune with Dynamic Symbolic

As in LLM Serving, the input shape is dynamic, we can use the dynamic symbolic to generate high performance kernel with dynamic shape.

```python
from bitblas.ops.matmul import Matmul, MatmulConfig

target = "cuda"
matmul_config = MatmulConfig(
    M=[1, 1024],
    N=1024,
    K=1024,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    layout="nt",
)
matmul = Matmul(
    config=matmul_config,
    target=target,
)
matmul.hardware_aware_finetune(topk=20)
```

Here is an example of the code generation result:

```python
@IRModule
class MatmulNT:

    def matmul_nt_opt_m_1(A: Tensor, T_reshape: Tensor, m: int):
        ...

    def matmul_nt_opt_m_256(A: Tensor, T_reshape: Tensor, m: int):
        ...

    def dispatcher(args):
        if m <= 1:
            matmul_nt_opt_m_1(A.data, T_reshape.data, m)
        if m > 1 and m <= 256:
            matmul_nt_opt_m_256(A.data, T_reshape.data, m)
        if m > 256:
            matmul_nt_m_256(A.data, T_reshape.data, m)
```

## Weight Only Dequantize Operator

The Weight Only Dequantize Operator is designed to efficiently perform dequantization of weights in neural networks that use quantization techniques to reduce the model size and potentially speed up computations. In quantized neural networks, weights are typically stored in a lower precision format (e.g., 4-bit integers in LLM) to save memory and computational resources. However, during inference, these quantized weights often need to be converted back to a higher precision format (e.g., 32-bit floating-point) to perform matrix multiplication with the input data.

The MatmulWeightOnlyDequantize operator specifically targets the scenario where only the weights are quantized, and the input data remains in a higher precision format. This operator can be particularly useful in systems where memory bandwidth is a bottleneck, as it allows for the storage of weight matrices in a compressed format and performs on-the-fly dequantization during the matrix multiplication operation.

Here's a brief overview of how you can configure and use the MatmulWeightOnlyDequantize operator of GPTQ INT4 Format (It's can be easily extended to other quantization formats like BitNET 1bit or NF 4bit):

```python
from bitblas.ops.matmul_dequantize import (
    MatmulWeightOnlyDequantize,
    MatmulWeightOnlyDequantizeConfig,
)

target = "cuda"
matmul_config = MatmulWeightOnlyDequantizeConfig(
    M=[1, 32, 64, 128, 256, 512],
    N=4096,
    K=4096,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    bit=4,
    storage_dtype="int8",
    source_format="uint",
    with_scaling=True,
    with_zeros=True,
    group_size=128,
    zeros_type="quantized",
)
matmul = MatmulWeightOnlyDequantize(
    config=matmul_config,
    target=target,
)

matmul.hardware_aware_finetune(topk=20)

```

## Use Packed Operators as Linear Module in PyTorch
The section "Use Packed Operators as Linear Module in PyTorch" demonstrates how to leverage the optimized matrix multiplication operations from BitBLAS directly within a PyTorch `nn.Module`. This enables the utilization of highly optimized, low-precision matrix multiplication operations. Relevant examples can be found at [integration/pytorch](../integration/pytorch/).

### Implementing a Custom Linear Module

Here's a step-by-step guide to implementing a custom linear module using BitBLAS packed operators:

```python
import torch
import torch.nn as nn
from bitblas.ops.matmul import Matmul, MatmulConfig

class BitBLASLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float16, layout="nt"):
        super(BitBLASLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.dtype = dtype
        self.layout = layout

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=dtype))
        if self.bias:
            self.bias = nn.Parameter(torch.randn(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)

        # Configure the Matmul operator
        self.matmul_config = MatmulConfig(
            M=1,  # Dynamic input feature size
            N=out_features,
            K=in_features,
            in_dtype="float16",
            out_dtype="float16",
            accum_dtype="float16",
            layout=self.layout,
        )
        self.matmul_op = Matmul(
            config=self.matmul_config,
            target="cuda",  # Assuming CUDA is available
        )

    def forward(self, x):
        output = self.matmul_op(x, self.weight)
        if self.bias is not None:
            output += self.bias
        return output
```

### Integration with PyTorch

This custom module can now be used as any standard PyTorch layer within your model definitions:

```python
model = nn.Sequential(
    BitBLASLinear(1024, 512, bias=True, dtype=torch.float16),
    nn.ReLU(),
    BitBLASLinear(512, 256, bias=True, dtype=torch.float16),
)

# Move model to GPU
model.cuda()

# Dummy input
x = torch.randn(1, 1024, dtype=torch.float16).cuda()

# Forward pass
output = model(x)
```

### Benefits

- **Performance:** By leveraging BitBLAS's optimized matrix multiplication, you can achieve comparable performance with Library.
- **Low Precision Arithmetic:** Utilizing low-precision computations (e.g., float16, int8) can drastically reduce memory usage and improve computational efficiency.
- **Customization:** The flexibility to configure the matrix multiplication operation allows for fine-tuning the performance characteristics to match specific hardware capabilities.

## PyTorch Weight Only Quantization Module

This guide demonstrates how to implement a PyTorch module for weight-only quantization, leveraging the capabilities of BitBLAS to handle low-precision arithmetic efficiently. This quantization approach focuses exclusively on reducing the precision of the weights while keeping input data in its original format, which can be particularly useful for models where the precision of input data is crucial.

### Implementing the QuantLinear Module

The `QuantLinear` module quantizes the weights of a linear layer into a specified bit-width representation (e.g., 1-bit, 2-bit, or 4-bit) for storage and computation efficiency. During the forward pass, it dequantizes the weights on-the-fly for matrix multiplication with the input data. Which can be used as an efficient alternative inference kernel in frameworks like AutoGPTQ or vLLM.

The following is an example implementation of the `QuantLinear` module using the `MatmulWeightOnlyDequantize` operator from BitBLAS.Please Checkout the integration of AutoGPTQ and vLLM integration in the relevant [integration](../integration) part of this project.

```python
import torch
import torch.nn as nn
from bitblas.quantization.utils import general_compress, interleave_weight
from bitblas.ops.matmul_dequantize import MatmulWeightOnlyDequantize, MatmulWeightOnlyDequantizeConfig
from bitblas.utils import auto_detect_nvidia_target

class QuantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        bias: bool = True,
        group_size: int = -1,
    ):
        super(QuantLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size if group_size != -1 else in_features
        self.bias = bias

        # Define quantized weight and bias parameters
        self.qweight = nn.Parameter(torch.empty(out_features, in_features // (8 // bits), dtype=torch.int8))
        if self.bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)

        # Configuration for the BitBLAS MatmulWeightOnlyDequantize operation
        self.matmul_config = MatmulWeightOnlyDequantizeConfig(
            M=1,  # The input size is dynamic
            N=out_features,
            K=in_features,
            bit=bits,
            layout="nt",
            with_bias=bias,
            group_size=self.group_size,
            storage_dtype="int8",
            source_format="uint",
            with_scaling=True,
            with_zeros=True,
            zeros_type="quantized",
        )
        self.matmul_op = MatmulWeightOnlyDequantize(config=self.matmul_config, target=auto_detect_nvidia_target())

    def forward(self, input):
        if self.bias is not None:
            output = self.matmul_op(input, self.qweight, self.bias)
        else:
            output = self.matmul_op(input, self.qweight)
        return output
```

### Using the QuantLinear Module

After defining the `QuantLinear` module, you can incorporate it into your neural network architectures in PyTorch. Here's an example of using `QuantLinear` in a simple feedforward network:

```python
class QuantizedNetwork(nn.Module):
    def __init__(self):
        super(QuantizedNetwork, self).__init__()
        self.layer1 = QuantLinear(in_features=784, out_features=256, bits=4, bias=True)
        self.relu = nn.ReLU()
        self.layer2 = QuantLinear(in_features=256, out_features=10, bits=4, bias=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Initialize the network and move it to GPU
net = QuantizedNetwork().cuda()

# Dummy input tensor
x = torch.randn(64, 784).float().cuda()

# Forward pass
output = net(x)
```

### Benefits of Weight Only Quantization

Certainly, here's a more detailed exploration of each point:

- **Diverse Data Type Support**: Weight-only quantization allows for a wider range of data type combinations, enabling the model to process inputs in high-precision formats (e.g., FP16/INT8) while the weights are stored and computed in a quantized, low-precision format (INT4/FP4/NF4).

- **Optimized Performance**: By quantizing the weights, models can leverage the specialized hardware acceleration features present in modern CPUs and GPUs, which are optimized for low-bit-width arithmetic operations.

- **Adaptability**: This ensures that quantized models can achieve optimal performance across a diverse array of platforms, without requiring significant modifications to the underlying architecture.

## Additional Notes

Please see more examples and relevant usages in `testing/python/operators`, and we currently only provide two operators: `Matmul` and `MatmulWeightOnlyDequantize`. But the packed operators can be easily extended to other operators through a triton-like DSL.

If you are interest in extending Operators to another Compute by yourself, Please Checkout [ExtendOperatorsWithDSL](./ExtendOperatorsWithDSL.md). 
