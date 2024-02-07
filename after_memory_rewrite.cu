#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
#include <sm_61_intrinsics.h>


#if defined(__CUDACC_RTC__)
#define __SM_61_INTRINSICS_DECL__ __device__
#else /* !__CUDACC_RTC__ */
#define __SM_61_INTRINSICS_DECL__ static __device__ __inline__
#endif /* __CUDACC_RTC__ */

#ifndef __CUDA_ARCH__
#define __DEF_IF_HOST { }
#else  /* !__CUDA_ARCH__ */
#define __DEF_IF_HOST ;
#endif /* __CUDA_ARCH__ */

__SM_61_INTRINSICS_DECL__ int __dp4a(unsigned int srcA, int srcB, int c) __DEF_IF_HOST
__SM_61_INTRINSICS_DECL__ int __dp4a(int srcA, unsigned int srcB, int c) __DEF_IF_HOST

#undef __DEF_IF_HOST

#if !defined(__CUDACC_RTC__) && defined(__CUDA_ARCH__)
__SM_61_INTRINSICS_DECL__ int __dp4a(unsigned int srcA, int srcB, int c) {
    int ret;
    asm volatile ("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(ret) : "r"(srcA), "r"(srcB), "r"(c));
    return ret;
}

__SM_61_INTRINSICS_DECL__ int __dp4a(int srcA, unsigned int srcB, int c) {
    int ret;
    asm volatile ("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(srcA), "r"(srcB), "r"(c));
    return ret;
}
#endif /* !__CUDACC_RTC__ && defined(__CUDA_ARCH__) */

#undef __SM_61_INTRINSICS_DECL__

#endif
__forceinline__ __device__ unsigned int
cast_smem_ptr_to_int(const void* const smem_ptr)
{
  unsigned int smem_int;
  asm volatile ("{ .reg .u64 smem_int; cvta.to.shared.u64 smem_int, %1; cvt.u32.u64 %0, smem_int; }"
    : "=r"(smem_int) : "l"(smem_ptr));
  return smem_int;
}

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800) 
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 1
#else
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 0
#endif
extern "C" __global__ void __launch_bounds__(128) default_function_kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ D);
extern "C" __global__ void __launch_bounds__(128) default_function_kernel(signed char* __restrict__ A, signed char* __restrict__ B, signed char* __restrict__ D) {

        const int MAX_BLOCK_N = 10;
        const auto baseBlockIdx = blockIdx.x + gridDim.x *blockIdx.y;
        const auto totalPanel = (gridDim.x * gridDim.y +MAX_BLOCK_N * gridDim.x - 1) / (MAX_BLOCK_N * gridDim.x);
        const auto totalBlock = gridDim.x * gridDim.y;
        const auto panelIdx = baseBlockIdx / (MAX_BLOCK_N *gridDim.x);
        const auto strideLd = panelIdx + 1 < totalPanel ?MAX_BLOCK_N : (totalBlock - panelIdx * (MAX_BLOCK_N *gridDim.x)) / gridDim.x;
        const auto bx = (panelIdx & 1) ? gridDim.x -(baseBlockIdx - panelIdx * MAX_BLOCK_N * gridDim.x) /strideLd - 1 : (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) / strideLd;
        const auto by = (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) % strideLd + panelIdx * MAX_BLOCK_N;
        const auto bz = blockIdx.z;
        const dim3 blockIdx(bx, by, bz);
      int C_reindex_shared_warp[128];
  __shared__ signed char A_reindex_reindex_shared[32768];
  __shared__ signed char B_shared[2048];
  __shared__ signed char B_reindex_reindex_shared[4096];
  signed char B_local[4];
  signed char B_reindex_reindex_local[16];
  signed char A_reindex_reindex_shared_warp[128];
  signed char B_reindex_reindex_shared_warp[32];
  signed char B_local_1[4];
  signed char B_reindex_reindex_local_1[16];
  signed char A_reindex_reindex_shared_warp_1[128];
  signed char B_reindex_reindex_shared_warp_1[32];
  for (int var = 0; var < 1; ++var) {
    for (int ax1_0_3_init = 0; ax1_0_3_init < 8; ++ax1_0_3_init) {
      for (int ax2_0_3_init = 0; ax2_0_3_init < 2; ++ax2_0_3_init) {
        for (int i = 0; i < 8; ++i) {
C_reindex_shared_warp[((ax1_0_3_init * 16) + (ax2_0_3_init * 8)) + i] = 0.0;}
;
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_ax2_ax3_ax4_fused_0 = 0; ax0_ax1_ax2_ax3_ax4_fused_0 < 8; ++ax0_ax1_ax2_ax3_ax4_fused_0) {

  {
        unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_reindex_reindex_shared + ((((ax0_ax1_ax2_ax3_ax4_fused_0 * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_reindex_reindex_shared + ((((ax0_ax1_ax2_ax3_ax4_fused_0 * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + (((((((int)blockIdx.y) * 4194304) + (ax0_ax1_ax2_ax3_ax4_fused_0 * 524288)) + (((int)threadIdx.y) * 262144)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16)))), "n"(16)
    );
  }
    }
    #pragma unroll
    for (int ax0_ax1_ax2_ax3_fused_0 = 0; ax0_ax1_ax2_ax3_fused_0 < 1; ++ax0_ax1_ax2_ax3_fused_0) {
      if (((int)threadIdx.z) < 1) {

  {
        unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(B_shared + ((((int)threadIdx.y) * 512) + (((int)threadIdx.x) * 16)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + ((((int)threadIdx.y) * 512) + (((int)threadIdx.x) * 16))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + (((((((int)blockIdx.x) * 262144) + (((int)threadIdx.z) * 262144)) + (((int)threadIdx.y) * 131072)) + ((((int)threadIdx.x) >> 4) * 65536)) + ((((int)threadIdx.x) & 15) * 16)))), "n"(16)
    );
  }
      }
    }
__asm__ __volatile__("cp.async.commit_group;");

    for (int ax3_0_0 = 0; ax3_0_0 < 255; ++ax3_0_0) {
      __syncthreads();
      #pragma unroll
      for (int ax0_ax1_ax2_ax3_ax4_fused_0_1 = 0; ax0_ax1_ax2_ax3_ax4_fused_0_1 < 8; ++ax0_ax1_ax2_ax3_ax4_fused_0_1) {

  {
        unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_reindex_reindex_shared + (((((((ax3_0_0 + 1) & 1) * 16384) + (ax0_ax1_ax2_ax3_ax4_fused_0_1 * 2048)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_reindex_reindex_shared + (((((((ax3_0_0 + 1) & 1) * 16384) + (ax0_ax1_ax2_ax3_ax4_fused_0_1 * 2048)) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + (((((((((int)blockIdx.y) * 4194304) + (ax0_ax1_ax2_ax3_ax4_fused_0_1 * 524288)) + (((int)threadIdx.y) * 262144)) + (ax3_0_0 * 1024)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16)) + 1024))), "n"(16)
    );
  }
      }
      #pragma unroll
      for (int ax0_ax1_ax2_ax3_fused_0_1 = 0; ax0_ax1_ax2_ax3_fused_0_1 < 1; ++ax0_ax1_ax2_ax3_fused_0_1) {
        if (((int)threadIdx.z) < 1) {

  {
        unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(B_shared + (((((ax3_0_0 + 1) & 1) * 1024) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + (((((ax3_0_0 + 1) & 1) * 1024) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + (((((((((int)blockIdx.x) * 262144) + (((int)threadIdx.z) * 262144)) + (((int)threadIdx.y) * 131072)) + ((((int)threadIdx.x) >> 4) * 65536)) + (ax3_0_0 * 256)) + ((((int)threadIdx.x) & 15) * 16)) + 256))), "n"(16)
    );
  }
        }
      }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

      __syncthreads();
      for (int ax1_ax2_ax3_ax4_0_fused_0 = 0; ax1_ax2_ax3_ax4_0_fused_0 < 2; ++ax1_ax2_ax3_ax4_0_fused_0) {
        *(int*)(B_local + 0) = *(int*)(B_shared + ((((((ax3_0_0 & 1) * 1024) + (ax1_ax2_ax3_ax4_0_fused_0 * 512)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.x) * 4)));
        for (int ax4 = 0; ax4 < 16; ++ax4) {
          B_reindex_reindex_local[ax4] = ((B_local[(ax4 >> 2)] >> ((signed char)((ax4 & 3) * 2))) & (signed char)3);
        }
        *(int4*)(B_reindex_reindex_shared + ((((ax1_ax2_ax3_ax4_0_fused_0 * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(B_reindex_reindex_local + 0);
      }
      __syncthreads();
      for (int ax3_0_1 = 0; ax3_0_1 < 2; ++ax3_0_1) {
        for (int ax1 = 0; ax1 < 8; ++ax1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(A_reindex_reindex_shared[(((((ax3_0_0 & 1) * 16384) + (((int)threadIdx.y) * 8192)) + (ax1 * 1024)) + (ax3_0_1 * 512))])) + (((int)threadIdx.x) * 16))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_reindex_reindex_shared[(((((ax3_0_0 & 1) * 16384) + (((int)threadIdx.y) * 8192)) + (ax1 * 1024)) + (ax3_0_1 * 512))])) + (((int)threadIdx.x) * 16)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_reindex_shared_warp + (ax1 * 16)))[0]), "=r"(((unsigned *)(A_reindex_reindex_shared_warp + (ax1 * 16)))[1]), "=r"(((unsigned *)(A_reindex_reindex_shared_warp + (ax1 * 16)))[2]), "=r"(((unsigned *)(A_reindex_reindex_shared_warp + (ax1 * 16)))[3])
      : "r"(addr)
    );
  }
        }
        for (int ax1_1 = 0; ax1_1 < 2; ++ax1_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(B_reindex_reindex_shared[(((((int)threadIdx.z) * 2048) + (ax1_1 * 1024)) + (ax3_0_1 * 512))])) + (((int)threadIdx.x) * 16))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(B_reindex_reindex_shared[(((((int)threadIdx.z) * 2048) + (ax1_1 * 1024)) + (ax3_0_1 * 512))])) + (((int)threadIdx.x) * 16)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_reindex_reindex_shared_warp + (ax1_1 * 16)))[0]), "=r"(((unsigned *)(B_reindex_reindex_shared_warp + (ax1_1 * 16)))[1]), "=r"(((unsigned *)(B_reindex_reindex_shared_warp + (ax1_1 * 16)))[2]), "=r"(((unsigned *)(B_reindex_reindex_shared_warp + (ax1_1 * 16)))[3])
      : "r"(addr)
    );
  }
        }
        for (int ax1_0_3 = 0; ax1_0_3 < 8; ++ax1_0_3) {
          for (int ax2_0_3 = 0; ax2_0_3 < 2; ++ax2_0_3) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=r"(((int *)(C_reindex_shared_warp + ((ax1_0_3 * 16) + (ax2_0_3 * 8))))[0]), "=r"(((int *)(C_reindex_shared_warp + ((ax1_0_3 * 16) + (ax2_0_3 * 8))))[1]), "=r"(((int *)(C_reindex_shared_warp + ((ax1_0_3 * 16) + (ax2_0_3 * 8))))[2]), "=r"(((int *)(C_reindex_shared_warp + ((ax1_0_3 * 16) + (ax2_0_3 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_reindex_shared_warp + (ax1_0_3 * 16)))[0]), "r"(((unsigned *)(A_reindex_reindex_shared_warp + (ax1_0_3 * 16)))[1]), "r"(((unsigned *)(A_reindex_reindex_shared_warp + (ax1_0_3 * 16)))[2]), "r"(((unsigned *)(A_reindex_reindex_shared_warp + (ax1_0_3 * 16)))[3]), "r"(((unsigned *)(B_reindex_reindex_shared_warp + (ax2_0_3 * 16)))[0]), "r"(((unsigned *)(B_reindex_reindex_shared_warp + (ax2_0_3 * 16)))[1]), "r"(((int *)(C_reindex_shared_warp + ((ax1_0_3 * 16) + (ax2_0_3 * 8))))[0]), "r"(((int *)(C_reindex_shared_warp + ((ax1_0_3 * 16) + (ax2_0_3 * 8))))[1]), "r"(((int *)(C_reindex_shared_warp + ((ax1_0_3 * 16) + (ax2_0_3 * 8))))[2]), "r"(((int *)(C_reindex_shared_warp + ((ax1_0_3 * 16) + (ax2_0_3 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=r"(((int *)(C_reindex_shared_warp + (((ax1_0_3 * 16) + (ax2_0_3 * 8)) + 4)))[0]), "=r"(((int *)(C_reindex_shared_warp + (((ax1_0_3 * 16) + (ax2_0_3 * 8)) + 4)))[1]), "=r"(((int *)(C_reindex_shared_warp + (((ax1_0_3 * 16) + (ax2_0_3 * 8)) + 4)))[2]), "=r"(((int *)(C_reindex_shared_warp + (((ax1_0_3 * 16) + (ax2_0_3 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_reindex_shared_warp + (ax1_0_3 * 16)))[0]), "r"(((unsigned *)(A_reindex_reindex_shared_warp + (ax1_0_3 * 16)))[1]), "r"(((unsigned *)(A_reindex_reindex_shared_warp + (ax1_0_3 * 16)))[2]), "r"(((unsigned *)(A_reindex_reindex_shared_warp + (ax1_0_3 * 16)))[3]), "r"(((unsigned *)(B_reindex_reindex_shared_warp + ((ax2_0_3 * 16) + 8)))[0]), "r"(((unsigned *)(B_reindex_reindex_shared_warp + ((ax2_0_3 * 16) + 8)))[1]), "r"(((int *)(C_reindex_shared_warp + (((ax1_0_3 * 16) + (ax2_0_3 * 8)) + 4)))[0]), "r"(((int *)(C_reindex_shared_warp + (((ax1_0_3 * 16) + (ax2_0_3 * 8)) + 4)))[1]), "r"(((int *)(C_reindex_shared_warp + (((ax1_0_3 * 16) + (ax2_0_3 * 8)) + 4)))[2]), "r"(((int *)(C_reindex_shared_warp + (((ax1_0_3 * 16) + (ax2_0_3 * 8)) + 4)))[3]));
  }
          }
        }
      }
    }
__asm__ __volatile__("cp.async.wait_group 0;");

    __syncthreads();
    for (int ax1_ax2_ax3_ax4_0_fused_0_1 = 0; ax1_ax2_ax3_ax4_0_fused_0_1 < 2; ++ax1_ax2_ax3_ax4_0_fused_0_1) {
      *(int*)(B_local_1 + 0) = *(int*)(B_shared + (((((ax1_ax2_ax3_ax4_0_fused_0_1 * 512) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.x) * 4)) + 1024));
      for (int ax4_1 = 0; ax4_1 < 16; ++ax4_1) {
        B_reindex_reindex_local_1[ax4_1] = ((B_local_1[(ax4_1 >> 2)] >> ((signed char)((ax4_1 & 3) * 2))) & (signed char)3);
      }
      *(int4*)(B_reindex_reindex_shared + ((((ax1_ax2_ax3_ax4_0_fused_0_1 * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(B_reindex_reindex_local_1 + 0);
    }
    __syncthreads();
    for (int ax3_0_1_1 = 0; ax3_0_1_1 < 2; ++ax3_0_1_1) {
      for (int ax1_2 = 0; ax1_2 < 8; ++ax1_2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(A_reindex_reindex_shared[((((((int)threadIdx.y) * 8192) + (ax1_2 * 1024)) + (ax3_0_1_1 * 512)) + 16384)])) + (((int)threadIdx.x) * 16))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_reindex_reindex_shared[((((((int)threadIdx.y) * 8192) + (ax1_2 * 1024)) + (ax3_0_1_1 * 512)) + 16384)])) + (((int)threadIdx.x) * 16)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_reindex_shared_warp_1 + (ax1_2 * 16)))[0]), "=r"(((unsigned *)(A_reindex_reindex_shared_warp_1 + (ax1_2 * 16)))[1]), "=r"(((unsigned *)(A_reindex_reindex_shared_warp_1 + (ax1_2 * 16)))[2]), "=r"(((unsigned *)(A_reindex_reindex_shared_warp_1 + (ax1_2 * 16)))[3])
      : "r"(addr)
    );
  }
      }
      for (int ax1_3 = 0; ax1_3 < 2; ++ax1_3) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(B_reindex_reindex_shared[(((((int)threadIdx.z) * 2048) + (ax1_3 * 1024)) + (ax3_0_1_1 * 512))])) + (((int)threadIdx.x) * 16))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(B_reindex_reindex_shared[(((((int)threadIdx.z) * 2048) + (ax1_3 * 1024)) + (ax3_0_1_1 * 512))])) + (((int)threadIdx.x) * 16)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_reindex_reindex_shared_warp_1 + (ax1_3 * 16)))[0]), "=r"(((unsigned *)(B_reindex_reindex_shared_warp_1 + (ax1_3 * 16)))[1]), "=r"(((unsigned *)(B_reindex_reindex_shared_warp_1 + (ax1_3 * 16)))[2]), "=r"(((unsigned *)(B_reindex_reindex_shared_warp_1 + (ax1_3 * 16)))[3])
      : "r"(addr)
    );
  }
      }
      for (int ax1_0_3_1 = 0; ax1_0_3_1 < 8; ++ax1_0_3_1) {
        for (int ax2_0_3_1 = 0; ax2_0_3_1 < 2; ++ax2_0_3_1) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=r"(((int *)(C_reindex_shared_warp + ((ax1_0_3_1 * 16) + (ax2_0_3_1 * 8))))[0]), "=r"(((int *)(C_reindex_shared_warp + ((ax1_0_3_1 * 16) + (ax2_0_3_1 * 8))))[1]), "=r"(((int *)(C_reindex_shared_warp + ((ax1_0_3_1 * 16) + (ax2_0_3_1 * 8))))[2]), "=r"(((int *)(C_reindex_shared_warp + ((ax1_0_3_1 * 16) + (ax2_0_3_1 * 8))))[3])
      : "r"(((unsigned *)(A_reindex_reindex_shared_warp_1 + (ax1_0_3_1 * 16)))[0]), "r"(((unsigned *)(A_reindex_reindex_shared_warp_1 + (ax1_0_3_1 * 16)))[1]), "r"(((unsigned *)(A_reindex_reindex_shared_warp_1 + (ax1_0_3_1 * 16)))[2]), "r"(((unsigned *)(A_reindex_reindex_shared_warp_1 + (ax1_0_3_1 * 16)))[3]), "r"(((unsigned *)(B_reindex_reindex_shared_warp_1 + (ax2_0_3_1 * 16)))[0]), "r"(((unsigned *)(B_reindex_reindex_shared_warp_1 + (ax2_0_3_1 * 16)))[1]), "r"(((int *)(C_reindex_shared_warp + ((ax1_0_3_1 * 16) + (ax2_0_3_1 * 8))))[0]), "r"(((int *)(C_reindex_shared_warp + ((ax1_0_3_1 * 16) + (ax2_0_3_1 * 8))))[1]), "r"(((int *)(C_reindex_shared_warp + ((ax1_0_3_1 * 16) + (ax2_0_3_1 * 8))))[2]), "r"(((int *)(C_reindex_shared_warp + ((ax1_0_3_1 * 16) + (ax2_0_3_1 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=r"(((int *)(C_reindex_shared_warp + (((ax1_0_3_1 * 16) + (ax2_0_3_1 * 8)) + 4)))[0]), "=r"(((int *)(C_reindex_shared_warp + (((ax1_0_3_1 * 16) + (ax2_0_3_1 * 8)) + 4)))[1]), "=r"(((int *)(C_reindex_shared_warp + (((ax1_0_3_1 * 16) + (ax2_0_3_1 * 8)) + 4)))[2]), "=r"(((int *)(C_reindex_shared_warp + (((ax1_0_3_1 * 16) + (ax2_0_3_1 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_reindex_reindex_shared_warp_1 + (ax1_0_3_1 * 16)))[0]), "r"(((unsigned *)(A_reindex_reindex_shared_warp_1 + (ax1_0_3_1 * 16)))[1]), "r"(((unsigned *)(A_reindex_reindex_shared_warp_1 + (ax1_0_3_1 * 16)))[2]), "r"(((unsigned *)(A_reindex_reindex_shared_warp_1 + (ax1_0_3_1 * 16)))[3]), "r"(((unsigned *)(B_reindex_reindex_shared_warp_1 + ((ax2_0_3_1 * 16) + 8)))[0]), "r"(((unsigned *)(B_reindex_reindex_shared_warp_1 + ((ax2_0_3_1 * 16) + 8)))[1]), "r"(((int *)(C_reindex_shared_warp + (((ax1_0_3_1 * 16) + (ax2_0_3_1 * 8)) + 4)))[0]), "r"(((int *)(C_reindex_shared_warp + (((ax1_0_3_1 * 16) + (ax2_0_3_1 * 8)) + 4)))[1]), "r"(((int *)(C_reindex_shared_warp + (((ax1_0_3_1 * 16) + (ax2_0_3_1 * 8)) + 4)))[2]), "r"(((int *)(C_reindex_shared_warp + (((ax1_0_3_1 * 16) + (ax2_0_3_1 * 8)) + 4)))[3]));
  }
        }
      }
    }
    for (int ax0 = 0; ax0 < 8; ++ax0) {
      for (int ax1_4 = 0; ax1_4 < 2; ++ax1_4) {
        __syncthreads();
        for (int local_id = 0; local_id < 8; ++local_id) {
(&(((int*)A_reindex_reindex_shared)[((((int)threadIdx.y) * 6144) + (((int)threadIdx.z) * 512))]))[((((((local_id % 4) / 2) * 8) + (threadIdx.x / 4)) * 16) + ((((local_id / 4) * 8) + ((threadIdx.x % 4) * 2)) + (local_id % 2)))] = C_reindex_shared_warp[((ax0 * 16) + (ax1_4 * 8)) + local_id];
}
;
        __syncthreads();
        #pragma unroll
        for (int ax0_ax1_ax2_ax3_ax4_fused_0_2 = 0; ax0_ax1_ax2_ax3_ax4_fused_0_2 < 2; ++ax0_ax1_ax2_ax3_ax4_fused_0_2) {
          int __1;
          int4 v_ = *(int4*)(((int*)A_reindex_reindex_shared) + ((((((int)threadIdx.y) * 6144) + (((int)threadIdx.z) * 512)) + (ax0_ax1_ax2_ax3_ax4_fused_0_2 * 128)) + (((int)threadIdx.x) * 4)));
          __1=((signed char)(v_.x) << 0);
          __1=__1 & ~(0x000000ff << 8) |((signed char)(v_.y) << 8);
          __1=__1 & ~(0x000000ff << 16) |((signed char)(v_.z) << 16);
          __1=__1 & ~(0x000000ff << 24) |((signed char)(v_.w) << 24);
          *(int*)(D + (((((((((((int)blockIdx.y) * 4194304) + (((int)threadIdx.y) * 2097152)) + (ax0 * 262144)) + (ax0_ax1_ax2_ax3_ax4_fused_0_2 * 131072)) + ((((int)threadIdx.x) >> 2) * 16384)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.z) * 32)) + (ax1_4 * 16)) + ((((int)threadIdx.x) & 3) * 4))) = __1;
        }
      }
    }
  }
}

