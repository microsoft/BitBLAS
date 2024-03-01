// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include <cuda_runtime.h>
#include <assert.h>
#include "ladder_kernel.h"
#include "mma.h"
// nvcc ladder_kernel.cu  -gencode arch=compute_80,code=sm_80

__global__ void __launch_bounds__(128) bitblas_kernel_fp16_int2_fp16_m1n15360k5120_nt(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ D) {
            signed char* B = ((int8_t *)QB);
	 half* Scale = (half *)((int8_t *)QB + 19660800); 
	 half* Zeros = (half *)((int8_t *)QB + 20889600);                 
            // const dim3 GridDim(15360, 1, 1);
            // const dim3 BlockDim(128, 1, 1);
            // bitblas_kernel_fp16_int2_fp16_m1n15360k5120_nt<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
  half in_thread_C_local[1];
  signed char B_local[4];
  half B_decode_local[8];
  half A_local[8];
  __shared__ half red_result[1];
  in_thread_C_local[0] = __float2half_rn(0.000000e+00f);
  for (int ax1_0 = 0; ax1_0 < 5; ++ax1_0) {
    *(int*)(B_local + 0) = *(int*)(B + (((((int)blockIdx.x) * 2560) + (ax1_0 * 512)) + (((int)threadIdx.x) * 4)));
    for (int ax1 = 0; ax1 < 8; ++ax1) {
      B_decode_local[ax1] = ((((half)((((uint)B_local[(ax1 >> 1)]) >> (((uint)(ax1 & 1)) * (uint)4)) & (uint)15)) - __float2half_rn(7.000000e+00f)) * Scale[(((((int)blockIdx.x) * 40) + (ax1_0 * 8)) + (((int)threadIdx.x) >> 4))]);
    }
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((ax1_0 * 1024) + (((int)threadIdx.x) * 8)));
    for (int ax1_2 = 0; ax1_2 < 8; ++ax1_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[ax1_2] * (B_decode_local[ax1_2] - Zeros[(((((int)blockIdx.x) * 40) + (ax1_0 * 8)) + (((int)threadIdx.x) >> 4))])));
    }
  }
  half red_buf0[1];
  uint mask[1];
  half t0[1];
  half red_buf0_1[1];
  uint mask_1[1];
  half t0_1[1];
  __shared__ half red_buf_staging[4];
  red_buf0_1[0] = in_thread_C_local[0];
  mask_1[0] = __activemask();
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  if ((((int)threadIdx.x) % 32) == 0) {
    red_buf_staging[(((int)threadIdx.x) >> 5)] = red_buf0_1[0];
  }
  __syncthreads();
  if (((int)threadIdx.x) < 4) {
    red_buf0[0] = red_buf_staging[((int)threadIdx.x)];
  }
  mask[0] = (__activemask() & (uint)15);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  if (((int)threadIdx.x) == 0) {
    ((volatile half*)red_result)[0] = red_buf0[0];
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    D[((int)blockIdx.x)] = (half)(((volatile half*)red_result)[0]);
  }
}



__global__ void __launch_bounds__(128) bitblas_kernel_fp16_int2_fp16_m128n15360k5120_nt(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ D) {
            signed char* B = ((int8_t *)QB);
	 half* Scale = (half *)((int8_t *)QB + 19660800); 
	 half* Zeros = (half *)((int8_t *)QB + 20889600);                 
            // const dim3 GridDim(240, 1, 1);
            // const dim3 BlockDim(1, 4, 1);
            // bitblas_kernel_fp16_int2_fp16_m128n15360k5120_nt<<<GridDim, BlockDim>>>(input_0, input_1, output);
        

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
      half C_reindex_shared_warp[64];
  __shared__ half A_reindex_shared[4096];
  __shared__ half B_zeros_reindex_shared[2048];
  signed char B_local[4];
  half B_zeros_reindex_local[8];
  half A_reindex_shared_warp[16];
  half B_zeros_reindex_shared_warp[32];
  for (int var = 0; var < 1; ++var) {
    for (int ax1_0_3_init = 0; ax1_0_3_init < 2; ++ax1_0_3_init) {
      for (int ax2_0_3_init = 0; ax2_0_3_init < 4; ++ax2_0_3_init) {
        for (int i = 0; i < 8; ++i) {
C_reindex_shared_warp[((ax1_0_3_init * 32) + (ax2_0_3_init * 8)) + i] = 0.0;}
;
      }
    }
    for (int ax3_0_0 = 0; ax3_0_0 < 160; ++ax3_0_0) {
      __syncthreads();
      #pragma unroll
      for (int ax0_ax1_ax2_fused_0 = 0; ax0_ax1_ax2_fused_0 < 4; ++ax0_ax1_ax2_fused_0) {
        *(uint4*)(A_reindex_shared + (((ax0_ax1_ax2_fused_0 * 1024) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(A + (((((ax0_ax1_ax2_fused_0 * 163840) + (((int)threadIdx.y) * 40960)) + ((((int)threadIdx.x) >> 2) * 5120)) + (ax3_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)));
      }
      for (int ax1_ax2_0_fused_0 = 0; ax1_ax2_0_fused_0 < 2; ++ax1_ax2_0_fused_0) {
        *(int*)(B_local + 0) = *(int*)(B + ((((((((int)blockIdx.x) * 163840) + (ax1_ax2_0_fused_0 * 81920)) + (((int)threadIdx.y) * 20480)) + ((((int)threadIdx.x) >> 2) * 2560)) + (ax3_0_0 * 16)) + ((((int)threadIdx.x) & 3) * 4)));
        for (int ax2 = 0; ax2 < 8; ++ax2) {
          B_zeros_reindex_local[ax2] = (((((half)((((uint)B_local[(ax2 >> 1)]) >> (((uint)(ax2 & 1)) * (uint)4)) & (uint)15)) - __float2half_rn(7.000000e+00f)) * Scale[(((((((int)blockIdx.x) * 2560) + (ax1_ax2_0_fused_0 * 1280)) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + (ax3_0_0 >> 2))]) - Zeros[(((((((int)blockIdx.x) * 2560) + (ax1_ax2_0_fused_0 * 1280)) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + (ax3_0_0 >> 2))]);
        }
        *(uint4*)(B_zeros_reindex_shared + (((ax1_ax2_0_fused_0 * 1024) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(B_zeros_reindex_local + 0);
      }
      __syncthreads();
      for (int ax3_0_1 = 0; ax3_0_1 < 2; ++ax3_0_1) {
        for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(A_reindex_shared[(((((int)threadIdx.y) * 1024) + (ax1_0 * 512)) + (ax3_0_1 * 16))])) + 0)));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_reindex_shared[(((((int)threadIdx.y) * 1024) + (ax1_0 * 512)) + (ax3_0_1 * 16))])) + 0))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_warp + (ax1_0 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_warp + (ax1_0 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_warp + (ax1_0 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_warp + (ax1_0 * 8)))[3])
      : "r"(addr)
    );
  }
        }
        for (int ax1_0_1 = 0; ax1_0_1 < 4; ++ax1_0_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(B_zeros_reindex_shared[((ax1_0_1 * 512) + (ax3_0_1 * 16))])) + 0)));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(B_zeros_reindex_shared[((ax1_0_1 * 512) + (ax3_0_1 * 16))])) + 0))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_zeros_reindex_shared_warp + (ax1_0_1 * 8)))[0]), "=r"(((unsigned *)(B_zeros_reindex_shared_warp + (ax1_0_1 * 8)))[1]), "=r"(((unsigned *)(B_zeros_reindex_shared_warp + (ax1_0_1 * 8)))[2]), "=r"(((unsigned *)(B_zeros_reindex_shared_warp + (ax1_0_1 * 8)))[3])
      : "r"(addr)
    );
  }
        }
        for (int ax1_0_3 = 0; ax1_0_3 < 2; ++ax1_0_3) {
          for (int ax2_0_3 = 0; ax2_0_3 < 4; ++ax2_0_3) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(C_reindex_shared_warp + ((ax1_0_3 * 32) + (ax2_0_3 * 8))))[0]), "=r"(((unsigned *)(C_reindex_shared_warp + ((ax1_0_3 * 32) + (ax2_0_3 * 8))))[1])
      : "r"(((unsigned *)(A_reindex_shared_warp + (ax1_0_3 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_warp + (ax1_0_3 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_warp + (ax1_0_3 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_warp + (ax1_0_3 * 8)))[3]), "r"(((unsigned *)(B_zeros_reindex_shared_warp + (ax2_0_3 * 8)))[0]), "r"(((unsigned *)(B_zeros_reindex_shared_warp + (ax2_0_3 * 8)))[1]), "r"(((unsigned *)(C_reindex_shared_warp + ((ax1_0_3 * 32) + (ax2_0_3 * 8))))[0]), "r"(((unsigned *)(C_reindex_shared_warp + ((ax1_0_3 * 32) + (ax2_0_3 * 8))))[1]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(C_reindex_shared_warp + (((ax1_0_3 * 32) + (ax2_0_3 * 8)) + 4)))[0]), "=r"(((unsigned *)(C_reindex_shared_warp + (((ax1_0_3 * 32) + (ax2_0_3 * 8)) + 4)))[1])
      : "r"(((unsigned *)(A_reindex_shared_warp + (ax1_0_3 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_warp + (ax1_0_3 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_warp + (ax1_0_3 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_warp + (ax1_0_3 * 8)))[3]), "r"(((unsigned *)(B_zeros_reindex_shared_warp + ((ax2_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(B_zeros_reindex_shared_warp + ((ax2_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(C_reindex_shared_warp + (((ax1_0_3 * 32) + (ax2_0_3 * 8)) + 4)))[0]), "r"(((unsigned *)(C_reindex_shared_warp + (((ax1_0_3 * 32) + (ax2_0_3 * 8)) + 4)))[1]));
  }
          }
        }
      }
    }
    for (int ax0 = 0; ax0 < 2; ++ax0) {
      for (int ax1 = 0; ax1 < 4; ++ax1) {
        __syncthreads();
        for (int local_id = 0; local_id < 8; local_id+=2) {
*((uint *)&(&(B_zeros_reindex_shared[(((int)threadIdx.y) * 512)]))[((((((local_id % 4) / 2) * 8) + (threadIdx.x / 4)) * 16) + ((((local_id / 4) * 8) + ((threadIdx.x % 4) * 2)) + (local_id % 2)))]) = *((uint *)&C_reindex_shared_warp[((ax0 * 32) + (ax1 * 8)) + local_id]);
}
;
        __syncthreads();
        #pragma unroll
        for (int ax0_ax1_ax2_ax3_ax4_fused_0 = 0; ax0_ax1_ax2_ax3_ax4_fused_0 < 1; ++ax0_ax1_ax2_ax3_ax4_fused_0) {
          *(uint4*)(D + ((((((((int)threadIdx.y) * 491520) + (ax0 * 245760)) + ((((int)threadIdx.x) >> 1) * 15360)) + (((int)blockIdx.x) * 64)) + (ax1 * 16)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(B_zeros_reindex_shared + ((((int)threadIdx.y) * 512) + (((int)threadIdx.x) * 8)));
        }
      }
    }
  }
}





int ladder_gemm_fp16xint2_fp16(half *input_0, half *input_1, half *output, const int M, const int N, const int K, const int trans_a, const int trans_b, half *workspace_ptr)
{
    assert(trans_a == 0 && trans_b == 1);
    
    if (M == 1 && N == 15360 && K == 5120){
        
             const dim3 GridDim(15360, 1, 1);
             const dim3 BlockDim(128, 1, 1);
             bitblas_kernel_fp16_int2_fp16_m1n15360k5120_nt<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
        return 0;
    }

    
    if (M == 128 && N == 15360 && K == 5120){
        
             const dim3 GridDim(240, 1, 1);
             const dim3 BlockDim(1, 4, 1);
             bitblas_kernel_fp16_int2_fp16_m128n15360k5120_nt<<<GridDim, BlockDim>>>(input_0, input_1, output);
        
        return 0;
    }

    
    return -1;
}