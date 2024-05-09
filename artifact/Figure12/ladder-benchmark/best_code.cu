__global__ void __launch_bounds__(128) Fused(int8_t* __restrict__ A, int8_t* __restrict__ B, int* __restrict__ C) {
  
  int C_warp[16];
  __shared__ signed char A_shared[1024];
  __shared__ signed char B_shared[8192];
  signed char A_shared_warp[16];
  signed char B_shared_warp[32];

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
  
  for (int i_2_init = 0; i_2_init < 1; ++i_2_init) {
    for (int j_2_init = 0; j_2_init < 2; ++j_2_init) {
      for (int i = 0; i < 8; ++i) {
C_warp[(j_2_init * 8) + i] = 0.0;}
;
    }
  }
  for (int k_0 = 0; k_0 < 448; ++k_0) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_ax2_ax3_0_fused_0 = 0; ax0_ax1_ax2_ax3_0_fused_0 < 1; ++ax0_ax1_ax2_ax3_0_fused_0) {
      if (((int)threadIdx.z) < 2) {
        *(int4*)(A_shared + ((((int)threadIdx.z) * 512) + (((int)threadIdx.x) * 16))) = *(int4*)(A + ((((((int)blockIdx.y) * 458752) + (k_0 * 1024)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16)));
      }
    }
    #pragma unroll
    for (int ax0_ax1_ax2_ax3_0_fused_0_1 = 0; ax0_ax1_ax2_ax3_0_fused_0_1 < 4; ++ax0_ax1_ax2_ax3_0_fused_0_1) {
      *(int4*)(B_shared + (((ax0_ax1_ax2_ax3_0_fused_0_1 * 2048) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(B + ((((((((int)blockIdx.x) * 3670016) + (ax0_ax1_ax2_ax3_0_fused_0_1 * 917504)) + ((((int)threadIdx.z) >> 1) * 458752)) + (k_0 * 1024)) + ((((int)threadIdx.z) & 1) * 512)) + (((int)threadIdx.x) * 16)));
    }
    __syncthreads();
    for (int k_1 = 0; k_1 < 2; ++k_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(A_shared[(k_1 * 512)])) + (((int)threadIdx.x) * 16))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_shared[(k_1 * 512)])) + (((int)threadIdx.x) * 16)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_shared_warp + 0))[0]), "=r"(((unsigned *)(A_shared_warp + 0))[1]), "=r"(((unsigned *)(A_shared_warp + 0))[2]), "=r"(((unsigned *)(A_shared_warp + 0))[3])
      : "r"(addr)
    );
  }
      for (int ax0 = 0; ax0 < 2; ++ax0) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(B_shared[(((((int)threadIdx.z) * 2048) + (ax0 * 1024)) + (k_1 * 512))])) + (((int)threadIdx.x) * 16))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(B_shared[(((((int)threadIdx.z) * 2048) + (ax0 * 1024)) + (k_1 * 512))])) + (((int)threadIdx.x) * 16)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_shared_warp + (ax0 * 16)))[0]), "=r"(((unsigned *)(B_shared_warp + (ax0 * 16)))[1]), "=r"(((unsigned *)(B_shared_warp + (ax0 * 16)))[2]), "=r"(((unsigned *)(B_shared_warp + (ax0 * 16)))[3])
      : "r"(addr)
    );
  }
      }
      for (int j_2 = 0; j_2 < 2; ++j_2) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=r"(((int *)(C_warp + (j_2 * 8)))[0]), "=r"(((int *)(C_warp + (j_2 * 8)))[1]), "=r"(((int *)(C_warp + (j_2 * 8)))[2]), "=r"(((int *)(C_warp + (j_2 * 8)))[3])
      : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + (j_2 * 16)))[0]), "r"(((unsigned *)(B_shared_warp + (j_2 * 16)))[1]), "r"(((int *)(C_warp + (j_2 * 8)))[0]), "r"(((int *)(C_warp + (j_2 * 8)))[1]), "r"(((int *)(C_warp + (j_2 * 8)))[2]), "r"(((int *)(C_warp + (j_2 * 8)))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=r"(((int *)(C_warp + ((j_2 * 8) + 4)))[0]), "=r"(((int *)(C_warp + ((j_2 * 8) + 4)))[1]), "=r"(((int *)(C_warp + ((j_2 * 8) + 4)))[2]), "=r"(((int *)(C_warp + ((j_2 * 8) + 4)))[3])
      : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + ((j_2 * 16) + 8)))[0]), "r"(((unsigned *)(B_shared_warp + ((j_2 * 16) + 8)))[1]), "r"(((int *)(C_warp + ((j_2 * 8) + 4)))[0]), "r"(((int *)(C_warp + ((j_2 * 8) + 4)))[1]), "r"(((int *)(C_warp + ((j_2 * 8) + 4)))[2]), "r"(((int *)(C_warp + ((j_2 * 8) + 4)))[3]));
  }
      }
    }
  }
  for (int ax1 = 0; ax1 < 2; ++ax1) {
    for (int local_id = 0; local_id < 8; ++local_id) {
(&(C[((((((int)blockIdx.y) * 131072) + (((int)blockIdx.x) * 2048)) + (((int)threadIdx.z) * 512)) + (ax1 * 256))]))[((((((local_id % 4) / 2) * 8) + (threadIdx.x / 4)) * 16) + ((((local_id / 4) * 8) + ((threadIdx.x % 4) * 2)) + (local_id % 2)))] = C_warp[(ax1 * 8) + local_id];
}
;
  }
}

