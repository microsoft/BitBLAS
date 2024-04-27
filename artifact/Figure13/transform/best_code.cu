__global__ void __launch_bounds__(128) Fused(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C) {
  
  half C_warp[512];
  __shared__ half A_shared[16384];
  __shared__ half B_shared[4096];
  half A_shared_warp[128];
  half B_shared_warp[32];

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
  
  for (int i_2_init = 0; i_2_init < 16; ++i_2_init) {
    for (int j_2_init = 0; j_2_init < 4; ++j_2_init) {
      for (int i = 0; i < 8; ++i) {
C_warp[((i_2_init * 32) + (j_2_init * 8)) + i] = 0.0;}
;
    }
  }
  for (int k_0 = 0; k_0 < 256; ++k_0) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_ax2_ax3_0_fused_0 = 0; ax0_ax1_ax2_ax3_0_fused_0 < 16; ++ax0_ax1_ax2_ax3_0_fused_0) {
      *(uint4*)(A_shared + ((((ax0_ax1_ax2_ax3_0_fused_0 * 1024) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(A + ((((((((int)blockIdx.y) * 4194304) + (ax0_ax1_ax2_ax3_0_fused_0 * 262144)) + (((int)threadIdx.y) * 131072)) + (k_0 * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8)));
    }
    #pragma unroll
    for (int ax0_ax1_ax2_ax3_0_fused_0_1 = 0; ax0_ax1_ax2_ax3_0_fused_0_1 < 4; ++ax0_ax1_ax2_ax3_0_fused_0_1) {
      *(uint4*)(B_shared + ((((ax0_ax1_ax2_ax3_0_fused_0_1 * 1024) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(B + ((((((((int)blockIdx.x) * 1048576) + (ax0_ax1_ax2_ax3_0_fused_0_1 * 262144)) + (((int)threadIdx.y) * 131072)) + (k_0 * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8)));
    }
    __syncthreads();
    for (int k_1 = 0; k_1 < 2; ++k_1) {
      for (int ax0 = 0; ax0 < 16; ++ax0) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(A_shared[(((((int)threadIdx.y) * 8192) + (ax0 * 512)) + (k_1 * 256))])) + (((int)threadIdx.x) * 8))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_shared[(((((int)threadIdx.y) * 8192) + (ax0 * 512)) + (k_1 * 256))])) + (((int)threadIdx.x) * 8)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_shared_warp + (ax0 * 8)))[0]), "=r"(((unsigned *)(A_shared_warp + (ax0 * 8)))[1]), "=r"(((unsigned *)(A_shared_warp + (ax0 * 8)))[2]), "=r"(((unsigned *)(A_shared_warp + (ax0 * 8)))[3])
      : "r"(addr)
    );
  }
      }
      for (int ax0_1 = 0; ax0_1 < 4; ++ax0_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(B_shared[(((((int)threadIdx.z) * 2048) + (ax0_1 * 512)) + (k_1 * 256))])) + (((int)threadIdx.x) * 8))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(B_shared[(((((int)threadIdx.z) * 2048) + (ax0_1 * 512)) + (k_1 * 256))])) + (((int)threadIdx.x) * 8)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_shared_warp + (ax0_1 * 8)))[0]), "=r"(((unsigned *)(B_shared_warp + (ax0_1 * 8)))[1]), "=r"(((unsigned *)(B_shared_warp + (ax0_1 * 8)))[2]), "=r"(((unsigned *)(B_shared_warp + (ax0_1 * 8)))[3])
      : "r"(addr)
    );
  }
      }
      for (int i_2 = 0; i_2 < 16; ++i_2) {
        for (int j_2 = 0; j_2 < 4; ++j_2) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(C_warp + ((i_2 * 32) + (j_2 * 8))))[0]), "=r"(((unsigned *)(C_warp + ((i_2 * 32) + (j_2 * 8))))[1])
      : "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + (j_2 * 8)))[0]), "r"(((unsigned *)(B_shared_warp + (j_2 * 8)))[1]), "r"(((unsigned *)(C_warp + ((i_2 * 32) + (j_2 * 8))))[0]), "r"(((unsigned *)(C_warp + ((i_2 * 32) + (j_2 * 8))))[1]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(C_warp + (((i_2 * 32) + (j_2 * 8)) + 4)))[0]), "=r"(((unsigned *)(C_warp + (((i_2 * 32) + (j_2 * 8)) + 4)))[1])
      : "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + ((j_2 * 8) + 4)))[0]), "r"(((unsigned *)(B_shared_warp + ((j_2 * 8) + 4)))[1]), "r"(((unsigned *)(C_warp + (((i_2 * 32) + (j_2 * 8)) + 4)))[0]), "r"(((unsigned *)(C_warp + (((i_2 * 32) + (j_2 * 8)) + 4)))[1]));
  }
        }
      }
    }
  }
  for (int ax0_2 = 0; ax0_2 < 16; ++ax0_2) {
    for (int ax1 = 0; ax1 < 4; ++ax1) {
      for (int local_id = 0; local_id < 8; local_id+=2) {
*((uint *)&(&(C[((((((((int)blockIdx.y) * 524288) + (((int)threadIdx.y) * 262144)) + (ax0_2 * 16384)) + (((int)blockIdx.x) * 2048)) + (((int)threadIdx.z) * 1024)) + (ax1 * 256))]))[((((((local_id % 4) / 2) * 8) + (threadIdx.x / 4)) * 16) + ((((local_id / 4) * 8) + ((threadIdx.x % 4) * 2)) + (local_id % 2)))]) = *((uint *)&C_warp[((ax0_2 * 32) + (ax1 * 8)) + local_id]);
}
;
    }
  }
}

