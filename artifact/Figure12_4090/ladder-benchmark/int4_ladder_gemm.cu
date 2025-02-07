__global__ void __launch_bounds__(128) Fused(int8_t* __restrict__ A, int8_t* __restrict__ B, int8_t* __restrict__ d_transform) {
  
  int C_shared_warp[16];
  __shared__ signed char A_shared[2048];
  __shared__ signed char B_shared[4096];
  signed char A_shared_warp[16];
  signed char B_shared_warp[32];
  __shared__ int C_shared[1536];

  const int MAX_BLOCK_N = 11;
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
C_shared_warp[(j_2_init * 8) + i] = 0.0;}
;
    }
  }
  for (int k_0 = 0; k_0 < 224; ++k_0) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_ax2_ax3_0_fused_0 = 0; ax0_ax1_ax2_ax3_0_fused_0 < 1; ++ax0_ax1_ax2_ax3_0_fused_0) {
      *(int4*)(A_shared + (((((int)threadIdx.y) * 1024) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(A + (((((((int)blockIdx.y) * 458752) + (((int)threadIdx.y) * 229376)) + (k_0 * 1024)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16)));
    }
    #pragma unroll
    for (int ax0_ax1_ax2_ax3_0_fused_0_1 = 0; ax0_ax1_ax2_ax3_0_fused_0_1 < 2; ++ax0_ax1_ax2_ax3_0_fused_0_1) {
      *(int4*)(B_shared + ((((ax0_ax1_ax2_ax3_0_fused_0_1 * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16))) = *(int4*)(B + ((((((((int)blockIdx.x) * 917504) + (ax0_ax1_ax2_ax3_0_fused_0_1 * 458752)) + (((int)threadIdx.y) * 229376)) + (k_0 * 1024)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16)));
    }
    __syncthreads();
    for (int k_1 = 0; k_1 < 2; ++k_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(A_shared[((((int)threadIdx.y) * 1024) + (k_1 * 512))])) + (((int)threadIdx.x) * 16))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_shared[((((int)threadIdx.y) * 1024) + (k_1 * 512))])) + (((int)threadIdx.x) * 16)))
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
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32"
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
      :  "=r"(((int *)(C_shared_warp + (j_2 * 8)))[0]), "=r"(((int *)(C_shared_warp + (j_2 * 8)))[1]), "=r"(((int *)(C_shared_warp + (j_2 * 8)))[2]), "=r"(((int *)(C_shared_warp + (j_2 * 8)))[3])
      : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(B_shared_warp + (j_2 * 16)))[0]), "r"(((int *)(C_shared_warp + (j_2 * 8)))[0]), "r"(((int *)(C_shared_warp + (j_2 * 8)))[1]), "r"(((int *)(C_shared_warp + (j_2 * 8)))[2]), "r"(((int *)(C_shared_warp + (j_2 * 8)))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32"
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
      :  "=r"(((int *)(C_shared_warp + (j_2 * 8)))[0]), "=r"(((int *)(C_shared_warp + (j_2 * 8)))[1]), "=r"(((int *)(C_shared_warp + (j_2 * 8)))[2]), "=r"(((int *)(C_shared_warp + (j_2 * 8)))[3])
      : "r"(((unsigned *)(A_shared_warp + 8))[0]), "r"(((unsigned *)(A_shared_warp + 8))[1]), "r"(((unsigned *)(B_shared_warp + ((j_2 * 16) + 8)))[0]), "r"(((int *)(C_shared_warp + (j_2 * 8)))[0]), "r"(((int *)(C_shared_warp + (j_2 * 8)))[1]), "r"(((int *)(C_shared_warp + (j_2 * 8)))[2]), "r"(((int *)(C_shared_warp + (j_2 * 8)))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32"
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
      :  "=r"(((int *)(C_shared_warp + ((j_2 * 8) + 4)))[0]), "=r"(((int *)(C_shared_warp + ((j_2 * 8) + 4)))[1]), "=r"(((int *)(C_shared_warp + ((j_2 * 8) + 4)))[2]), "=r"(((int *)(C_shared_warp + ((j_2 * 8) + 4)))[3])
      : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(B_shared_warp + (j_2 * 16)))[0]), "r"(((int *)(C_shared_warp + ((j_2 * 8) + 4)))[0]), "r"(((int *)(C_shared_warp + ((j_2 * 8) + 4)))[1]), "r"(((int *)(C_shared_warp + ((j_2 * 8) + 4)))[2]), "r"(((int *)(C_shared_warp + ((j_2 * 8) + 4)))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32"
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
      :  "=r"(((int *)(C_shared_warp + ((j_2 * 8) + 4)))[0]), "=r"(((int *)(C_shared_warp + ((j_2 * 8) + 4)))[1]), "=r"(((int *)(C_shared_warp + ((j_2 * 8) + 4)))[2]), "=r"(((int *)(C_shared_warp + ((j_2 * 8) + 4)))[3])
      : "r"(((unsigned *)(A_shared_warp + 8))[0]), "r"(((unsigned *)(A_shared_warp + 8))[1]), "r"(((unsigned *)(B_shared_warp + ((j_2 * 16) + 8)))[0]), "r"(((int *)(C_shared_warp + ((j_2 * 8) + 4)))[0]), "r"(((int *)(C_shared_warp + ((j_2 * 8) + 4)))[1]), "r"(((int *)(C_shared_warp + ((j_2 * 8) + 4)))[2]), "r"(((int *)(C_shared_warp + ((j_2 * 8) + 4)))[3]));
  }
      }
    }
  }
  for (int ax1 = 0; ax1 < 2; ++ax1) {
    __syncthreads();
    for (int local_id = 0; local_id < 8; ++local_id) {
(&(C_shared[((((int)threadIdx.y) * 768) + (((int)threadIdx.z) * 512))]))[((((((local_id % 4) / 2) * 8) + (threadIdx.x / 4)) * 16) + ((((local_id / 4) * 8) + ((threadIdx.x % 4) * 2)) + (local_id % 2)))] = C_shared_warp[(ax1 * 8) + local_id];
}
;
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_ax2_ax3_fused_0 = 0; ax0_ax1_ax2_ax3_fused_0 < 1; ++ax0_ax1_ax2_ax3_fused_0) {
      int2 __1;
      longlong4 __2 = *(longlong4*)(C_shared + (((((int)threadIdx.y) * 768) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 8)));
      __1.x=((signed char)(((int2*)(&(__2.x)))->x) << 0);
      __1.x=__1.x & ~(0x000000ff << 8) |((signed char)(((int2*)(&(__2.x)))->y) << 8);
      __1.x=__1.x & ~(0x000000ff << 16) |((signed char)(((int2*)(&(__2.y)))->x) << 16);
      __1.x=__1.x & ~(0x000000ff << 24) |((signed char)(((int2*)(&(__2.y)))->y) << 24);
      __1.y=__1.y & ~(0x000000ff << 0) |((signed char)(((int2*)(&(__2.z)))->x) << 0);
      __1.y=__1.y & ~(0x000000ff << 8) |((signed char)(((int2*)(&(__2.z)))->y) << 8);
      __1.y=__1.y & ~(0x000000ff << 16) |((signed char)(((int2*)(&(__2.w)))->x) << 16);
      __1.y=__1.y & ~(0x000000ff << 24) |((signed char)(((int2*)(&(__2.w)))->y) << 24);
      *(int2*)(d_transform + ((((((((int)blockIdx.y) * 262144) + (((int)threadIdx.y) * 131072)) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.z) * 512)) + (ax1 * 256)) + (((int)threadIdx.x) * 8))) = __1;
    }
  }
}

