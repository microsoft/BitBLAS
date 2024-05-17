__global__ void __launch_bounds__(64) Fused(int8_t* __restrict__ A, int8_t* __restrict__ B, int8_t* __restrict__ d_transform) {
  
  int C_shared_warp[8];
  __shared__ signed char A_shared[2048];
  __shared__ signed char B_shared[4096];
  signed char A_shared_warp[16];
  signed char B_shared_warp[16];
  signed char A_shared_warp_1[16];
  signed char B_shared_warp_1[16];
  __shared__ int C_shared[512];

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
    for (int j_2_init = 0; j_2_init < 1; ++j_2_init) {
      for (int i = 0; i < 8; ++i) {
C_shared_warp[0 + i] = 0.0;}
;
    }
  }
  #pragma unroll
  for (int ax0_ax1_ax2_ax3_0_fused_0 = 0; ax0_ax1_ax2_ax3_0_fused_0 < 1; ++ax0_ax1_ax2_ax3_0_fused_0) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + ((((int)threadIdx.z) * 512) + (((int)threadIdx.x) * 16)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + ((((int)threadIdx.z) * 512) + (((int)threadIdx.x) * 16))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + (((((int)blockIdx.y) * 229376) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16)))), "n"(16)
    );
  }
  }
  #pragma unroll
  for (int ax0_ax1_ax2_ax3_0_fused_0_1 = 0; ax0_ax1_ax2_ax3_0_fused_0_1 < 2; ++ax0_ax1_ax2_ax3_0_fused_0_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(B_shared + (((ax0_ax1_ax2_ax3_0_fused_0_1 * 1024) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + (((ax0_ax1_ax2_ax3_0_fused_0_1 * 1024) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((int)blockIdx.x) * 458752) + (ax0_ax1_ax2_ax3_0_fused_0_1 * 229376)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16)))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int k_0 = 0; k_0 < 223; ++k_0) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_ax2_ax3_0_fused_0_2 = 0; ax0_ax1_ax2_ax3_0_fused_0_2 < 1; ++ax0_ax1_ax2_ax3_0_fused_0_2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + (((((k_0 + 1) & 1) * 1024) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + (((((k_0 + 1) & 1) * 1024) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + (((((((int)blockIdx.y) * 229376) + (k_0 * 1024)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16)) + 1024))), "n"(16)
    );
  }
    }
    #pragma unroll
    for (int ax0_ax1_ax2_ax3_0_fused_0_3 = 0; ax0_ax1_ax2_ax3_0_fused_0_3 < 2; ++ax0_ax1_ax2_ax3_0_fused_0_3) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(B_shared + ((((((k_0 + 1) & 1) * 2048) + (ax0_ax1_ax2_ax3_0_fused_0_3 * 1024)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + ((((((k_0 + 1) & 1) * 2048) + (ax0_ax1_ax2_ax3_0_fused_0_3 * 1024)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((int)blockIdx.x) * 458752) + (ax0_ax1_ax2_ax3_0_fused_0_3 * 229376)) + (k_0 * 1024)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16)) + 1024))), "n"(16)
    );
  }
    }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

    __syncthreads();
    for (int k_1 = 0; k_1 < 2; ++k_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(A_shared[(((k_0 & 1) * 1024) + (k_1 * 512))])) + (((int)threadIdx.x) * 16))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_shared[(((k_0 & 1) * 1024) + (k_1 * 512))])) + (((int)threadIdx.x) * 16)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_shared_warp + 0))[0]), "=r"(((unsigned *)(A_shared_warp + 0))[1]), "=r"(((unsigned *)(A_shared_warp + 0))[2]), "=r"(((unsigned *)(A_shared_warp + 0))[3])
      : "r"(addr)
    );
  }

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(B_shared[((((k_0 & 1) * 2048) + (((int)threadIdx.z) * 1024)) + (k_1 * 512))])) + (((int)threadIdx.x) * 16))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(B_shared[((((k_0 & 1) * 2048) + (((int)threadIdx.z) * 1024)) + (k_1 * 512))])) + (((int)threadIdx.x) * 16)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_shared_warp + 0))[0]), "=r"(((unsigned *)(B_shared_warp + 0))[1]), "=r"(((unsigned *)(B_shared_warp + 0))[2]), "=r"(((unsigned *)(B_shared_warp + 0))[3])
      : "r"(addr)
    );
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32"
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
      :  "=r"(((int *)(C_shared_warp + 0))[0]), "=r"(((int *)(C_shared_warp + 0))[1]), "=r"(((int *)(C_shared_warp + 0))[2]), "=r"(((int *)(C_shared_warp + 0))[3])
      : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(B_shared_warp + 0))[0]), "r"(((int *)(C_shared_warp + 0))[0]), "r"(((int *)(C_shared_warp + 0))[1]), "r"(((int *)(C_shared_warp + 0))[2]), "r"(((int *)(C_shared_warp + 0))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32"
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
      :  "=r"(((int *)(C_shared_warp + 0))[0]), "=r"(((int *)(C_shared_warp + 0))[1]), "=r"(((int *)(C_shared_warp + 0))[2]), "=r"(((int *)(C_shared_warp + 0))[3])
      : "r"(((unsigned *)(A_shared_warp + 8))[0]), "r"(((unsigned *)(A_shared_warp + 8))[1]), "r"(((unsigned *)(B_shared_warp + 8))[0]), "r"(((int *)(C_shared_warp + 0))[0]), "r"(((int *)(C_shared_warp + 0))[1]), "r"(((int *)(C_shared_warp + 0))[2]), "r"(((int *)(C_shared_warp + 0))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32"
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
      :  "=r"(((int *)(C_shared_warp + 4))[0]), "=r"(((int *)(C_shared_warp + 4))[1]), "=r"(((int *)(C_shared_warp + 4))[2]), "=r"(((int *)(C_shared_warp + 4))[3])
      : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(B_shared_warp + 0))[0]), "r"(((int *)(C_shared_warp + 4))[0]), "r"(((int *)(C_shared_warp + 4))[1]), "r"(((int *)(C_shared_warp + 4))[2]), "r"(((int *)(C_shared_warp + 4))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32"
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
      :  "=r"(((int *)(C_shared_warp + 4))[0]), "=r"(((int *)(C_shared_warp + 4))[1]), "=r"(((int *)(C_shared_warp + 4))[2]), "=r"(((int *)(C_shared_warp + 4))[3])
      : "r"(((unsigned *)(A_shared_warp + 8))[0]), "r"(((unsigned *)(A_shared_warp + 8))[1]), "r"(((unsigned *)(B_shared_warp + 8))[0]), "r"(((int *)(C_shared_warp + 4))[0]), "r"(((int *)(C_shared_warp + 4))[1]), "r"(((int *)(C_shared_warp + 4))[2]), "r"(((int *)(C_shared_warp + 4))[3]));
  }
    }
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int k_1_1 = 0; k_1_1 < 2; ++k_1_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(A_shared[((k_1_1 * 512) + 1024)])) + (((int)threadIdx.x) * 16))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_shared[((k_1_1 * 512) + 1024)])) + (((int)threadIdx.x) * 16)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_shared_warp_1 + 0))[0]), "=r"(((unsigned *)(A_shared_warp_1 + 0))[1]), "=r"(((unsigned *)(A_shared_warp_1 + 0))[2]), "=r"(((unsigned *)(A_shared_warp_1 + 0))[3])
      : "r"(addr)
    );
  }

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(B_shared[(((((int)threadIdx.z) * 1024) + (k_1_1 * 512)) + 2048)])) + (((int)threadIdx.x) * 16))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(B_shared[(((((int)threadIdx.z) * 1024) + (k_1_1 * 512)) + 2048)])) + (((int)threadIdx.x) * 16)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_shared_warp_1 + 0))[0]), "=r"(((unsigned *)(B_shared_warp_1 + 0))[1]), "=r"(((unsigned *)(B_shared_warp_1 + 0))[2]), "=r"(((unsigned *)(B_shared_warp_1 + 0))[3])
      : "r"(addr)
    );
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32"
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
      :  "=r"(((int *)(C_shared_warp + 0))[0]), "=r"(((int *)(C_shared_warp + 0))[1]), "=r"(((int *)(C_shared_warp + 0))[2]), "=r"(((int *)(C_shared_warp + 0))[3])
      : "r"(((unsigned *)(A_shared_warp_1 + 0))[0]), "r"(((unsigned *)(A_shared_warp_1 + 0))[1]), "r"(((unsigned *)(B_shared_warp_1 + 0))[0]), "r"(((int *)(C_shared_warp + 0))[0]), "r"(((int *)(C_shared_warp + 0))[1]), "r"(((int *)(C_shared_warp + 0))[2]), "r"(((int *)(C_shared_warp + 0))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32"
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
      :  "=r"(((int *)(C_shared_warp + 0))[0]), "=r"(((int *)(C_shared_warp + 0))[1]), "=r"(((int *)(C_shared_warp + 0))[2]), "=r"(((int *)(C_shared_warp + 0))[3])
      : "r"(((unsigned *)(A_shared_warp_1 + 8))[0]), "r"(((unsigned *)(A_shared_warp_1 + 8))[1]), "r"(((unsigned *)(B_shared_warp_1 + 8))[0]), "r"(((int *)(C_shared_warp + 0))[0]), "r"(((int *)(C_shared_warp + 0))[1]), "r"(((int *)(C_shared_warp + 0))[2]), "r"(((int *)(C_shared_warp + 0))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32"
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
      :  "=r"(((int *)(C_shared_warp + 4))[0]), "=r"(((int *)(C_shared_warp + 4))[1]), "=r"(((int *)(C_shared_warp + 4))[2]), "=r"(((int *)(C_shared_warp + 4))[3])
      : "r"(((unsigned *)(A_shared_warp_1 + 0))[0]), "r"(((unsigned *)(A_shared_warp_1 + 0))[1]), "r"(((unsigned *)(B_shared_warp_1 + 0))[0]), "r"(((int *)(C_shared_warp + 4))[0]), "r"(((int *)(C_shared_warp + 4))[1]), "r"(((int *)(C_shared_warp + 4))[2]), "r"(((int *)(C_shared_warp + 4))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32"
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
      :  "=r"(((int *)(C_shared_warp + 4))[0]), "=r"(((int *)(C_shared_warp + 4))[1]), "=r"(((int *)(C_shared_warp + 4))[2]), "=r"(((int *)(C_shared_warp + 4))[3])
      : "r"(((unsigned *)(A_shared_warp_1 + 8))[0]), "r"(((unsigned *)(A_shared_warp_1 + 8))[1]), "r"(((unsigned *)(B_shared_warp_1 + 8))[0]), "r"(((int *)(C_shared_warp + 4))[0]), "r"(((int *)(C_shared_warp + 4))[1]), "r"(((int *)(C_shared_warp + 4))[2]), "r"(((int *)(C_shared_warp + 4))[3]));
  }
  }
  for (int local_id = 0; local_id < 8; ++local_id) {
(&(C_shared[(((int)threadIdx.z) * 256)]))[((((((local_id % 4) / 2) * 8) + (threadIdx.x / 4)) * 16) + ((((local_id / 4) * 8) + ((threadIdx.x % 4) * 2)) + (local_id % 2)))] = C_shared_warp[0 + local_id];
}
;
  __syncthreads();
  #pragma unroll
  for (int ax0_ax1_ax2_ax3_fused_0 = 0; ax0_ax1_ax2_ax3_fused_0 < 1; ++ax0_ax1_ax2_ax3_fused_0) {
    int2 __1;
    longlong4 __2 = *(longlong4*)(C_shared + ((((int)threadIdx.z) * 256) + (((int)threadIdx.x) * 8)));
    __1.x=((signed char)(((int2*)(&(__2.x)))->x) << 0);
    __1.x=__1.x & ~(0x000000ff << 8) |((signed char)(((int2*)(&(__2.x)))->y) << 8);
    __1.x=__1.x & ~(0x000000ff << 16) |((signed char)(((int2*)(&(__2.y)))->x) << 16);
    __1.x=__1.x & ~(0x000000ff << 24) |((signed char)(((int2*)(&(__2.y)))->y) << 24);
    __1.y=__1.y & ~(0x000000ff << 0) |((signed char)(((int2*)(&(__2.z)))->x) << 0);
    __1.y=__1.y & ~(0x000000ff << 8) |((signed char)(((int2*)(&(__2.z)))->y) << 8);
    __1.y=__1.y & ~(0x000000ff << 16) |((signed char)(((int2*)(&(__2.w)))->x) << 16);
    __1.y=__1.y & ~(0x000000ff << 24) |((signed char)(((int2*)(&(__2.w)))->y) << 24);
    *(int2*)(d_transform + ((((((int)blockIdx.y) * 131072) + (((int)blockIdx.x) * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8))) = __1;
  }
}

