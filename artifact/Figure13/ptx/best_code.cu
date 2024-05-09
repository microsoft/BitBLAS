__global__ void __launch_bounds__(128) Fused(int8_t* __restrict__ A, int8_t* __restrict__ B, uint8_t* __restrict__ AScales, uint8_t* __restrict__ BScales, float* __restrict__ C) {
  
  float C_warp[32];
  __shared__ signed char A_shared[2048];
  __shared__ signed char B_shared[16384];
  __shared__ half A_decode_shared[512];
  __shared__ half B_decode_shared[8192];
  signed char A_shared_local[8];
  half A_decode_local[8];
  signed char B_shared_local[8];
  half B_decode_local[8];
  half A_decode_shared_warp[8];
  half B_decode_shared_warp[32];
  signed char A_shared_local_1[8];
  half A_decode_local_1[8];
  signed char B_shared_local_1[8];
  half B_decode_local_1[8];
  half A_decode_shared_warp_1[8];
  half B_decode_shared_warp_1[32];

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
    for (int j_2_init = 0; j_2_init < 4; ++j_2_init) {
      for (int i = 0; i < 8; ++i) {
C_warp[(j_2_init * 8) + i] = 0.0;}
;
    }
  }
  for (int ax0_ax1_ax2_ax3_fused_0_0_0 = 0; ax0_ax1_ax2_ax3_fused_0_0_0 < 2; ++ax0_ax1_ax2_ax3_fused_0_0_0) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + ((ax0_ax1_ax2_ax3_fused_0_0_0 * 512) + (((int)threadIdx.x) * 16)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + ((ax0_ax1_ax2_ax3_fused_0_0_0 * 512) + (((int)threadIdx.x) * 16))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + (((((int)blockIdx.y) * 458752) + (ax0_ax1_ax2_ax3_fused_0_0_0 * 512)) + (((int)threadIdx.x) * 16)))), "n"(16)
    );
  }
  }
  for (int ax0_ax1_ax2_ax3_fused_0_0_0_0 = 0; ax0_ax1_ax2_ax3_fused_0_0_0_0 < 4; ++ax0_ax1_ax2_ax3_fused_0_0_0_0) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(B_shared + (((ax0_ax1_ax2_ax3_fused_0_0_0_0 * 2048) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + (((ax0_ax1_ax2_ax3_fused_0_0_0_0 * 2048) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((int)blockIdx.x) * 7340032) + (ax0_ax1_ax2_ax3_fused_0_0_0_0 * 1835008)) + (((int)threadIdx.z) * 458752)) + (((int)threadIdx.x) * 16)))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int k_0 = 0; k_0 < 895; ++k_0) {
    for (int ax0_ax1_ax2_ax3_fused_0_0_0_1 = 0; ax0_ax1_ax2_ax3_fused_0_0_0_1 < 2; ++ax0_ax1_ax2_ax3_fused_0_0_0_1) {
      if ((k_0 + ax0_ax1_ax2_ax3_fused_0_0_0_1) < 895) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + (((((k_0 + 1) & 1) * 1024) + (ax0_ax1_ax2_ax3_fused_0_0_0_1 * 512)) + (((int)threadIdx.x) * 16)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + (((((k_0 + 1) & 1) * 1024) + (ax0_ax1_ax2_ax3_fused_0_0_0_1 * 512)) + (((int)threadIdx.x) * 16))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + (((((((int)blockIdx.y) * 458752) + (k_0 * 512)) + (ax0_ax1_ax2_ax3_fused_0_0_0_1 * 512)) + (((int)threadIdx.x) * 16)) + 512))), "n"(16)
    );
  }
      }
    }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

    __syncthreads();
    if (((int)threadIdx.z) < 2) {
      *(int2*)(A_shared_local + 0) = *(int2*)(A_shared + ((((k_0 & 1) * 1024) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8)));
    }
    for (int ax0 = 0; ax0 < 8; ++ax0) {
      if (((int)threadIdx.z) < 2) {
          uint __1 = ((max((((((((uint)A_shared_local[ax0]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)AScales[(((k_0 * 8192) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) >> 1))])), (uint)63) | ((((((uint)A_shared_local[ax0]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)7) | (((((((uint)A_shared_local[ax0]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
        A_decode_local[ax0] = (*(half *)(&(__1)));
      }
    }
    if (((int)threadIdx.z) < 2) {
      *(uint4*)(A_decode_shared + ((((int)threadIdx.z) * 256) + (((int)threadIdx.x) * 8))) = *(uint4*)(A_decode_local + 0);
    }
    for (int ax0_ax1_ax2_ax3_fused_0_0_0_0_1 = 0; ax0_ax1_ax2_ax3_fused_0_0_0_0_1 < 4; ++ax0_ax1_ax2_ax3_fused_0_0_0_0_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(B_shared + ((((((k_0 + 1) & 1) * 8192) + (ax0_ax1_ax2_ax3_fused_0_0_0_0_1 * 2048)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + ((((((k_0 + 1) & 1) * 8192) + (ax0_ax1_ax2_ax3_fused_0_0_0_0_1 * 2048)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((((int)blockIdx.x) * 7340032) + (ax0_ax1_ax2_ax3_fused_0_0_0_0_1 * 1835008)) + (((int)threadIdx.z) * 458752)) + (k_0 * 512)) + (((int)threadIdx.x) * 16)) + 512))), "n"(16)
    );
  }
    }
__asm__ __volatile__("cp.async.commit_group;");

    __syncthreads();
    for (int ax0_ax1_ax2_ax3_0_fused_0 = 0; ax0_ax1_ax2_ax3_0_fused_0 < 8; ++ax0_ax1_ax2_ax3_0_fused_0) {
      *(int2*)(B_shared_local + 0) = *(int2*)(B_shared + (((((k_0 & 1) * 8192) + (ax0_ax1_ax2_ax3_0_fused_0 * 1024)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8)));
      for (int ax0_1 = 0; ax0_1 < 8; ++ax0_1) {
          uint __2 = ((max((((((((uint)B_shared_local[ax0_1]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)BScales[(((((k_0 * 8192) + (((int)blockIdx.x) * 256)) + (ax0_ax1_ax2_ax3_0_fused_0 * 32)) + ((((int)threadIdx.z) >> 1) * 16)) + (((int)threadIdx.x) >> 1))])), (uint)63) | ((((((uint)B_shared_local[ax0_1]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)7) | (((((((uint)B_shared_local[ax0_1]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
        B_decode_local[ax0_1] = (*(half *)(&(__2)));
      }
      *(uint4*)(B_decode_shared + (((ax0_ax1_ax2_ax3_0_fused_0 * 1024) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(B_decode_local + 0);
    }
    __syncthreads();
    for (int k_1 = 0; k_1 < 2; ++k_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(A_decode_shared[(k_1 * 256)])) + (((int)threadIdx.x) * 8))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_decode_shared[(k_1 * 256)])) + (((int)threadIdx.x) * 8)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_decode_shared_warp + 0))[0]), "=r"(((unsigned *)(A_decode_shared_warp + 0))[1]), "=r"(((unsigned *)(A_decode_shared_warp + 0))[2]), "=r"(((unsigned *)(A_decode_shared_warp + 0))[3])
      : "r"(addr)
    );
  }
      for (int ax0_2 = 0; ax0_2 < 4; ++ax0_2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(B_decode_shared[(((((int)threadIdx.z) * 2048) + (ax0_2 * 512)) + (k_1 * 256))])) + (((int)threadIdx.x) * 8))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(B_decode_shared[(((((int)threadIdx.z) * 2048) + (ax0_2 * 512)) + (k_1 * 256))])) + (((int)threadIdx.x) * 8)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_decode_shared_warp + (ax0_2 * 8)))[0]), "=r"(((unsigned *)(B_decode_shared_warp + (ax0_2 * 8)))[1]), "=r"(((unsigned *)(B_decode_shared_warp + (ax0_2 * 8)))[2]), "=r"(((unsigned *)(B_decode_shared_warp + (ax0_2 * 8)))[3])
      : "r"(addr)
    );
  }
      }
      for (int j_2 = 0; j_2 < 4; ++j_2) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(C_warp + (j_2 * 8)))[0]), "=f"(((float *)(C_warp + (j_2 * 8)))[1]), "=f"(((float *)(C_warp + (j_2 * 8)))[2]), "=f"(((float *)(C_warp + (j_2 * 8)))[3])
      : "r"(((unsigned *)(A_decode_shared_warp + 0))[0]), "r"(((unsigned *)(A_decode_shared_warp + 0))[1]), "r"(((unsigned *)(A_decode_shared_warp + 0))[2]), "r"(((unsigned *)(A_decode_shared_warp + 0))[3]), "r"(((unsigned *)(B_decode_shared_warp + (j_2 * 8)))[0]), "r"(((unsigned *)(B_decode_shared_warp + (j_2 * 8)))[1]), "f"(((float *)(C_warp + (j_2 * 8)))[0]), "f"(((float *)(C_warp + (j_2 * 8)))[1]), "f"(((float *)(C_warp + (j_2 * 8)))[2]), "f"(((float *)(C_warp + (j_2 * 8)))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(C_warp + ((j_2 * 8) + 4)))[0]), "=f"(((float *)(C_warp + ((j_2 * 8) + 4)))[1]), "=f"(((float *)(C_warp + ((j_2 * 8) + 4)))[2]), "=f"(((float *)(C_warp + ((j_2 * 8) + 4)))[3])
      : "r"(((unsigned *)(A_decode_shared_warp + 0))[0]), "r"(((unsigned *)(A_decode_shared_warp + 0))[1]), "r"(((unsigned *)(A_decode_shared_warp + 0))[2]), "r"(((unsigned *)(A_decode_shared_warp + 0))[3]), "r"(((unsigned *)(B_decode_shared_warp + ((j_2 * 8) + 4)))[0]), "r"(((unsigned *)(B_decode_shared_warp + ((j_2 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((j_2 * 8) + 4)))[0]), "f"(((float *)(C_warp + ((j_2 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((j_2 * 8) + 4)))[2]), "f"(((float *)(C_warp + ((j_2 * 8) + 4)))[3]));
  }
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  if (((int)threadIdx.z) < 2) {
    *(int2*)(A_shared_local_1 + 0) = *(int2*)(A_shared + (((((int)threadIdx.z) * 256) + (((int)threadIdx.x) * 8)) + 1024));
  }
  for (int ax0_3 = 0; ax0_3 < 8; ++ax0_3) {
    if (((int)threadIdx.z) < 2) {
        uint __3 = ((max((((((((uint)A_shared_local_1[ax0_3]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)AScales[(((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 1)) + 7331840)])), (uint)63) | ((((((uint)A_shared_local_1[ax0_3]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)7) | (((((((uint)A_shared_local_1[ax0_3]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
      A_decode_local_1[ax0_3] = (*(half *)(&(__3)));
    }
  }
  if (((int)threadIdx.z) < 2) {
    *(uint4*)(A_decode_shared + ((((int)threadIdx.z) * 256) + (((int)threadIdx.x) * 8))) = *(uint4*)(A_decode_local_1 + 0);
  }
  for (int ax0_ax1_ax2_ax3_0_fused_0_1 = 0; ax0_ax1_ax2_ax3_0_fused_0_1 < 8; ++ax0_ax1_ax2_ax3_0_fused_0_1) {
    *(int2*)(B_shared_local_1 + 0) = *(int2*)(B_shared + ((((ax0_ax1_ax2_ax3_0_fused_0_1 * 1024) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8)) + 8192));
    for (int ax0_4 = 0; ax0_4 < 8; ++ax0_4) {
        uint __4 = ((max((((((((uint)B_shared_local_1[ax0_4]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)BScales[(((((((int)blockIdx.x) * 256) + (ax0_ax1_ax2_ax3_0_fused_0_1 * 32)) + ((((int)threadIdx.z) >> 1) * 16)) + (((int)threadIdx.x) >> 1)) + 7331840)])), (uint)63) | ((((((uint)B_shared_local_1[ax0_4]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)7) | (((((((uint)B_shared_local_1[ax0_4]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
      B_decode_local_1[ax0_4] = (*(half *)(&(__4)));
    }
    *(uint4*)(B_decode_shared + (((ax0_ax1_ax2_ax3_0_fused_0_1 * 1024) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(B_decode_local_1 + 0);
  }
  __syncthreads();
  for (int k_1_1 = 0; k_1_1 < 2; ++k_1_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(A_decode_shared[(k_1_1 * 256)])) + (((int)threadIdx.x) * 8))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_decode_shared[(k_1_1 * 256)])) + (((int)threadIdx.x) * 8)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_decode_shared_warp_1 + 0))[0]), "=r"(((unsigned *)(A_decode_shared_warp_1 + 0))[1]), "=r"(((unsigned *)(A_decode_shared_warp_1 + 0))[2]), "=r"(((unsigned *)(A_decode_shared_warp_1 + 0))[3])
      : "r"(addr)
    );
  }
    for (int ax0_5 = 0; ax0_5 < 4; ++ax0_5) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(B_decode_shared[(((((int)threadIdx.z) * 2048) + (ax0_5 * 512)) + (k_1_1 * 256))])) + (((int)threadIdx.x) * 8))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(B_decode_shared[(((((int)threadIdx.z) * 2048) + (ax0_5 * 512)) + (k_1_1 * 256))])) + (((int)threadIdx.x) * 8)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_decode_shared_warp_1 + (ax0_5 * 8)))[0]), "=r"(((unsigned *)(B_decode_shared_warp_1 + (ax0_5 * 8)))[1]), "=r"(((unsigned *)(B_decode_shared_warp_1 + (ax0_5 * 8)))[2]), "=r"(((unsigned *)(B_decode_shared_warp_1 + (ax0_5 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int j_2_1 = 0; j_2_1 < 4; ++j_2_1) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(C_warp + (j_2_1 * 8)))[0]), "=f"(((float *)(C_warp + (j_2_1 * 8)))[1]), "=f"(((float *)(C_warp + (j_2_1 * 8)))[2]), "=f"(((float *)(C_warp + (j_2_1 * 8)))[3])
      : "r"(((unsigned *)(A_decode_shared_warp_1 + 0))[0]), "r"(((unsigned *)(A_decode_shared_warp_1 + 0))[1]), "r"(((unsigned *)(A_decode_shared_warp_1 + 0))[2]), "r"(((unsigned *)(A_decode_shared_warp_1 + 0))[3]), "r"(((unsigned *)(B_decode_shared_warp_1 + (j_2_1 * 8)))[0]), "r"(((unsigned *)(B_decode_shared_warp_1 + (j_2_1 * 8)))[1]), "f"(((float *)(C_warp + (j_2_1 * 8)))[0]), "f"(((float *)(C_warp + (j_2_1 * 8)))[1]), "f"(((float *)(C_warp + (j_2_1 * 8)))[2]), "f"(((float *)(C_warp + (j_2_1 * 8)))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(C_warp + ((j_2_1 * 8) + 4)))[0]), "=f"(((float *)(C_warp + ((j_2_1 * 8) + 4)))[1]), "=f"(((float *)(C_warp + ((j_2_1 * 8) + 4)))[2]), "=f"(((float *)(C_warp + ((j_2_1 * 8) + 4)))[3])
      : "r"(((unsigned *)(A_decode_shared_warp_1 + 0))[0]), "r"(((unsigned *)(A_decode_shared_warp_1 + 0))[1]), "r"(((unsigned *)(A_decode_shared_warp_1 + 0))[2]), "r"(((unsigned *)(A_decode_shared_warp_1 + 0))[3]), "r"(((unsigned *)(B_decode_shared_warp_1 + ((j_2_1 * 8) + 4)))[0]), "r"(((unsigned *)(B_decode_shared_warp_1 + ((j_2_1 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((j_2_1 * 8) + 4)))[0]), "f"(((float *)(C_warp + ((j_2_1 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((j_2_1 * 8) + 4)))[2]), "f"(((float *)(C_warp + ((j_2_1 * 8) + 4)))[3]));
  }
    }
  }
  for (int ax1 = 0; ax1 < 4; ++ax1) {
    for (int local_id = 0; local_id < 8; ++local_id) {
(&(C[((((((int)blockIdx.y) * 131072) + (((int)blockIdx.x) * 4096)) + (((int)threadIdx.z) * 1024)) + (ax1 * 256))]))[((((((local_id % 4) / 2) * 8) + (threadIdx.x / 4)) * 16) + ((((local_id / 4) * 8) + ((threadIdx.x % 4) * 2)) + (local_id % 2)))] = C_warp[(ax1 * 8) + local_id];
}
;
  }
}

