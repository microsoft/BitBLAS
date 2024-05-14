__global__ void __launch_bounds__(128) Fused(int8_t* __restrict__ A, int8_t* __restrict__ B, int8_t* __restrict__ d_transform) {
  
  int C_shared_warp[144];
  __shared__ signed char A_shared[6144];
  __shared__ signed char B_shared[49152];
  signed char A_shared_warp[48];
  signed char B_shared_warp[96];
  signed char A_shared_warp_1[48];
  signed char B_shared_warp_1[96];
  __shared__ int C_shared[4864];
  for (int i_2_init = 0; i_2_init < 3; ++i_2_init) {
    for (int j_2_init = 0; j_2_init < 6; ++j_2_init) {
      for (int i = 0; i < 8; ++i) {
C_shared_warp[((i_2_init * 48) + (j_2_init * 8)) + i] = 0.0;}
;
    }
  }
  #pragma unroll
  for (int ax0_ax1_ax2_ax3_0_fused_0 = 0; ax0_ax1_ax2_ax3_0_fused_0 < 2; ++ax0_ax1_ax2_ax3_0_fused_0) {
    if (((ax0_ax1_ax2_ax3_0_fused_0 * 2) + (((int)threadIdx.z) >> 1)) < 3) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + (((ax0_ax1_ax2_ax3_0_fused_0 * 2048) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + (((ax0_ax1_ax2_ax3_0_fused_0 * 2048) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((ax0_ax1_ax2_ax3_0_fused_0 * 24576) + ((((int)threadIdx.z) >> 1) * 12288)) + ((((int)threadIdx.z) & 1) * 512)) + (((int)threadIdx.x) * 16)))), "n"(16)
    );
  }
    }
  }
  #pragma unroll
  for (int ax0_ax1_ax2_ax3_0_fused_0_1 = 0; ax0_ax1_ax2_ax3_0_fused_0_1 < 12; ++ax0_ax1_ax2_ax3_0_fused_0_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(B_shared + (((ax0_ax1_ax2_ax3_0_fused_0_1 * 2048) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + (((ax0_ax1_ax2_ax3_0_fused_0_1 * 2048) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((ax0_ax1_ax2_ax3_0_fused_0_1 * 24576) + ((((int)threadIdx.z) >> 1) * 12288)) + ((((int)threadIdx.z) & 1) * 512)) + (((int)threadIdx.x) * 16)))), "n"(16)
    );
  }
  }
__asm__ __volatile__("cp.async.commit_group;");

  #pragma unroll
  for (int k_0 = 0; k_0 < 11; ++k_0) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_ax2_ax3_0_fused_0_2 = 0; ax0_ax1_ax2_ax3_0_fused_0_2 < 2; ++ax0_ax1_ax2_ax3_0_fused_0_2) {
      if (((ax0_ax1_ax2_ax3_0_fused_0_2 * 2) + (((int)threadIdx.z) >> 1)) < 3) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_shared + ((((((k_0 + 1) & 1) * 3072) + (ax0_ax1_ax2_ax3_0_fused_0_2 * 2048)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_shared + ((((((k_0 + 1) & 1) * 3072) + (ax0_ax1_ax2_ax3_0_fused_0_2 * 2048)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((ax0_ax1_ax2_ax3_0_fused_0_2 * 24576) + ((((int)threadIdx.z) >> 1) * 12288)) + (k_0 * 1024)) + ((((int)threadIdx.z) & 1) * 512)) + (((int)threadIdx.x) * 16)) + 1024))), "n"(16)
    );
  }
      }
    }
    #pragma unroll
    for (int ax0_ax1_ax2_ax3_0_fused_0_3 = 0; ax0_ax1_ax2_ax3_0_fused_0_3 < 12; ++ax0_ax1_ax2_ax3_0_fused_0_3) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(B_shared + ((((((k_0 + 1) & 1) * 24576) + (ax0_ax1_ax2_ax3_0_fused_0_3 * 2048)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + ((((((k_0 + 1) & 1) * 24576) + (ax0_ax1_ax2_ax3_0_fused_0_3 * 2048)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 16))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + ((((((ax0_ax1_ax2_ax3_0_fused_0_3 * 24576) + ((((int)threadIdx.z) >> 1) * 12288)) + (k_0 * 1024)) + ((((int)threadIdx.z) & 1) * 512)) + (((int)threadIdx.x) * 16)) + 1024))), "n"(16)
    );
  }
    }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

    __syncthreads();
    for (int k_1 = 0; k_1 < 2; ++k_1) {
      for (int ax0 = 0; ax0 < 3; ++ax0) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(A_shared[((((k_0 & 1) * 3072) + (ax0 * 1024)) + (k_1 * 512))])) + (((int)threadIdx.x) * 16))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_shared[((((k_0 & 1) * 3072) + (ax0 * 1024)) + (k_1 * 512))])) + (((int)threadIdx.x) * 16)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_shared_warp + (ax0 * 16)))[0]), "=r"(((unsigned *)(A_shared_warp + (ax0 * 16)))[1]), "=r"(((unsigned *)(A_shared_warp + (ax0 * 16)))[2]), "=r"(((unsigned *)(A_shared_warp + (ax0 * 16)))[3])
      : "r"(addr)
    );
  }
      }
      for (int ax0_1 = 0; ax0_1 < 6; ++ax0_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(B_shared[(((((k_0 & 1) * 24576) + (((int)threadIdx.z) * 6144)) + (ax0_1 * 1024)) + (k_1 * 512))])) + (((int)threadIdx.x) * 16))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(B_shared[(((((k_0 & 1) * 24576) + (((int)threadIdx.z) * 6144)) + (ax0_1 * 1024)) + (k_1 * 512))])) + (((int)threadIdx.x) * 16)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_shared_warp + (ax0_1 * 16)))[0]), "=r"(((unsigned *)(B_shared_warp + (ax0_1 * 16)))[1]), "=r"(((unsigned *)(B_shared_warp + (ax0_1 * 16)))[2]), "=r"(((unsigned *)(B_shared_warp + (ax0_1 * 16)))[3])
      : "r"(addr)
    );
  }
      }
      for (int i_2 = 0; i_2 < 3; ++i_2) {
        for (int j_2 = 0; j_2 < 6; ++j_2) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32"
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
      :  "=r"(((int *)(C_shared_warp + ((i_2 * 48) + (j_2 * 8))))[0]), "=r"(((int *)(C_shared_warp + ((i_2 * 48) + (j_2 * 8))))[1]), "=r"(((int *)(C_shared_warp + ((i_2 * 48) + (j_2 * 8))))[2]), "=r"(((int *)(C_shared_warp + ((i_2 * 48) + (j_2 * 8))))[3])
      : "r"(((unsigned *)(A_shared_warp + (i_2 * 16)))[0]), "r"(((unsigned *)(A_shared_warp + (i_2 * 16)))[1]), "r"(((unsigned *)(B_shared_warp + (j_2 * 16)))[0]), "r"(((int *)(C_shared_warp + ((i_2 * 48) + (j_2 * 8))))[0]), "r"(((int *)(C_shared_warp + ((i_2 * 48) + (j_2 * 8))))[1]), "r"(((int *)(C_shared_warp + ((i_2 * 48) + (j_2 * 8))))[2]), "r"(((int *)(C_shared_warp + ((i_2 * 48) + (j_2 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32"
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
      :  "=r"(((int *)(C_shared_warp + ((i_2 * 48) + (j_2 * 8))))[0]), "=r"(((int *)(C_shared_warp + ((i_2 * 48) + (j_2 * 8))))[1]), "=r"(((int *)(C_shared_warp + ((i_2 * 48) + (j_2 * 8))))[2]), "=r"(((int *)(C_shared_warp + ((i_2 * 48) + (j_2 * 8))))[3])
      : "r"(((unsigned *)(A_shared_warp + ((i_2 * 16) + 8)))[0]), "r"(((unsigned *)(A_shared_warp + ((i_2 * 16) + 8)))[1]), "r"(((unsigned *)(B_shared_warp + ((j_2 * 16) + 8)))[0]), "r"(((int *)(C_shared_warp + ((i_2 * 48) + (j_2 * 8))))[0]), "r"(((int *)(C_shared_warp + ((i_2 * 48) + (j_2 * 8))))[1]), "r"(((int *)(C_shared_warp + ((i_2 * 48) + (j_2 * 8))))[2]), "r"(((int *)(C_shared_warp + ((i_2 * 48) + (j_2 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32"
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
      :  "=r"(((int *)(C_shared_warp + (((i_2 * 48) + (j_2 * 8)) + 4)))[0]), "=r"(((int *)(C_shared_warp + (((i_2 * 48) + (j_2 * 8)) + 4)))[1]), "=r"(((int *)(C_shared_warp + (((i_2 * 48) + (j_2 * 8)) + 4)))[2]), "=r"(((int *)(C_shared_warp + (((i_2 * 48) + (j_2 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_shared_warp + (i_2 * 16)))[0]), "r"(((unsigned *)(A_shared_warp + (i_2 * 16)))[1]), "r"(((unsigned *)(B_shared_warp + (j_2 * 16)))[0]), "r"(((int *)(C_shared_warp + (((i_2 * 48) + (j_2 * 8)) + 4)))[0]), "r"(((int *)(C_shared_warp + (((i_2 * 48) + (j_2 * 8)) + 4)))[1]), "r"(((int *)(C_shared_warp + (((i_2 * 48) + (j_2 * 8)) + 4)))[2]), "r"(((int *)(C_shared_warp + (((i_2 * 48) + (j_2 * 8)) + 4)))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32"
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
      :  "=r"(((int *)(C_shared_warp + (((i_2 * 48) + (j_2 * 8)) + 4)))[0]), "=r"(((int *)(C_shared_warp + (((i_2 * 48) + (j_2 * 8)) + 4)))[1]), "=r"(((int *)(C_shared_warp + (((i_2 * 48) + (j_2 * 8)) + 4)))[2]), "=r"(((int *)(C_shared_warp + (((i_2 * 48) + (j_2 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_shared_warp + ((i_2 * 16) + 8)))[0]), "r"(((unsigned *)(A_shared_warp + ((i_2 * 16) + 8)))[1]), "r"(((unsigned *)(B_shared_warp + ((j_2 * 16) + 8)))[0]), "r"(((int *)(C_shared_warp + (((i_2 * 48) + (j_2 * 8)) + 4)))[0]), "r"(((int *)(C_shared_warp + (((i_2 * 48) + (j_2 * 8)) + 4)))[1]), "r"(((int *)(C_shared_warp + (((i_2 * 48) + (j_2 * 8)) + 4)))[2]), "r"(((int *)(C_shared_warp + (((i_2 * 48) + (j_2 * 8)) + 4)))[3]));
  }
        }
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int k_1_1 = 0; k_1_1 < 2; ++k_1_1) {
    for (int ax0_2 = 0; ax0_2 < 3; ++ax0_2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(A_shared[(((ax0_2 * 1024) + (k_1_1 * 512)) + 3072)])) + (((int)threadIdx.x) * 16))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_shared[(((ax0_2 * 1024) + (k_1_1 * 512)) + 3072)])) + (((int)threadIdx.x) * 16)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_shared_warp_1 + (ax0_2 * 16)))[0]), "=r"(((unsigned *)(A_shared_warp_1 + (ax0_2 * 16)))[1]), "=r"(((unsigned *)(A_shared_warp_1 + (ax0_2 * 16)))[2]), "=r"(((unsigned *)(A_shared_warp_1 + (ax0_2 * 16)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_3 = 0; ax0_3 < 6; ++ax0_3) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(B_shared[((((((int)threadIdx.z) * 6144) + (ax0_3 * 1024)) + (k_1_1 * 512)) + 24576)])) + (((int)threadIdx.x) * 16))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(B_shared[((((((int)threadIdx.z) * 6144) + (ax0_3 * 1024)) + (k_1_1 * 512)) + 24576)])) + (((int)threadIdx.x) * 16)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_shared_warp_1 + (ax0_3 * 16)))[0]), "=r"(((unsigned *)(B_shared_warp_1 + (ax0_3 * 16)))[1]), "=r"(((unsigned *)(B_shared_warp_1 + (ax0_3 * 16)))[2]), "=r"(((unsigned *)(B_shared_warp_1 + (ax0_3 * 16)))[3])
      : "r"(addr)
    );
  }
    }
    for (int i_2_1 = 0; i_2_1 < 3; ++i_2_1) {
      for (int j_2_1 = 0; j_2_1 < 6; ++j_2_1) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32"
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
      :  "=r"(((int *)(C_shared_warp + ((i_2_1 * 48) + (j_2_1 * 8))))[0]), "=r"(((int *)(C_shared_warp + ((i_2_1 * 48) + (j_2_1 * 8))))[1]), "=r"(((int *)(C_shared_warp + ((i_2_1 * 48) + (j_2_1 * 8))))[2]), "=r"(((int *)(C_shared_warp + ((i_2_1 * 48) + (j_2_1 * 8))))[3])
      : "r"(((unsigned *)(A_shared_warp_1 + (i_2_1 * 16)))[0]), "r"(((unsigned *)(A_shared_warp_1 + (i_2_1 * 16)))[1]), "r"(((unsigned *)(B_shared_warp_1 + (j_2_1 * 16)))[0]), "r"(((int *)(C_shared_warp + ((i_2_1 * 48) + (j_2_1 * 8))))[0]), "r"(((int *)(C_shared_warp + ((i_2_1 * 48) + (j_2_1 * 8))))[1]), "r"(((int *)(C_shared_warp + ((i_2_1 * 48) + (j_2_1 * 8))))[2]), "r"(((int *)(C_shared_warp + ((i_2_1 * 48) + (j_2_1 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32"
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
      :  "=r"(((int *)(C_shared_warp + ((i_2_1 * 48) + (j_2_1 * 8))))[0]), "=r"(((int *)(C_shared_warp + ((i_2_1 * 48) + (j_2_1 * 8))))[1]), "=r"(((int *)(C_shared_warp + ((i_2_1 * 48) + (j_2_1 * 8))))[2]), "=r"(((int *)(C_shared_warp + ((i_2_1 * 48) + (j_2_1 * 8))))[3])
      : "r"(((unsigned *)(A_shared_warp_1 + ((i_2_1 * 16) + 8)))[0]), "r"(((unsigned *)(A_shared_warp_1 + ((i_2_1 * 16) + 8)))[1]), "r"(((unsigned *)(B_shared_warp_1 + ((j_2_1 * 16) + 8)))[0]), "r"(((int *)(C_shared_warp + ((i_2_1 * 48) + (j_2_1 * 8))))[0]), "r"(((int *)(C_shared_warp + ((i_2_1 * 48) + (j_2_1 * 8))))[1]), "r"(((int *)(C_shared_warp + ((i_2_1 * 48) + (j_2_1 * 8))))[2]), "r"(((int *)(C_shared_warp + ((i_2_1 * 48) + (j_2_1 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32"
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
      :  "=r"(((int *)(C_shared_warp + (((i_2_1 * 48) + (j_2_1 * 8)) + 4)))[0]), "=r"(((int *)(C_shared_warp + (((i_2_1 * 48) + (j_2_1 * 8)) + 4)))[1]), "=r"(((int *)(C_shared_warp + (((i_2_1 * 48) + (j_2_1 * 8)) + 4)))[2]), "=r"(((int *)(C_shared_warp + (((i_2_1 * 48) + (j_2_1 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_shared_warp_1 + (i_2_1 * 16)))[0]), "r"(((unsigned *)(A_shared_warp_1 + (i_2_1 * 16)))[1]), "r"(((unsigned *)(B_shared_warp_1 + (j_2_1 * 16)))[0]), "r"(((int *)(C_shared_warp + (((i_2_1 * 48) + (j_2_1 * 8)) + 4)))[0]), "r"(((int *)(C_shared_warp + (((i_2_1 * 48) + (j_2_1 * 8)) + 4)))[1]), "r"(((int *)(C_shared_warp + (((i_2_1 * 48) + (j_2_1 * 8)) + 4)))[2]), "r"(((int *)(C_shared_warp + (((i_2_1 * 48) + (j_2_1 * 8)) + 4)))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32"
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
      :  "=r"(((int *)(C_shared_warp + (((i_2_1 * 48) + (j_2_1 * 8)) + 4)))[0]), "=r"(((int *)(C_shared_warp + (((i_2_1 * 48) + (j_2_1 * 8)) + 4)))[1]), "=r"(((int *)(C_shared_warp + (((i_2_1 * 48) + (j_2_1 * 8)) + 4)))[2]), "=r"(((int *)(C_shared_warp + (((i_2_1 * 48) + (j_2_1 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_shared_warp_1 + ((i_2_1 * 16) + 8)))[0]), "r"(((unsigned *)(A_shared_warp_1 + ((i_2_1 * 16) + 8)))[1]), "r"(((unsigned *)(B_shared_warp_1 + ((j_2_1 * 16) + 8)))[0]), "r"(((int *)(C_shared_warp + (((i_2_1 * 48) + (j_2_1 * 8)) + 4)))[0]), "r"(((int *)(C_shared_warp + (((i_2_1 * 48) + (j_2_1 * 8)) + 4)))[1]), "r"(((int *)(C_shared_warp + (((i_2_1 * 48) + (j_2_1 * 8)) + 4)))[2]), "r"(((int *)(C_shared_warp + (((i_2_1 * 48) + (j_2_1 * 8)) + 4)))[3]));
  }
      }
    }
  }
  for (int ax0_4 = 0; ax0_4 < 3; ++ax0_4) {
    for (int ax1 = 0; ax1 < 6; ++ax1) {
      __syncthreads();
      for (int local_id = 0; local_id < 8; ++local_id) {
(&(C_shared[(((int)threadIdx.z) * 1536)]))[((((((local_id % 4) / 2) * 8) + (threadIdx.x / 4)) * 16) + ((((local_id / 4) * 8) + ((threadIdx.x % 4) * 2)) + (local_id % 2)))] = C_shared_warp[((ax0_4 * 48) + (ax1 * 8)) + local_id];
}
;
      __syncthreads();
      #pragma unroll
      for (int ax0_ax1_ax2_ax3_fused_0 = 0; ax0_ax1_ax2_ax3_fused_0 < 1; ++ax0_ax1_ax2_ax3_fused_0) {
        int2 __1;
        longlong4 __2 = *(longlong4*)(C_shared + ((((int)threadIdx.z) * 1536) + (((int)threadIdx.x) * 8)));
        __1.x=((signed char)(((int2*)(&(__2.x)))->x) << 0);
        __1.x=__1.x & ~(0x000000ff << 8) |((signed char)(((int2*)(&(__2.x)))->y) << 8);
        __1.x=__1.x & ~(0x000000ff << 16) |((signed char)(((int2*)(&(__2.y)))->x) << 16);
        __1.x=__1.x & ~(0x000000ff << 24) |((signed char)(((int2*)(&(__2.y)))->y) << 24);
        __1.y=__1.y & ~(0x000000ff << 0) |((signed char)(((int2*)(&(__2.z)))->x) << 0);
        __1.y=__1.y & ~(0x000000ff << 8) |((signed char)(((int2*)(&(__2.z)))->y) << 8);
        __1.y=__1.y & ~(0x000000ff << 16) |((signed char)(((int2*)(&(__2.w)))->x) << 16);
        __1.y=__1.y & ~(0x000000ff << 24) |((signed char)(((int2*)(&(__2.w)))->y) << 24);
        *(int2*)(d_transform + ((((ax0_4 * 6144) + (((int)threadIdx.z) * 1536)) + (ax1 * 256)) + (((int)threadIdx.x) * 8))) = __1;
      }
    }
  }
}

