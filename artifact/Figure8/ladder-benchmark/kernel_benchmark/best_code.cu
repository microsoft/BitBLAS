__global__ void __launch_bounds__(128) Fused(half* __restrict__ A, int8_t* __restrict__ B, half* __restrict__ C) {
  
  half C_shared_warp[144];
  __shared__ half A_shared[3072];
  __shared__ half B_decode_shared[24576];
  signed char B_local[8];
  half B_decode_local[8];
  half A_shared_warp[24];
  half B_decode_shared_warp[48];
  __shared__ half C_shared[4864];
  for (int i_2_init = 0; i_2_init < 3; ++i_2_init) {
    for (int j_2_init = 0; j_2_init < 6; ++j_2_init) {
      for (int i = 0; i < 8; ++i) {
C_shared_warp[((i_2_init * 48) + (j_2_init * 8)) + i] = 0.0;}
;
    }
  }
  for (int k_0 = 0; k_0 < 24; ++k_0) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_ax2_ax3_0_fused_0 = 0; ax0_ax1_ax2_ax3_0_fused_0 < 3; ++ax0_ax1_ax2_ax3_0_fused_0) {
      *(uint4*)(A_shared + (((ax0_ax1_ax2_ax3_0_fused_0 * 1024) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(A + ((((ax0_ax1_ax2_ax3_0_fused_0 * 24576) + (k_0 * 1024)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8)));
    }
    for (int ax0_ax1_ax2_ax3_0_fused_0_1 = 0; ax0_ax1_ax2_ax3_0_fused_0_1 < 24; ++ax0_ax1_ax2_ax3_0_fused_0_1) {
      *(int2*)(B_local + 0) = *(int2*)(B + ((((ax0_ax1_ax2_ax3_0_fused_0_1 * 24576) + (k_0 * 1024)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8)));
      for (int ax0 = 0; ax0 < 8; ++ax0) {
          short __1 = ((short)B_local[ax0]) << (short)8;
        B_decode_local[ax0] = (*(half *)(&(__1)));
      }
      *(uint4*)(B_decode_shared + (((ax0_ax1_ax2_ax3_0_fused_0_1 * 1024) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(B_decode_local + 0);
    }
    __syncthreads();
    for (int k_1 = 0; k_1 < 4; ++k_1) {
      for (int ax0_1 = 0; ax0_1 < 3; ++ax0_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(A_shared[((ax0_1 * 1024) + (k_1 * 256))])) + (((int)threadIdx.x) * 8))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_shared[((ax0_1 * 1024) + (k_1 * 256))])) + (((int)threadIdx.x) * 8)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_shared_warp + (ax0_1 * 8)))[0]), "=r"(((unsigned *)(A_shared_warp + (ax0_1 * 8)))[1]), "=r"(((unsigned *)(A_shared_warp + (ax0_1 * 8)))[2]), "=r"(((unsigned *)(A_shared_warp + (ax0_1 * 8)))[3])
      : "r"(addr)
    );
  }
      }
      for (int ax0_2 = 0; ax0_2 < 6; ++ax0_2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(B_decode_shared[(((((int)threadIdx.z) * 6144) + (ax0_2 * 1024)) + (k_1 * 256))])) + (((int)threadIdx.x) * 8))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(B_decode_shared[(((((int)threadIdx.z) * 6144) + (ax0_2 * 1024)) + (k_1 * 256))])) + (((int)threadIdx.x) * 8)))
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
      for (int i_2 = 0; i_2 < 3; ++i_2) {
        for (int j_2 = 0; j_2 < 6; ++j_2) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(C_shared_warp + ((i_2 * 48) + (j_2 * 8))))[0]), "=r"(((unsigned *)(C_shared_warp + ((i_2 * 48) + (j_2 * 8))))[1])
      : "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[3]), "r"(((unsigned *)(B_decode_shared_warp + (j_2 * 8)))[0]), "r"(((unsigned *)(B_decode_shared_warp + (j_2 * 8)))[1]), "r"(((unsigned *)(C_shared_warp + ((i_2 * 48) + (j_2 * 8))))[0]), "r"(((unsigned *)(C_shared_warp + ((i_2 * 48) + (j_2 * 8))))[1]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(C_shared_warp + (((i_2 * 48) + (j_2 * 8)) + 4)))[0]), "=r"(((unsigned *)(C_shared_warp + (((i_2 * 48) + (j_2 * 8)) + 4)))[1])
      : "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i_2 * 8)))[3]), "r"(((unsigned *)(B_decode_shared_warp + ((j_2 * 8) + 4)))[0]), "r"(((unsigned *)(B_decode_shared_warp + ((j_2 * 8) + 4)))[1]), "r"(((unsigned *)(C_shared_warp + (((i_2 * 48) + (j_2 * 8)) + 4)))[0]), "r"(((unsigned *)(C_shared_warp + (((i_2 * 48) + (j_2 * 8)) + 4)))[1]));
  }
        }
      }
    }
  }
  for (int ax0_3 = 0; ax0_3 < 3; ++ax0_3) {
    for (int ax1 = 0; ax1 < 6; ++ax1) {
      __syncthreads();
      for (int local_id = 0; local_id < 8; local_id+=2) {
*((uint *)&(&(C_shared[(((int)threadIdx.z) * 1536)]))[((((((local_id % 4) / 2) * 8) + (threadIdx.x / 4)) * 16) + ((((local_id / 4) * 8) + ((threadIdx.x % 4) * 2)) + (local_id % 2)))]) = *((uint *)&C_shared_warp[((ax0_3 * 48) + (ax1 * 8)) + local_id]);
}
;
      __syncthreads();
      #pragma unroll
      for (int ax0_ax1_ax2_ax3_fused_0 = 0; ax0_ax1_ax2_ax3_fused_0 < 1; ++ax0_ax1_ax2_ax3_fused_0) {
        *(uint4*)(C + ((((ax0_3 * 6144) + (((int)threadIdx.z) * 1536)) + (ax1 * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(C_shared + ((((int)threadIdx.z) * 1536) + (((int)threadIdx.x) * 8)));
      }
    }
  }
}

