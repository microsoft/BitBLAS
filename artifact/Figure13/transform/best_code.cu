__global__ void __launch_bounds__(128) Fused(int8_t* __restrict__ A, int8_t* __restrict__ B, uint8_t* __restrict__ AScales, uint8_t* __restrict__ BScales, float* __restrict__ C) {
  
  float C_warp[8];
  signed char A_local[8];
  half A_decode_local[8];
  __shared__ half A_decode_shared[512];
  __shared__ half B_decode_shared[2048];
  signed char B_local[8];
  half B_decode_local[8];
  half A_decode_shared_warp[8];
  half B_decode_shared_warp[8];

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
C_warp[0 + i] = 0.0;}
;
    }
  }
  for (int k_0 = 0; k_0 < 256; ++k_0) {
    if (((int)threadIdx.z) < 2) {
      *(int2*)(A_local + 0) = *(int2*)(A + ((((((int)blockIdx.y) * 131072) + (k_0 * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8)));
    }
    for (int ax0 = 0; ax0 < 8; ++ax0) {
      if (((int)threadIdx.z) < 2) {
          uint __1 = ((max((((((((uint)A_local[ax0]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)AScales[(((k_0 * 1024) + (((int)blockIdx.y) * 16)) + (((int)threadIdx.x) >> 1))])), (uint)63) | ((((((uint)A_local[ax0]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)7) | (((((((uint)A_local[ax0]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
        A_decode_local[ax0] = (*(half *)(&(__1)));
      }
    }
    __syncthreads();
    if (((int)threadIdx.z) < 2) {
      *(uint4*)(A_decode_shared + ((((int)threadIdx.z) * 256) + (((int)threadIdx.x) * 8))) = *(uint4*)(A_decode_local + 0);
    }
    for (int ax0_ax1_ax2_ax3_0_fused_0 = 0; ax0_ax1_ax2_ax3_0_fused_0 < 2; ++ax0_ax1_ax2_ax3_0_fused_0) {
      *(int2*)(B_local + 0) = *(int2*)(B + ((((((((int)blockIdx.x) * 524288) + (ax0_ax1_ax2_ax3_0_fused_0 * 262144)) + ((((int)threadIdx.z) >> 1) * 131072)) + (k_0 * 512)) + ((((int)threadIdx.z) & 1) * 256)) + (((int)threadIdx.x) * 8)));
      for (int ax0_1 = 0; ax0_1 < 8; ++ax0_1) {
          uint __2 = ((max((((((((uint)B_local[ax0_1]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)BScales[(((((k_0 * 1024) + (((int)blockIdx.x) * 64)) + (ax0_ax1_ax2_ax3_0_fused_0 * 32)) + ((((int)threadIdx.z) >> 1) * 16)) + (((int)threadIdx.x) >> 1))])), (uint)63) | ((((((uint)B_local[ax0_1]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)7) | (((((((uint)B_local[ax0_1]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
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

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(B_decode_shared[((((int)threadIdx.z) * 512) + (k_1 * 256))])) + (((int)threadIdx.x) * 8))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(B_decode_shared[((((int)threadIdx.z) * 512) + (k_1 * 256))])) + (((int)threadIdx.x) * 8)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_decode_shared_warp + 0))[0]), "=r"(((unsigned *)(B_decode_shared_warp + 0))[1]), "=r"(((unsigned *)(B_decode_shared_warp + 0))[2]), "=r"(((unsigned *)(B_decode_shared_warp + 0))[3])
      : "r"(addr)
    );
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(C_warp + 0))[0]), "=f"(((float *)(C_warp + 0))[1]), "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
      : "r"(((unsigned *)(A_decode_shared_warp + 0))[0]), "r"(((unsigned *)(A_decode_shared_warp + 0))[1]), "r"(((unsigned *)(A_decode_shared_warp + 0))[2]), "r"(((unsigned *)(A_decode_shared_warp + 0))[3]), "r"(((unsigned *)(B_decode_shared_warp + 0))[0]), "r"(((unsigned *)(B_decode_shared_warp + 0))[1]), "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]), "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(C_warp + 4))[0]), "=f"(((float *)(C_warp + 4))[1]), "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
      : "r"(((unsigned *)(A_decode_shared_warp + 0))[0]), "r"(((unsigned *)(A_decode_shared_warp + 0))[1]), "r"(((unsigned *)(A_decode_shared_warp + 0))[2]), "r"(((unsigned *)(A_decode_shared_warp + 0))[3]), "r"(((unsigned *)(B_decode_shared_warp + 4))[0]), "r"(((unsigned *)(B_decode_shared_warp + 4))[1]), "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]), "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
  }
    }
  }
  for (int local_id = 0; local_id < 8; ++local_id) {
(&(C[(((((int)blockIdx.y) * 16384) + (((int)blockIdx.x) * 1024)) + (((int)threadIdx.z) * 256))]))[((((((local_id % 4) / 2) * 8) + (threadIdx.x / 4)) * 16) + ((((local_id / 4) * 8) + ((threadIdx.x % 4) * 2)) + (local_id % 2)))] = C_warp[0 + local_id];
}
;
}

