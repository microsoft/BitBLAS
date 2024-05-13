__global__ void __launch_bounds__(128) Fused(half* __restrict__ A, int8_t* __restrict__ B, half* __restrict__ C) {
  
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> C_shared_wmma_accumulator[4];
  __shared__ half A_shared[1024];
  __shared__ half B_decode_shared[4096];
  signed char B_local[2];
  half B_decode_local[8];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> B_decode_shared_wmma_matrix_b[4];
  __shared__ half C_shared[2560];
  for (int i_2_init = 0; i_2_init < 1; ++i_2_init) {
    for (int j_2_init = 0; j_2_init < 4; ++j_2_init) {
      nvcuda::wmma::fill_fragment(C_shared_wmma_accumulator[j_2_init], 0.000000e+00f);
    }
  }
  for (int k_0 = 0; k_0 < 896; ++k_0) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_ax2_ax3_0_fused_0 = 0; ax0_ax1_ax2_ax3_0_fused_0 < 1; ++ax0_ax1_ax2_ax3_0_fused_0) {
      *(uint4*)(A_shared + (((((int)threadIdx.y) * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(A + (((((((int)blockIdx.y) * 917504) + (((int)threadIdx.y) * 458752)) + (k_0 * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8)));
    }
    for (int ax0_ax1_ax2_ax3_0_fused_0_1 = 0; ax0_ax1_ax2_ax3_0_fused_0_1 < 4; ++ax0_ax1_ax2_ax3_0_fused_0_1) {
      *(char2*)(B_local + 0) = *(char2*)(B + ((((((((int)blockIdx.x) * 917504) + (ax0_ax1_ax2_ax3_0_fused_0_1 * 229376)) + (((int)threadIdx.y) * 114688)) + (k_0 * 128)) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.x) * 2)));
      for (int ax0 = 0; ax0 < 8; ++ax0) {
        B_decode_local[ax0] = ((half)((B_local[(ax0 >> 2)] >> ((signed char)((ax0 & 3) * 2))) & (signed char)3));
      }
      *(uint4*)(B_decode_shared + ((((ax0_ax1_ax2_ax3_0_fused_0_1 * 1024) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(B_decode_local + 0);
    }
    __syncthreads();
    for (int k_1 = 0; k_1 < 2; ++k_1) {
      nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[0], (&(A_shared[((((int)threadIdx.y) * 512) + (k_1 * 256))])), 16);
      for (int ax0_1 = 0; ax0_1 < 4; ++ax0_1) {
        nvcuda::wmma::load_matrix_sync(B_decode_shared_wmma_matrix_b[ax0_1], (&(B_decode_shared[(((((int)threadIdx.z) * 2048) + (ax0_1 * 512)) + (k_1 * 256))])), 16);
      }
      for (int j_2 = 0; j_2 < 4; ++j_2) {
        nvcuda::wmma::mma_sync(C_shared_wmma_accumulator[j_2], A_shared_wmma_matrix_a[0], B_decode_shared_wmma_matrix_b[j_2], C_shared_wmma_accumulator[j_2]);
      }
    }
  }
  __syncthreads();
  for (int ax1 = 0; ax1 < 4; ++ax1) {
    __syncthreads();
    nvcuda::wmma::store_matrix_sync((&(C_shared[((((int)threadIdx.y) * 1280) + (((int)threadIdx.z) * 1024))])), C_shared_wmma_accumulator[ax1], 16, nvcuda::wmma::mem_row_major);
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_ax2_ax3_fused_0 = 0; ax0_ax1_ax2_ax3_fused_0 < 1; ++ax0_ax1_ax2_ax3_fused_0) {
      *(uint4*)(C + ((((((((int)blockIdx.y) * 262144) + (((int)threadIdx.y) * 131072)) + (((int)blockIdx.x) * 2048)) + (((int)threadIdx.z) * 1024)) + (ax1 * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(C_shared + (((((int)threadIdx.y) * 1280) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.x) * 8)));
    }
  }
  __syncthreads();
}

