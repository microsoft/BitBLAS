__global__ void __launch_bounds__(128) Fused(int8_t* __restrict__ A, int8_t* __restrict__ B, int8_t* __restrict__ Scales, float* __restrict__ C) {
  
  float normal_reduce_temp0[1];
  __shared__ float A_decode_shared[2048];
  __shared__ float B_decode_shared[2048];
  __shared__ float red_buf0[128];
  normal_reduce_temp0[0] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 14; ++k_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_inner_s = 0; ax0_ax1_fused_inner_s < 4; ++ax0_ax1_fused_inner_s) {
        uint v_ = ((max((((((((uint)A[(((k_outer * 2048) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)Scales[((k_outer * 524288) + ((((int)threadIdx.x) >> 3) * 8192))])), (uint)63) | ((((((uint)A[(((k_outer * 2048) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s)]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)23) | (((((((uint)A[(((k_outer * 2048) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
      A_decode_shared[((((int)threadIdx.x) * 4) + ax0_ax1_fused_inner_s)] = (*(float *)(&(v_)));
    }
    for (int ax0_ax1_fused_inner_s_1 = 0; ax0_ax1_fused_inner_s_1 < 4; ++ax0_ax1_fused_inner_s_1) {
        uint v__1 = ((max((((((((uint)A[((((k_outer * 2048) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_1) + 512)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)Scales[(((k_outer * 524288) + ((((int)threadIdx.x) >> 3) * 8192)) + 131072)])), (uint)63) | ((((((uint)A[((((k_outer * 2048) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_1) + 512)]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)23) | (((((((uint)A[((((k_outer * 2048) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_1) + 512)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
      A_decode_shared[(((((int)threadIdx.x) * 4) + ax0_ax1_fused_inner_s_1) + 512)] = (*(float *)(&(v__1)));
    }
    for (int ax0_ax1_fused_inner_s_2 = 0; ax0_ax1_fused_inner_s_2 < 4; ++ax0_ax1_fused_inner_s_2) {
        uint v__2 = ((max((((((((uint)A[((((k_outer * 2048) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_2) + 1024)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)Scales[(((k_outer * 524288) + ((((int)threadIdx.x) >> 3) * 8192)) + 262144)])), (uint)63) | ((((((uint)A[((((k_outer * 2048) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_2) + 1024)]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)23) | (((((((uint)A[((((k_outer * 2048) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_2) + 1024)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
      A_decode_shared[(((((int)threadIdx.x) * 4) + ax0_ax1_fused_inner_s_2) + 1024)] = (*(float *)(&(v__2)));
    }
    for (int ax0_ax1_fused_inner_s_3 = 0; ax0_ax1_fused_inner_s_3 < 4; ++ax0_ax1_fused_inner_s_3) {
        uint v__3 = ((max((((((((uint)A[((((k_outer * 2048) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_3) + 1536)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)Scales[(((k_outer * 524288) + ((((int)threadIdx.x) >> 3) * 8192)) + 393216)])), (uint)63) | ((((((uint)A[((((k_outer * 2048) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_3) + 1536)]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)23) | (((((((uint)A[((((k_outer * 2048) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_3) + 1536)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
      A_decode_shared[(((((int)threadIdx.x) * 4) + ax0_ax1_fused_inner_s_3) + 1536)] = (*(float *)(&(v__3)));
    }
    for (int ax0_ax1_fused_inner_s_4 = 0; ax0_ax1_fused_inner_s_4 < 4; ++ax0_ax1_fused_inner_s_4) {
        uint v__4 = ((max((((((((uint)B[((((((int)blockIdx.x) * 28672) + (k_outer * 2048)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_4)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)Scales[(((k_outer * 524288) + ((((int)threadIdx.x) >> 3) * 8192)) + ((int)blockIdx.x))])), (uint)63) | ((((((uint)B[((((((int)blockIdx.x) * 28672) + (k_outer * 2048)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_4)]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)23) | (((((((uint)B[((((((int)blockIdx.x) * 28672) + (k_outer * 2048)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_4)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
      B_decode_shared[((((int)threadIdx.x) * 4) + ax0_ax1_fused_inner_s_4)] = (*(float *)(&(v__4)));
    }
    for (int ax0_ax1_fused_inner_s_5 = 0; ax0_ax1_fused_inner_s_5 < 4; ++ax0_ax1_fused_inner_s_5) {
        uint v__5 = ((max((((((((uint)B[(((((((int)blockIdx.x) * 28672) + (k_outer * 2048)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_5) + 512)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)Scales[((((k_outer * 524288) + ((((int)threadIdx.x) >> 3) * 8192)) + ((int)blockIdx.x)) + 131072)])), (uint)63) | ((((((uint)B[(((((((int)blockIdx.x) * 28672) + (k_outer * 2048)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_5) + 512)]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)23) | (((((((uint)B[(((((((int)blockIdx.x) * 28672) + (k_outer * 2048)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_5) + 512)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
      B_decode_shared[(((((int)threadIdx.x) * 4) + ax0_ax1_fused_inner_s_5) + 512)] = (*(float *)(&(v__5)));
    }
    for (int ax0_ax1_fused_inner_s_6 = 0; ax0_ax1_fused_inner_s_6 < 4; ++ax0_ax1_fused_inner_s_6) {
        uint v__6 = ((max((((((((uint)B[(((((((int)blockIdx.x) * 28672) + (k_outer * 2048)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_6) + 1024)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)Scales[((((k_outer * 524288) + ((((int)threadIdx.x) >> 3) * 8192)) + ((int)blockIdx.x)) + 262144)])), (uint)63) | ((((((uint)B[(((((((int)blockIdx.x) * 28672) + (k_outer * 2048)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_6) + 1024)]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)23) | (((((((uint)B[(((((((int)blockIdx.x) * 28672) + (k_outer * 2048)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_6) + 1024)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
      B_decode_shared[(((((int)threadIdx.x) * 4) + ax0_ax1_fused_inner_s_6) + 1024)] = (*(float *)(&(v__6)));
    }
    for (int ax0_ax1_fused_inner_s_7 = 0; ax0_ax1_fused_inner_s_7 < 4; ++ax0_ax1_fused_inner_s_7) {
        uint v__7 = ((max((((((((uint)B[(((((((int)blockIdx.x) * 28672) + (k_outer * 2048)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_7) + 1536)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)Scales[((((k_outer * 524288) + ((((int)threadIdx.x) >> 3) * 8192)) + ((int)blockIdx.x)) + 393216)])), (uint)63) | ((((((uint)B[(((((((int)blockIdx.x) * 28672) + (k_outer * 2048)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_7) + 1536)]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)23) | (((((((uint)B[(((((((int)blockIdx.x) * 28672) + (k_outer * 2048)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_7) + 1536)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
      B_decode_shared[(((((int)threadIdx.x) * 4) + ax0_ax1_fused_inner_s_7) + 1536)] = (*(float *)(&(v__7)));
    }
    __syncthreads();
    for (int k_inner_outer = 0; k_inner_outer < 16; ++k_inner_outer) {
      normal_reduce_temp0[0] = (normal_reduce_temp0[0] + (A_decode_shared[((k_inner_outer * 128) + ((int)threadIdx.x))] * B_decode_shared[((k_inner_outer * 128) + ((int)threadIdx.x))]));
    }
  }
  __syncthreads();
  ((volatile float*)red_buf0)[((int)threadIdx.x)] = normal_reduce_temp0[0];
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = (((volatile float*)red_buf0)[((int)threadIdx.x)] + ((volatile float*)red_buf0)[(((int)threadIdx.x) + 64)]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = (((volatile float*)red_buf0)[((int)threadIdx.x)] + ((volatile float*)red_buf0)[(((int)threadIdx.x) + 32)]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    float w_16_0 = (((volatile float*)red_buf0)[((int)threadIdx.x)] + ((volatile float*)red_buf0)[(((int)threadIdx.x) + 16)]);
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = w_16_0;
    float w_8_0 = (((volatile float*)red_buf0)[((int)threadIdx.x)] + ((volatile float*)red_buf0)[(((int)threadIdx.x) + 8)]);
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = w_8_0;
    float w_4_0 = (((volatile float*)red_buf0)[((int)threadIdx.x)] + ((volatile float*)red_buf0)[(((int)threadIdx.x) + 4)]);
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = w_4_0;
    float w_2_0 = (((volatile float*)red_buf0)[((int)threadIdx.x)] + ((volatile float*)red_buf0)[(((int)threadIdx.x) + 2)]);
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = w_2_0;
    float w_1_0 = (((volatile float*)red_buf0)[((int)threadIdx.x)] + ((volatile float*)red_buf0)[(((int)threadIdx.x) + 1)]);
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = w_1_0;
  }
  __syncthreads();
  C[((int)blockIdx.x)] = ((volatile float*)red_buf0)[0];
}

