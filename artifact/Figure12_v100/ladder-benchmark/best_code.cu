__global__ void __launch_bounds__(128) Fused(float* __restrict__ A, int8_t* __restrict__ B, uint8_t* __restrict__ Scales, float* __restrict__ C) {
  
  float in_thread_C_local[1];
  float A_local[4];
  signed char B_local[4];
  float B_decode_local[4];
  __shared__ float red_buf0[128];
  in_thread_C_local[0] = 0.000000e+00f;
  for (int k_0 = 0; k_0 < 56; ++k_0) {
    *(float4*)(A_local + 0) = *(float4*)(A + ((k_0 * 512) + (((int)threadIdx.x) * 4)));
    *(int*)(B_local + 0) = *(int*)(B + (((((int)blockIdx.x) * 28672) + (k_0 * 512)) + (((int)threadIdx.x) * 4)));
    for (int ax1 = 0; ax1 < 4; ++ax1) {
        uint __1 = (((max((((((((uint)B_local[ax1]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)Scales[(((k_0 * 131072) + ((((int)threadIdx.x) >> 3) * 8192)) + ((int)blockIdx.x))])), (uint)63) | ((((((uint)B_local[ax1]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)2) | (((((((uint)B_local[ax1]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2)) << (uint)25;
      B_decode_local[ax1] = (*(float *)(&(__1)));
    }
    for (int k_2 = 0; k_2 < 4; ++k_2) {
      in_thread_C_local[0] = (in_thread_C_local[0] + (A_local[k_2] * B_decode_local[k_2]));
    }
  }
  __syncthreads();
  ((volatile float*)red_buf0)[((int)threadIdx.x)] = in_thread_C_local[0];
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

