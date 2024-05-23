__global__ void __launch_bounds__(128) Fused(int8_t* __restrict__ A, int8_t* __restrict__ B, int8_t* __restrict__ AScales, int8_t* __restrict__ Scales, float* __restrict__ C) {
  
  float normal_reduce_temp0[1];
  __shared__ float A_decode_shared[2048];
  __shared__ float B_decode_shared[4096];
  float red_buf0[1];
  normal_reduce_temp0[0] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_inner_s = 0; ax0_ax1_fused_inner_s < 4; ++ax0_ax1_fused_inner_s) {
        uint __1 = ((max((((((((uint)A[((((((((int)blockIdx.x) >> 7) * 32768) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)AScales[(((k_outer * 65536) + ((((int)threadIdx.y) >> 1) * 4096)) + ((((int)blockIdx.x) >> 7) * 4))])), (uint)63) | ((((((uint)A[((((((((int)blockIdx.x) >> 7) * 32768) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s)]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)23) | (((((((uint)A[((((((((int)blockIdx.x) >> 7) * 32768) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
      A_decode_shared[(((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s)] = (*(float *)(&(__1)));
    }
    for (int ax0_ax1_fused_inner_s_1 = 0; ax0_ax1_fused_inner_s_1 < 4; ++ax0_ax1_fused_inner_s_1) {
        uint __2 = ((max((((((((uint)A[(((((((((int)blockIdx.x) >> 7) * 32768) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_1) + 8192)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)AScales[((((k_outer * 65536) + ((((int)threadIdx.y) >> 1) * 4096)) + ((((int)blockIdx.x) >> 7) * 4)) + 1)])), (uint)63) | ((((((uint)A[(((((((((int)blockIdx.x) >> 7) * 32768) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_1) + 8192)]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)23) | (((((((uint)A[(((((((((int)blockIdx.x) >> 7) * 32768) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_1) + 8192)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
      A_decode_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_1) + 512)] = (*(float *)(&(__2)));
    }
    for (int ax0_ax1_fused_inner_s_2 = 0; ax0_ax1_fused_inner_s_2 < 4; ++ax0_ax1_fused_inner_s_2) {
        uint __3 = ((max((((((((uint)A[(((((((((int)blockIdx.x) >> 7) * 32768) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_2) + 16384)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)AScales[((((k_outer * 65536) + ((((int)threadIdx.y) >> 1) * 4096)) + ((((int)blockIdx.x) >> 7) * 4)) + 2)])), (uint)63) | ((((((uint)A[(((((((((int)blockIdx.x) >> 7) * 32768) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_2) + 16384)]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)23) | (((((((uint)A[(((((((((int)blockIdx.x) >> 7) * 32768) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_2) + 16384)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
      A_decode_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_2) + 1024)] = (*(float *)(&(__3)));
    }
    for (int ax0_ax1_fused_inner_s_3 = 0; ax0_ax1_fused_inner_s_3 < 4; ++ax0_ax1_fused_inner_s_3) {
        uint __4 = ((max((((((((uint)A[(((((((((int)blockIdx.x) >> 7) * 32768) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_3) + 24576)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)AScales[((((k_outer * 65536) + ((((int)threadIdx.y) >> 1) * 4096)) + ((((int)blockIdx.x) >> 7) * 4)) + 3)])), (uint)63) | ((((((uint)A[(((((((((int)blockIdx.x) >> 7) * 32768) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_3) + 24576)]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)23) | (((((((uint)A[(((((((((int)blockIdx.x) >> 7) * 32768) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_3) + 24576)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
      A_decode_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_3) + 1536)] = (*(float *)(&(__4)));
    }
    for (int ax0_ax1_fused_inner_s_4 = 0; ax0_ax1_fused_inner_s_4 < 4; ++ax0_ax1_fused_inner_s_4) {
        uint __5 = ((max((((((((uint)B[((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_4)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)Scales[(((k_outer * 16384) + ((((int)threadIdx.y) >> 1) * 1024)) + ((((int)blockIdx.x) & 127) * 8))])), (uint)63) | ((((((uint)B[((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_4)]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)23) | (((((((uint)B[((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_4)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
      B_decode_shared[(((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_4)] = (*(float *)(&(__5)));
    }
    for (int ax0_ax1_fused_inner_s_5 = 0; ax0_ax1_fused_inner_s_5 < 4; ++ax0_ax1_fused_inner_s_5) {
        uint __6 = ((max((((((((uint)B[(((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_5) + 8192)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)Scales[((((k_outer * 16384) + ((((int)threadIdx.y) >> 1) * 1024)) + ((((int)blockIdx.x) & 127) * 8)) + 1)])), (uint)63) | ((((((uint)B[(((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_5) + 8192)]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)23) | (((((((uint)B[(((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_5) + 8192)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
      B_decode_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_5) + 512)] = (*(float *)(&(__6)));
    }
    for (int ax0_ax1_fused_inner_s_6 = 0; ax0_ax1_fused_inner_s_6 < 4; ++ax0_ax1_fused_inner_s_6) {
        uint __7 = ((max((((((((uint)B[(((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_6) + 16384)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)Scales[((((k_outer * 16384) + ((((int)threadIdx.y) >> 1) * 1024)) + ((((int)blockIdx.x) & 127) * 8)) + 2)])), (uint)63) | ((((((uint)B[(((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_6) + 16384)]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)23) | (((((((uint)B[(((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_6) + 16384)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
      B_decode_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_6) + 1024)] = (*(float *)(&(__7)));
    }
    for (int ax0_ax1_fused_inner_s_7 = 0; ax0_ax1_fused_inner_s_7 < 4; ++ax0_ax1_fused_inner_s_7) {
        uint __8 = ((max((((((((uint)B[(((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_7) + 24576)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)Scales[((((k_outer * 16384) + ((((int)threadIdx.y) >> 1) * 1024)) + ((((int)blockIdx.x) & 127) * 8)) + 3)])), (uint)63) | ((((((uint)B[(((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_7) + 24576)]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)23) | (((((((uint)B[(((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_7) + 24576)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
      B_decode_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_7) + 1536)] = (*(float *)(&(__8)));
    }
    for (int ax0_ax1_fused_inner_s_8 = 0; ax0_ax1_fused_inner_s_8 < 4; ++ax0_ax1_fused_inner_s_8) {
        uint __9 = ((max((((((((uint)B[(((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_8) + 32768)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)Scales[((((k_outer * 16384) + ((((int)threadIdx.y) >> 1) * 1024)) + ((((int)blockIdx.x) & 127) * 8)) + 4)])), (uint)63) | ((((((uint)B[(((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_8) + 32768)]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)23) | (((((((uint)B[(((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_8) + 32768)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
      B_decode_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_8) + 2048)] = (*(float *)(&(__9)));
    }
    for (int ax0_ax1_fused_inner_s_9 = 0; ax0_ax1_fused_inner_s_9 < 4; ++ax0_ax1_fused_inner_s_9) {
        uint __10 = ((max((((((((uint)B[(((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_9) + 40960)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)Scales[((((k_outer * 16384) + ((((int)threadIdx.y) >> 1) * 1024)) + ((((int)blockIdx.x) & 127) * 8)) + 5)])), (uint)63) | ((((((uint)B[(((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_9) + 40960)]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)23) | (((((((uint)B[(((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_9) + 40960)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
      B_decode_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_9) + 2560)] = (*(float *)(&(__10)));
    }
    for (int ax0_ax1_fused_inner_s_10 = 0; ax0_ax1_fused_inner_s_10 < 4; ++ax0_ax1_fused_inner_s_10) {
        uint __11 = ((max((((((((uint)B[(((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_10) + 49152)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)Scales[((((k_outer * 16384) + ((((int)threadIdx.y) >> 1) * 1024)) + ((((int)blockIdx.x) & 127) * 8)) + 6)])), (uint)63) | ((((((uint)B[(((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_10) + 49152)]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)23) | (((((((uint)B[(((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_10) + 49152)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
      B_decode_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_10) + 3072)] = (*(float *)(&(__11)));
    }
    for (int ax0_ax1_fused_inner_s_11 = 0; ax0_ax1_fused_inner_s_11 < 4; ++ax0_ax1_fused_inner_s_11) {
        uint __12 = ((max((((((((uint)B[(((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_11) + 57344)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) + ((uint)Scales[((((k_outer * 16384) + ((((int)threadIdx.y) >> 1) * 1024)) + ((((int)blockIdx.x) & 127) * 8)) + 7)])), (uint)63) | ((((((uint)B[(((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_11) + 57344)]) >> (uint)0) & (uint)255) >> (uint)7) << (uint)8)) << (uint)23) | (((((((uint)B[(((((((((int)blockIdx.x) & 127) * 65536) + (k_outer * 512)) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_11) + 57344)]) >> (uint)0) & (uint)255) >> (uint)2) & (uint)31) & (uint)2);
      B_decode_shared[((((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_inner_s_11) + 3584)] = (*(float *)(&(__12)));
    }
    __syncthreads();
    for (int k_inner_outer = 0; k_inner_outer < 128; ++k_inner_outer) {
      normal_reduce_temp0[0] = (normal_reduce_temp0[0] + (A_decode_shared[((((((int)threadIdx.y) >> 3) * 512) + (k_inner_outer * 4)) + ((int)threadIdx.x))] * B_decode_shared[((((((int)threadIdx.y) & 7) * 512) + (k_inner_outer * 4)) + ((int)threadIdx.x))]));
    }
  }
  uint mask[1];
  float t0[1];
  red_buf0[0] = normal_reduce_temp0[0];
  mask[0] = (__activemask() & ((uint)(15 << (((int)threadIdx.y) * 4))));
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], (((int)threadIdx.y) * 4), 32);
  C[(((((((int)blockIdx.x) >> 7) * 4096) + ((((int)threadIdx.y) >> 3) * 1024)) + ((((int)blockIdx.x) & 127) * 8)) + (((int)threadIdx.y) & 7))] = red_buf0[0];
}

