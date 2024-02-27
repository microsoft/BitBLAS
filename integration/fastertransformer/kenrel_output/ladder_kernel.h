// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef __LADDER_KERNEL_H__
#define __LADDER_KERNEL_H__
#include <cuda_fp16.h>
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800) 
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800) 
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 1
#else
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 0
#endif


template <typename T1, typename T2, bool isSigned = false>
__device__ void decode_i4b_to_f16(T1 *_i4s, T2 *B_local_decode, const int N = 8)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x000f000f;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint MEDIAN_NUM = isSigned ? 0x64076407 : 0x64006400;
    uint const i4s = *reinterpret_cast<uint *>(_i4s);
#pragma unroll
    for (int i = 0; i < (N / 2); i++)
    {

        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[i])
                     : "r"(i4s >> (4 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
    }
}

template <typename T1, typename T2>
__device__ void decode_i4s_to_f16(T1 *_i4s, T2 *B_local_decode, const int N = 8)
{
    decode_i4b_to_f16<T1, T2, true>(_i4s, B_local_decode, N);
}

template <typename T1, typename T2>
__device__ void decode_i4u_to_f16(T1 *_i4u, T2 *B_local_decode, const int N = 8)
{
    decode_i4b_to_f16<T1, T2, false>(_i4u, B_local_decode, N);
}
,


template <typename T1, typename T2, typename T3, bool isSigned = false>
__device__ void decode_i4b_to_f16_scale(T1 *_i4s, T2 *B_local_decode, T3 *scale, const int N = 8)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x000f000f;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint MEDIAN_NUM = isSigned ? 0x64076407 : 0x64006400;
    uint const i4s = *reinterpret_cast<uint *>(_i4s);
#pragma unroll
    for (int i = 0; i < (N / 2); i++)
    {

        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[i])
                     : "r"(i4s >> (4 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
        unsigned v0 = *((unsigned short *)scale);
        unsigned v1 = *((unsigned short *)scale);
        unsigned __packed_scale = (v1 << 16) | v0;
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[i]) : "r"(h[i]), "r"(__packed_scale), "r"(0));
    }
    
}

template <typename T1, typename T2, typename T3>
__device__ void decode_i4s_to_f16_scale(T1 *_i4s, T2 *B_local_decode, T3 *scale, const int N = 8)
{
    decode_i4b_to_f16_scale<T1, T2, T3, true>(_i4s, B_local_decode, scale, N);
}

template <typename T1, typename T2, typename T3>
__device__ void decode_i4u_to_f16_scale(T1 *_i4u, T2 *B_local_decode,  T3 *scale, const int N = 8)
{
    decode_i4b_to_f16_scale<T1, T2, T3, false>(_i4u, B_local_decode, scale, N);
}
,


template <typename T1, typename T2, bool isSigned = false>
__device__ void decode_i2b_to_f16(T1 *_i2s, T2 *B_local_decode, const int N = 8)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x00030003;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint MEDIAN_NUM = isSigned ? 0x64016401 : 0x64006400;
    int16_t const i2s_i16 = *reinterpret_cast<int16_t *>(_i2s);
    // decode 2 elems at one time.
    // interleave {e15,e13,e11,e9,e7,e5,e3,e1,e14,e12,e10,e8,e6,e4,e2,e0}
    // only decode for {x,x,x,x,e7,e5,e3,e1,x,x,x,x,e6,e4,e2,e0}
    // otherwise the pointer of _i2s should be moved to
    int i2s = (i2s_i16 & 0x00ff);
    i2s |= ((i2s_i16 & 0xff00) << 8);

#pragma unroll
    for (int i = 0; i < (N / 2); i++)
    {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[i])
                     : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
    }
}

template <typename T1, typename T2>
__device__ void decode_i2s_to_f16(T1 *_i2s, T2 *B_local_decode, const int N = 8)
{
    decode_i2b_to_f16<T1, T2, true>(_i2s, B_local_decode, N);
}

template <typename T1, typename T2>
__device__ void decode_i2u_to_f16(T1 *_i2u, T2 *B_local_decode, const int N = 8)
{
    decode_i2b_to_f16<T1, T2, false>(_i2u, B_local_decode, N);
}
,


template <typename T1, typename T2, typename T3, bool isSigned = false>
__device__ void decode_i2b_to_f16_scale(T1 *_i2s, T2 *B_local_decode, T3 *scale = nullptr, const int N = 8)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x00030003;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint MEDIAN_NUM = isSigned ? 0x64016401 : 0x64006400;
    int16_t const i2s_i16 = *reinterpret_cast<int16_t *>(_i2s);
    // decode 2 elems at one time.
    // interleave {e15,e13,e11,e9,e7,e5,e3,e1,e14,e12,e10,e8,e6,e4,e2,e0}
    // only decode for {x,x,x,x,e7,e5,e3,e1,x,x,x,x,e6,e4,e2,e0}
    // otherwise the pointer of _i2s should be moved to
    int i2s = (i2s_i16 & 0x00ff);
    i2s |= ((i2s_i16 & 0xff00) << 8);

#pragma unroll
    for (int i = 0; i < (N / 2); i++)
    {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[i])
                     : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[i]) : "r"(h[i]), "r"(__pack_half2(*scale, *scale)), "r"(0));
    }
}

template <typename T1, typename T2, typename T3>
__device__ void decode_i2s_to_f16_scale(T1 *_i2s, T2 *B_local_decode, T3 *scale, const int N = 8)
{
    decode_i2b_to_f16_scale<T1, T2, T3, true>(_i2s, B_local_decode, scale, N);
}

template <typename T1, typename T2, typename T3>
__device__ void decode_i2u_to_f16_scale(T1 *_i2u, T2 *B_local_decode,  T3 *scale, const int N = 8)
{
    decode_i2b_to_f16_scale<T1, T2, T3, false>(_i2u, B_local_decode, scale, N);
}
,


// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}



int ladder_gemm_fp16xint2_fp16(half *input_0, half *input_1, half *output, const int M, const int N, const int K, const int trans_a, const int trans_b, half *workspace_ptr);

#endif

    