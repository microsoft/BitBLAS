// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#include <cuda_fp16.h>
__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}
#else

typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

#define TVM_FORCE_INLINE inline __attribute__((always_inline))
#define TVM_XINLINE TVM_FORCE_INLINE __device__ __host__
#define TVM_ALIGNED(x) __attribute__ ((aligned(x)))
#define TVM_HALF_OPERATOR(RTYPE, OP)                              \
  TVM_XINLINE RTYPE operator OP (half a, half b) {                \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (half a, T b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (T a, half b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }

#define TVM_HALF_ASSIGNOP(AOP, OP)                                \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const T& a) {                    \
    return *this = half(float(*this) OP float(a));                \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const volatile T& a) volatile {  \
    return *this = half(float(*this) OP float(a));                \
  }

class TVM_ALIGNED(2) half {
 public:
  uint16_t half_;

  static TVM_XINLINE half Binary(uint16_t value) {
    half res;
    res.half_ = value;
    return res;
  }

  TVM_XINLINE half() {}

  TVM_XINLINE half(const float& value) { constructor(value); }
  TVM_XINLINE explicit half(const double& value) { constructor(value); }
  TVM_XINLINE explicit half(const int8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const int32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const long long& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint64_t& value) { constructor(value); }

  TVM_XINLINE operator float() const {                          \
    return float(half2float(half_));                            \
  }                                                             \
  TVM_XINLINE operator float() const volatile {                 \
    return float(half2float(half_));                            \
  }


  TVM_HALF_ASSIGNOP(+=, +)
  TVM_HALF_ASSIGNOP(-=, -)
  TVM_HALF_ASSIGNOP(*=, *)
  TVM_HALF_ASSIGNOP(/=, /)

  TVM_XINLINE half operator+() {
    return *this;
  }

  TVM_XINLINE half operator-() {
    return half(-float(*this));
  }

  TVM_XINLINE half operator=(const half& a) {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) {
    return *this = half(a);
  }

  TVM_XINLINE half operator=(const half& a) volatile {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) volatile {
    return *this = half(a);
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static int const fp16FractionBits = 10;
  static int const fp32FractionBits = 23;
  static int32_t const fp32FractionMask = ~(~0u << fp32FractionBits);   // == 0x7fffff
  static int32_t const fp32HiddenBit = 1 << fp32FractionBits;   // == 0x800000
  static int const shift = fp32FractionBits - fp16FractionBits;   // == 13
  static int const shiftSign = 16;
  static int32_t const expAdjust = 127 - 15;   // exp32-127 = exp16-15, so exp16 = exp32 - (127-15)

  static int32_t const infN = 0x7F800000;   // flt32 infinity
  static int32_t const maxN = 0x477FFFFF;   // max flt32 that's a flt16 normal after >> by shift
  static int32_t const minN = 0x38800000;   // min flt16 normal as a flt32
  static int32_t const maxZ = 0x33000000;   // max fp32 number that's still rounded to zero in fp16
  static int32_t const signN = 0x80000000;  // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift;   // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

  static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
  static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400;  // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  TVM_XINLINE uint16_t float2half(const float& value) const {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  // Same as above routine, except for addition of volatile keyword
  TVM_XINLINE uint16_t float2half(
    const volatile float& value) const volatile {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  TVM_XINLINE float half2float(const uint16_t& value) const {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  TVM_XINLINE float half2float(
    const volatile uint16_t& value) const volatile {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  template<typename T>
  TVM_XINLINE void constructor(const T& value) {
    half_ = float2half(float(value));
  }
};

TVM_HALF_OPERATOR(half, +)
TVM_HALF_OPERATOR(half, -)
TVM_HALF_OPERATOR(half, *)
TVM_HALF_OPERATOR(half, /)
TVM_HALF_OPERATOR(bool, >)
TVM_HALF_OPERATOR(bool, <)
TVM_HALF_OPERATOR(bool, >=)
TVM_HALF_OPERATOR(bool, <=)

TVM_XINLINE half __float2half_rn(const float a) {
  return half(a);
}
#endif


// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x, half y) {   \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x) {          \
  float tmp_x = __half2float(x);                                         \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2half(result);                                           \
}

// Some fp16 math functions are not supported in cuda_fp16.h,
// so we define them here to make sure the generated CUDA code
// is valid.
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ >= 530)
CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)
#else
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hexp, exp)
#endif
#endif

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY
#include <mma.h>

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
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
extern "C" __global__ void __launch_bounds__(64) main_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> C_reindex_shared_dyn_wmma_accumulator[32];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_reindex_shared_dyn_wmma_matrix_a[32];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> B_reindex_shared_dyn_wmma_matrix_b[16];
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[0], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[4], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[1], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[5], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[2], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[6], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[3], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[7], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[8], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[12], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[9], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[13], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[10], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[14], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[11], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[15], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[16], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[20], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[17], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[21], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[18], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[22], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[19], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[23], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[24], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[28], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[25], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[29], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[26], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[30], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[27], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(C_reindex_shared_dyn_wmma_accumulator[31], 0.000000e+00f);
  for (int ax2_0_0 = 0; ax2_0_0 < 256; ++ax2_0_0) {
    __syncthreads();
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 9216)) = *(uint1*)(A + ((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 9360)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 32768));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 9504)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 65536));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 9648)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 98304));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 9792)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 131072));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 9936)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 163840));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 10080)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 196608));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 10224)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 229376));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 10368)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 262144));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 10512)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 294912));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 10656)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 327680));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 10800)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 360448));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 10944)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 393216));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 11088)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 425984));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 11232)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 458752));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 11376)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 491520));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 11520)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 524288));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 11664)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 557056));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 11808)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 589824));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 11952)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 622592));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 12096)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 655360));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 12240)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 688128));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 12384)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 720896));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 12528)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 753664));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 12672)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 786432));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 12816)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 819200));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 12960)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 851968));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 13104)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 884736));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 13248)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 917504));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 13392)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 950272));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 13536)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 983040));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 13680)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1015808));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 13824)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1048576));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 13968)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1081344));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 14112)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1114112));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 14256)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1146880));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 14400)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1179648));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 14544)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1212416));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 14688)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1245184));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 14832)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1277952));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 14976)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1310720));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 15120)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1343488));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 15264)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1376256));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 15408)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1409024));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 15552)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1441792));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 15696)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1474560));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 15840)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1507328));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 15984)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1540096));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 16128)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1572864));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 16272)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1605632));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 16416)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1638400));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 16560)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1671168));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 16704)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1703936));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 16848)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1736704));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 16992)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1769472));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 17136)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1802240));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 17280)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1835008));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 17424)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1867776));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 17568)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1900544));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 17712)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1933312));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 17856)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1966080));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 18000)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 1998848));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 18144)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 2031616));
    *(uint1*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 72) + (((int)threadIdx.x) * 2)) + 18288)) = *(uint1*)(A + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + (((int)threadIdx.y) * 16384)) + (ax2_0_0 * 64)) + (((int)threadIdx.x) * 2)) + 2064384));
    *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(B + ((((((((int)blockIdx.y) & 127) * 2097152) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 576)) = *(uint4*)(B + (((((((((int)blockIdx.y) & 127) * 2097152) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 131072));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 1152)) = *(uint4*)(B + (((((((((int)blockIdx.y) & 127) * 2097152) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 262144));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 1728)) = *(uint4*)(B + (((((((((int)blockIdx.y) & 127) * 2097152) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 393216));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 2304)) = *(uint4*)(B + (((((((((int)blockIdx.y) & 127) * 2097152) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 524288));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 2880)) = *(uint4*)(B + (((((((((int)blockIdx.y) & 127) * 2097152) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 655360));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 3456)) = *(uint4*)(B + (((((((((int)blockIdx.y) & 127) * 2097152) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 786432));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 4032)) = *(uint4*)(B + (((((((((int)blockIdx.y) & 127) * 2097152) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 917504));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 4608)) = *(uint4*)(B + (((((((((int)blockIdx.y) & 127) * 2097152) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 1048576));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 5184)) = *(uint4*)(B + (((((((((int)blockIdx.y) & 127) * 2097152) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 1179648));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 5760)) = *(uint4*)(B + (((((((((int)blockIdx.y) & 127) * 2097152) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 1310720));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 6336)) = *(uint4*)(B + (((((((((int)blockIdx.y) & 127) * 2097152) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 1441792));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 6912)) = *(uint4*)(B + (((((((((int)blockIdx.y) & 127) * 2097152) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 1572864));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 7488)) = *(uint4*)(B + (((((((((int)blockIdx.y) & 127) * 2097152) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 1703936));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 8064)) = *(uint4*)(B + (((((((((int)blockIdx.y) & 127) * 2097152) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 1835008));
    *(uint4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)) + 8640)) = *(uint4*)(B + (((((((((int)blockIdx.y) & 127) * 2097152) + (((int)threadIdx.y) * 65536)) + ((((int)threadIdx.x) >> 3) * 16384)) + (ax2_0_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 1966080));
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[9216])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[9232])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[2], (&(((half*)buf_dyn_shmem)[9248])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[3], (&(((half*)buf_dyn_shmem)[9264])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[4], (&(((half*)buf_dyn_shmem)[10368])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[5], (&(((half*)buf_dyn_shmem)[10384])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[6], (&(((half*)buf_dyn_shmem)[10400])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[7], (&(((half*)buf_dyn_shmem)[10416])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[8], (&(((half*)buf_dyn_shmem)[11520])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[9], (&(((half*)buf_dyn_shmem)[11536])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[10], (&(((half*)buf_dyn_shmem)[11552])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[11], (&(((half*)buf_dyn_shmem)[11568])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[12], (&(((half*)buf_dyn_shmem)[12672])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[13], (&(((half*)buf_dyn_shmem)[12688])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[14], (&(((half*)buf_dyn_shmem)[12704])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[15], (&(((half*)buf_dyn_shmem)[12720])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[16], (&(((half*)buf_dyn_shmem)[13824])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[17], (&(((half*)buf_dyn_shmem)[13840])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[18], (&(((half*)buf_dyn_shmem)[13856])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[19], (&(((half*)buf_dyn_shmem)[13872])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[20], (&(((half*)buf_dyn_shmem)[14976])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[21], (&(((half*)buf_dyn_shmem)[14992])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[22], (&(((half*)buf_dyn_shmem)[15008])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[23], (&(((half*)buf_dyn_shmem)[15024])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[24], (&(((half*)buf_dyn_shmem)[16128])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[25], (&(((half*)buf_dyn_shmem)[16144])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[26], (&(((half*)buf_dyn_shmem)[16160])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[27], (&(((half*)buf_dyn_shmem)[16176])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[28], (&(((half*)buf_dyn_shmem)[17280])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[29], (&(((half*)buf_dyn_shmem)[17296])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[30], (&(((half*)buf_dyn_shmem)[17312])), 72);
    nvcuda::wmma::load_matrix_sync(A_reindex_shared_dyn_wmma_matrix_a[31], (&(((half*)buf_dyn_shmem)[17328])), 72);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[(((int)threadIdx.y) * 4608)])), 72);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 4608) + 16)])), 72);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 4608) + 32)])), 72);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 4608) + 48)])), 72);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[4], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 4608) + 1152)])), 72);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[5], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 4608) + 1168)])), 72);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[6], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 4608) + 1184)])), 72);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[7], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 4608) + 1200)])), 72);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[8], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 4608) + 2304)])), 72);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[9], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 4608) + 2320)])), 72);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[10], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 4608) + 2336)])), 72);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[11], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 4608) + 2352)])), 72);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[12], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 4608) + 3456)])), 72);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[13], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 4608) + 3472)])), 72);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[14], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 4608) + 3488)])), 72);
    nvcuda::wmma::load_matrix_sync(B_reindex_shared_dyn_wmma_matrix_b[15], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 4608) + 3504)])), 72);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[0]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[4], A_reindex_shared_dyn_wmma_matrix_a[4], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[4]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[0]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[4], A_reindex_shared_dyn_wmma_matrix_a[5], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[4]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[0]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[4], A_reindex_shared_dyn_wmma_matrix_a[6], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[4]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[0], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[0]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[4], A_reindex_shared_dyn_wmma_matrix_a[7], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[4]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[1]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[5], A_reindex_shared_dyn_wmma_matrix_a[4], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[5]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[1]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[5], A_reindex_shared_dyn_wmma_matrix_a[5], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[5]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[1]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[5], A_reindex_shared_dyn_wmma_matrix_a[6], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[5]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[1], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[1]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[5], A_reindex_shared_dyn_wmma_matrix_a[7], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[5]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[2], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[8], C_reindex_shared_dyn_wmma_accumulator[2]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[6], A_reindex_shared_dyn_wmma_matrix_a[4], B_reindex_shared_dyn_wmma_matrix_b[8], C_reindex_shared_dyn_wmma_accumulator[6]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[2], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[9], C_reindex_shared_dyn_wmma_accumulator[2]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[6], A_reindex_shared_dyn_wmma_matrix_a[5], B_reindex_shared_dyn_wmma_matrix_b[9], C_reindex_shared_dyn_wmma_accumulator[6]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[2], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[10], C_reindex_shared_dyn_wmma_accumulator[2]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[6], A_reindex_shared_dyn_wmma_matrix_a[6], B_reindex_shared_dyn_wmma_matrix_b[10], C_reindex_shared_dyn_wmma_accumulator[6]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[2], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[11], C_reindex_shared_dyn_wmma_accumulator[2]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[6], A_reindex_shared_dyn_wmma_matrix_a[7], B_reindex_shared_dyn_wmma_matrix_b[11], C_reindex_shared_dyn_wmma_accumulator[6]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[3], A_reindex_shared_dyn_wmma_matrix_a[0], B_reindex_shared_dyn_wmma_matrix_b[12], C_reindex_shared_dyn_wmma_accumulator[3]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[7], A_reindex_shared_dyn_wmma_matrix_a[4], B_reindex_shared_dyn_wmma_matrix_b[12], C_reindex_shared_dyn_wmma_accumulator[7]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[3], A_reindex_shared_dyn_wmma_matrix_a[1], B_reindex_shared_dyn_wmma_matrix_b[13], C_reindex_shared_dyn_wmma_accumulator[3]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[7], A_reindex_shared_dyn_wmma_matrix_a[5], B_reindex_shared_dyn_wmma_matrix_b[13], C_reindex_shared_dyn_wmma_accumulator[7]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[3], A_reindex_shared_dyn_wmma_matrix_a[2], B_reindex_shared_dyn_wmma_matrix_b[14], C_reindex_shared_dyn_wmma_accumulator[3]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[7], A_reindex_shared_dyn_wmma_matrix_a[6], B_reindex_shared_dyn_wmma_matrix_b[14], C_reindex_shared_dyn_wmma_accumulator[7]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[3], A_reindex_shared_dyn_wmma_matrix_a[3], B_reindex_shared_dyn_wmma_matrix_b[15], C_reindex_shared_dyn_wmma_accumulator[3]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[7], A_reindex_shared_dyn_wmma_matrix_a[7], B_reindex_shared_dyn_wmma_matrix_b[15], C_reindex_shared_dyn_wmma_accumulator[7]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[8], A_reindex_shared_dyn_wmma_matrix_a[8], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[8]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[12], A_reindex_shared_dyn_wmma_matrix_a[12], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[12]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[8], A_reindex_shared_dyn_wmma_matrix_a[9], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[8]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[12], A_reindex_shared_dyn_wmma_matrix_a[13], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[12]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[8], A_reindex_shared_dyn_wmma_matrix_a[10], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[8]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[12], A_reindex_shared_dyn_wmma_matrix_a[14], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[12]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[8], A_reindex_shared_dyn_wmma_matrix_a[11], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[8]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[12], A_reindex_shared_dyn_wmma_matrix_a[15], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[12]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[9], A_reindex_shared_dyn_wmma_matrix_a[8], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[9]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[13], A_reindex_shared_dyn_wmma_matrix_a[12], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[13]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[9], A_reindex_shared_dyn_wmma_matrix_a[9], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[9]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[13], A_reindex_shared_dyn_wmma_matrix_a[13], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[13]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[9], A_reindex_shared_dyn_wmma_matrix_a[10], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[9]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[13], A_reindex_shared_dyn_wmma_matrix_a[14], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[13]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[9], A_reindex_shared_dyn_wmma_matrix_a[11], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[9]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[13], A_reindex_shared_dyn_wmma_matrix_a[15], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[13]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[10], A_reindex_shared_dyn_wmma_matrix_a[8], B_reindex_shared_dyn_wmma_matrix_b[8], C_reindex_shared_dyn_wmma_accumulator[10]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[14], A_reindex_shared_dyn_wmma_matrix_a[12], B_reindex_shared_dyn_wmma_matrix_b[8], C_reindex_shared_dyn_wmma_accumulator[14]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[10], A_reindex_shared_dyn_wmma_matrix_a[9], B_reindex_shared_dyn_wmma_matrix_b[9], C_reindex_shared_dyn_wmma_accumulator[10]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[14], A_reindex_shared_dyn_wmma_matrix_a[13], B_reindex_shared_dyn_wmma_matrix_b[9], C_reindex_shared_dyn_wmma_accumulator[14]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[10], A_reindex_shared_dyn_wmma_matrix_a[10], B_reindex_shared_dyn_wmma_matrix_b[10], C_reindex_shared_dyn_wmma_accumulator[10]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[14], A_reindex_shared_dyn_wmma_matrix_a[14], B_reindex_shared_dyn_wmma_matrix_b[10], C_reindex_shared_dyn_wmma_accumulator[14]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[10], A_reindex_shared_dyn_wmma_matrix_a[11], B_reindex_shared_dyn_wmma_matrix_b[11], C_reindex_shared_dyn_wmma_accumulator[10]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[14], A_reindex_shared_dyn_wmma_matrix_a[15], B_reindex_shared_dyn_wmma_matrix_b[11], C_reindex_shared_dyn_wmma_accumulator[14]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[11], A_reindex_shared_dyn_wmma_matrix_a[8], B_reindex_shared_dyn_wmma_matrix_b[12], C_reindex_shared_dyn_wmma_accumulator[11]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[15], A_reindex_shared_dyn_wmma_matrix_a[12], B_reindex_shared_dyn_wmma_matrix_b[12], C_reindex_shared_dyn_wmma_accumulator[15]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[11], A_reindex_shared_dyn_wmma_matrix_a[9], B_reindex_shared_dyn_wmma_matrix_b[13], C_reindex_shared_dyn_wmma_accumulator[11]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[15], A_reindex_shared_dyn_wmma_matrix_a[13], B_reindex_shared_dyn_wmma_matrix_b[13], C_reindex_shared_dyn_wmma_accumulator[15]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[11], A_reindex_shared_dyn_wmma_matrix_a[10], B_reindex_shared_dyn_wmma_matrix_b[14], C_reindex_shared_dyn_wmma_accumulator[11]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[15], A_reindex_shared_dyn_wmma_matrix_a[14], B_reindex_shared_dyn_wmma_matrix_b[14], C_reindex_shared_dyn_wmma_accumulator[15]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[11], A_reindex_shared_dyn_wmma_matrix_a[11], B_reindex_shared_dyn_wmma_matrix_b[15], C_reindex_shared_dyn_wmma_accumulator[11]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[15], A_reindex_shared_dyn_wmma_matrix_a[15], B_reindex_shared_dyn_wmma_matrix_b[15], C_reindex_shared_dyn_wmma_accumulator[15]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[16], A_reindex_shared_dyn_wmma_matrix_a[16], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[16]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[20], A_reindex_shared_dyn_wmma_matrix_a[20], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[20]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[16], A_reindex_shared_dyn_wmma_matrix_a[17], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[16]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[20], A_reindex_shared_dyn_wmma_matrix_a[21], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[20]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[16], A_reindex_shared_dyn_wmma_matrix_a[18], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[16]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[20], A_reindex_shared_dyn_wmma_matrix_a[22], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[20]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[16], A_reindex_shared_dyn_wmma_matrix_a[19], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[16]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[20], A_reindex_shared_dyn_wmma_matrix_a[23], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[20]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[17], A_reindex_shared_dyn_wmma_matrix_a[16], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[17]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[21], A_reindex_shared_dyn_wmma_matrix_a[20], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[21]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[17], A_reindex_shared_dyn_wmma_matrix_a[17], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[17]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[21], A_reindex_shared_dyn_wmma_matrix_a[21], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[21]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[17], A_reindex_shared_dyn_wmma_matrix_a[18], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[17]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[21], A_reindex_shared_dyn_wmma_matrix_a[22], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[21]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[17], A_reindex_shared_dyn_wmma_matrix_a[19], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[17]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[21], A_reindex_shared_dyn_wmma_matrix_a[23], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[21]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[18], A_reindex_shared_dyn_wmma_matrix_a[16], B_reindex_shared_dyn_wmma_matrix_b[8], C_reindex_shared_dyn_wmma_accumulator[18]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[22], A_reindex_shared_dyn_wmma_matrix_a[20], B_reindex_shared_dyn_wmma_matrix_b[8], C_reindex_shared_dyn_wmma_accumulator[22]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[18], A_reindex_shared_dyn_wmma_matrix_a[17], B_reindex_shared_dyn_wmma_matrix_b[9], C_reindex_shared_dyn_wmma_accumulator[18]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[22], A_reindex_shared_dyn_wmma_matrix_a[21], B_reindex_shared_dyn_wmma_matrix_b[9], C_reindex_shared_dyn_wmma_accumulator[22]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[18], A_reindex_shared_dyn_wmma_matrix_a[18], B_reindex_shared_dyn_wmma_matrix_b[10], C_reindex_shared_dyn_wmma_accumulator[18]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[22], A_reindex_shared_dyn_wmma_matrix_a[22], B_reindex_shared_dyn_wmma_matrix_b[10], C_reindex_shared_dyn_wmma_accumulator[22]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[18], A_reindex_shared_dyn_wmma_matrix_a[19], B_reindex_shared_dyn_wmma_matrix_b[11], C_reindex_shared_dyn_wmma_accumulator[18]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[22], A_reindex_shared_dyn_wmma_matrix_a[23], B_reindex_shared_dyn_wmma_matrix_b[11], C_reindex_shared_dyn_wmma_accumulator[22]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[19], A_reindex_shared_dyn_wmma_matrix_a[16], B_reindex_shared_dyn_wmma_matrix_b[12], C_reindex_shared_dyn_wmma_accumulator[19]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[23], A_reindex_shared_dyn_wmma_matrix_a[20], B_reindex_shared_dyn_wmma_matrix_b[12], C_reindex_shared_dyn_wmma_accumulator[23]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[19], A_reindex_shared_dyn_wmma_matrix_a[17], B_reindex_shared_dyn_wmma_matrix_b[13], C_reindex_shared_dyn_wmma_accumulator[19]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[23], A_reindex_shared_dyn_wmma_matrix_a[21], B_reindex_shared_dyn_wmma_matrix_b[13], C_reindex_shared_dyn_wmma_accumulator[23]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[19], A_reindex_shared_dyn_wmma_matrix_a[18], B_reindex_shared_dyn_wmma_matrix_b[14], C_reindex_shared_dyn_wmma_accumulator[19]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[23], A_reindex_shared_dyn_wmma_matrix_a[22], B_reindex_shared_dyn_wmma_matrix_b[14], C_reindex_shared_dyn_wmma_accumulator[23]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[19], A_reindex_shared_dyn_wmma_matrix_a[19], B_reindex_shared_dyn_wmma_matrix_b[15], C_reindex_shared_dyn_wmma_accumulator[19]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[23], A_reindex_shared_dyn_wmma_matrix_a[23], B_reindex_shared_dyn_wmma_matrix_b[15], C_reindex_shared_dyn_wmma_accumulator[23]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[24], A_reindex_shared_dyn_wmma_matrix_a[24], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[24]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[28], A_reindex_shared_dyn_wmma_matrix_a[28], B_reindex_shared_dyn_wmma_matrix_b[0], C_reindex_shared_dyn_wmma_accumulator[28]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[24], A_reindex_shared_dyn_wmma_matrix_a[25], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[24]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[28], A_reindex_shared_dyn_wmma_matrix_a[29], B_reindex_shared_dyn_wmma_matrix_b[1], C_reindex_shared_dyn_wmma_accumulator[28]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[24], A_reindex_shared_dyn_wmma_matrix_a[26], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[24]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[28], A_reindex_shared_dyn_wmma_matrix_a[30], B_reindex_shared_dyn_wmma_matrix_b[2], C_reindex_shared_dyn_wmma_accumulator[28]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[24], A_reindex_shared_dyn_wmma_matrix_a[27], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[24]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[28], A_reindex_shared_dyn_wmma_matrix_a[31], B_reindex_shared_dyn_wmma_matrix_b[3], C_reindex_shared_dyn_wmma_accumulator[28]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[25], A_reindex_shared_dyn_wmma_matrix_a[24], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[25]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[29], A_reindex_shared_dyn_wmma_matrix_a[28], B_reindex_shared_dyn_wmma_matrix_b[4], C_reindex_shared_dyn_wmma_accumulator[29]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[25], A_reindex_shared_dyn_wmma_matrix_a[25], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[25]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[29], A_reindex_shared_dyn_wmma_matrix_a[29], B_reindex_shared_dyn_wmma_matrix_b[5], C_reindex_shared_dyn_wmma_accumulator[29]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[25], A_reindex_shared_dyn_wmma_matrix_a[26], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[25]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[29], A_reindex_shared_dyn_wmma_matrix_a[30], B_reindex_shared_dyn_wmma_matrix_b[6], C_reindex_shared_dyn_wmma_accumulator[29]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[25], A_reindex_shared_dyn_wmma_matrix_a[27], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[25]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[29], A_reindex_shared_dyn_wmma_matrix_a[31], B_reindex_shared_dyn_wmma_matrix_b[7], C_reindex_shared_dyn_wmma_accumulator[29]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[26], A_reindex_shared_dyn_wmma_matrix_a[24], B_reindex_shared_dyn_wmma_matrix_b[8], C_reindex_shared_dyn_wmma_accumulator[26]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[30], A_reindex_shared_dyn_wmma_matrix_a[28], B_reindex_shared_dyn_wmma_matrix_b[8], C_reindex_shared_dyn_wmma_accumulator[30]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[26], A_reindex_shared_dyn_wmma_matrix_a[25], B_reindex_shared_dyn_wmma_matrix_b[9], C_reindex_shared_dyn_wmma_accumulator[26]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[30], A_reindex_shared_dyn_wmma_matrix_a[29], B_reindex_shared_dyn_wmma_matrix_b[9], C_reindex_shared_dyn_wmma_accumulator[30]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[26], A_reindex_shared_dyn_wmma_matrix_a[26], B_reindex_shared_dyn_wmma_matrix_b[10], C_reindex_shared_dyn_wmma_accumulator[26]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[30], A_reindex_shared_dyn_wmma_matrix_a[30], B_reindex_shared_dyn_wmma_matrix_b[10], C_reindex_shared_dyn_wmma_accumulator[30]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[26], A_reindex_shared_dyn_wmma_matrix_a[27], B_reindex_shared_dyn_wmma_matrix_b[11], C_reindex_shared_dyn_wmma_accumulator[26]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[30], A_reindex_shared_dyn_wmma_matrix_a[31], B_reindex_shared_dyn_wmma_matrix_b[11], C_reindex_shared_dyn_wmma_accumulator[30]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[27], A_reindex_shared_dyn_wmma_matrix_a[24], B_reindex_shared_dyn_wmma_matrix_b[12], C_reindex_shared_dyn_wmma_accumulator[27]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[31], A_reindex_shared_dyn_wmma_matrix_a[28], B_reindex_shared_dyn_wmma_matrix_b[12], C_reindex_shared_dyn_wmma_accumulator[31]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[27], A_reindex_shared_dyn_wmma_matrix_a[25], B_reindex_shared_dyn_wmma_matrix_b[13], C_reindex_shared_dyn_wmma_accumulator[27]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[31], A_reindex_shared_dyn_wmma_matrix_a[29], B_reindex_shared_dyn_wmma_matrix_b[13], C_reindex_shared_dyn_wmma_accumulator[31]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[27], A_reindex_shared_dyn_wmma_matrix_a[26], B_reindex_shared_dyn_wmma_matrix_b[14], C_reindex_shared_dyn_wmma_accumulator[27]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[31], A_reindex_shared_dyn_wmma_matrix_a[30], B_reindex_shared_dyn_wmma_matrix_b[14], C_reindex_shared_dyn_wmma_accumulator[31]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[27], A_reindex_shared_dyn_wmma_matrix_a[27], B_reindex_shared_dyn_wmma_matrix_b[15], C_reindex_shared_dyn_wmma_accumulator[27]);
    nvcuda::wmma::mma_sync(C_reindex_shared_dyn_wmma_accumulator[31], A_reindex_shared_dyn_wmma_matrix_a[31], B_reindex_shared_dyn_wmma_matrix_b[15], C_reindex_shared_dyn_wmma_accumulator[31]);
  }
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9216)])), C_reindex_shared_dyn_wmma_accumulator[0], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9472)])), C_reindex_shared_dyn_wmma_accumulator[1], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9728)])), C_reindex_shared_dyn_wmma_accumulator[2], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9984)])), C_reindex_shared_dyn_wmma_accumulator[3], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(uint4*)(C + (((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 9216));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 32)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 9728));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 64)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 10240));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 96)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 10752));
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9216)])), C_reindex_shared_dyn_wmma_accumulator[4], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9472)])), C_reindex_shared_dyn_wmma_accumulator[5], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9728)])), C_reindex_shared_dyn_wmma_accumulator[6], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9984)])), C_reindex_shared_dyn_wmma_accumulator[7], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 262144)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 9216));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 262176)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 9728));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 262208)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 10240));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 262240)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 10752));
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9216)])), C_reindex_shared_dyn_wmma_accumulator[8], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9472)])), C_reindex_shared_dyn_wmma_accumulator[9], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9728)])), C_reindex_shared_dyn_wmma_accumulator[10], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9984)])), C_reindex_shared_dyn_wmma_accumulator[11], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 524288)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 9216));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 524320)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 9728));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 524352)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 10240));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 524384)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 10752));
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9216)])), C_reindex_shared_dyn_wmma_accumulator[12], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9472)])), C_reindex_shared_dyn_wmma_accumulator[13], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9728)])), C_reindex_shared_dyn_wmma_accumulator[14], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9984)])), C_reindex_shared_dyn_wmma_accumulator[15], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 786432)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 9216));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 786464)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 9728));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 786496)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 10240));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 786528)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 10752));
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9216)])), C_reindex_shared_dyn_wmma_accumulator[16], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9472)])), C_reindex_shared_dyn_wmma_accumulator[17], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9728)])), C_reindex_shared_dyn_wmma_accumulator[18], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9984)])), C_reindex_shared_dyn_wmma_accumulator[19], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1048576)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 9216));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1048608)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 9728));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1048640)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 10240));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1048672)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 10752));
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9216)])), C_reindex_shared_dyn_wmma_accumulator[20], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9472)])), C_reindex_shared_dyn_wmma_accumulator[21], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9728)])), C_reindex_shared_dyn_wmma_accumulator[22], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9984)])), C_reindex_shared_dyn_wmma_accumulator[23], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1310720)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 9216));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1310752)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 9728));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1310784)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 10240));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1310816)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 10752));
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9216)])), C_reindex_shared_dyn_wmma_accumulator[24], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9472)])), C_reindex_shared_dyn_wmma_accumulator[25], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9728)])), C_reindex_shared_dyn_wmma_accumulator[26], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9984)])), C_reindex_shared_dyn_wmma_accumulator[27], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1572864)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 9216));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1572896)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 9728));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1572928)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 10240));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1572960)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 10752));
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9216)])), C_reindex_shared_dyn_wmma_accumulator[28], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9472)])), C_reindex_shared_dyn_wmma_accumulator[29], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9728)])), C_reindex_shared_dyn_wmma_accumulator[30], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 9984)])), C_reindex_shared_dyn_wmma_accumulator[31], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1835008)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 9216));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1835040)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 9728));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1835072)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 10240));
  *(uint4*)(C + ((((((((((int)blockIdx.y) >> 7) * 8388608) + (((int)blockIdx.x) * 2097152)) + ((((int)threadIdx.x) >> 1) * 16384)) + ((((int)blockIdx.y) & 127) * 128)) + (((int)threadIdx.y) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + 1835104)) = *(uint4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)) + 10752));
}

