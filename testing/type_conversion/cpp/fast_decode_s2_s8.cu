#include <stdio.h>
#include <cuda_runtime.h>
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
typedef unsigned int uint;

#define TVM_FORCE_INLINE inline __attribute__((always_inline))
#define TVM_XINLINE TVM_FORCE_INLINE __device__ __host__
#define TVM_ALIGNED(x) __attribute__((aligned(x)))
#define TVM_HALF_OPERATOR(RTYPE, OP)            \
  TVM_XINLINE RTYPE operator OP(half a, half b) \
  {                                             \
    return RTYPE(float(a) OP float(b));         \
  }                                             \
  template <typename T>                         \
  TVM_XINLINE RTYPE operator OP(half a, T b)    \
  {                                             \
    return RTYPE(float(a) OP float(b));         \
  }                                             \
  template <typename T>                         \
  TVM_XINLINE RTYPE operator OP(T a, half b)    \
  {                                             \
    return RTYPE(float(a) OP float(b));         \
  }

#define TVM_HALF_ASSIGNOP(AOP, OP)                            \
  template <typename T>                                       \
  TVM_XINLINE half operator AOP(const T &a)                   \
  {                                                           \
    return *this = half(float(*this) OP float(a));            \
  }                                                           \
  template <typename T>                                       \
  TVM_XINLINE half operator AOP(const volatile T &a) volatile \
  {                                                           \
    return *this = half(float(*this) OP float(a));            \
  }

class TVM_ALIGNED(2) half
{
public:
  uint16_t half_;

  static TVM_XINLINE half Binary(uint16_t value)
  {
    half res;
    res.half_ = value;
    return res;
  }

  TVM_XINLINE half() {}

  TVM_XINLINE half(const float &value) { constructor(value); }
  TVM_XINLINE explicit half(const double &value) { constructor(value); }
  TVM_XINLINE explicit half(const int8_t &value) { constructor(value); }
  TVM_XINLINE explicit half(const uint8_t &value) { constructor(value); }
  TVM_XINLINE explicit half(const int32_t &value) { constructor(value); }
  TVM_XINLINE explicit half(const uint &value) { constructor(value); }
  TVM_XINLINE explicit half(const long long &value) { constructor(value); }
  TVM_XINLINE explicit half(const uint64_t &value) { constructor(value); }

  TVM_XINLINE operator float() const
  {
    return float(half2float(half_));
  }
  TVM_XINLINE operator float() const volatile
  {
    return float(half2float(half_));
  }

  TVM_HALF_ASSIGNOP(+=, +)
  TVM_HALF_ASSIGNOP(-=, -)
  TVM_HALF_ASSIGNOP(*=, *)
  TVM_HALF_ASSIGNOP(/=, /)

  TVM_XINLINE half operator+()
  {
    return *this;
  }

  TVM_XINLINE half operator-()
  {
    return half(-float(*this));
  }

  TVM_XINLINE half operator=(const half &a)
  {
    half_ = a.half_;
    return a;
  }

  template <typename T>
  TVM_XINLINE half operator=(const T &a)
  {
    return *this = half(a);
  }

  TVM_XINLINE half operator=(const half &a) volatile
  {
    half_ = a.half_;
    return a;
  }

  template <typename T>
  TVM_XINLINE half operator=(const T &a) volatile
  {
    return *this = half(a);
  }

private:
  union Bits
  {
    float f;
    int32_t si;
    uint ui;
  };

  static int const fp16FractionBits = 10;
  static int const fp32FractionBits = 23;
  static int32_t const fp32FractionMask = ~(~0u << fp32FractionBits); // == 0x7fffff
  static int32_t const fp32HiddenBit = 1 << fp32FractionBits;         // == 0x800000
  static int const shift = fp32FractionBits - fp16FractionBits;       // == 13
  static int const shiftSign = 16;
  static int32_t const expAdjust = 127 - 15; // exp32-127 = exp16-15, so exp16 = exp32 - (127-15)

  static int32_t const infN = 0x7F800000;  // flt32 infinity
  static int32_t const maxN = 0x477FFFFF;  // max flt32 that's a flt16 normal after >> by shift
  static int32_t const minN = 0x38800000;  // min flt16 normal as a flt32
  static int32_t const maxZ = 0x33000000;  // max fp32 number that's still rounded to zero in fp16
  static int32_t const signN = 0x80000000; // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift; // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign; // flt16 sign bit

  static int32_t const mulN = 0x52000000; // (1 << 23) / minN
  static int32_t const mulC = 0x33800000; // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF; // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400; // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  TVM_XINLINE uint16_t float2half(const float &value) const
  {
    Bits v;
    v.f = value;
    uint sign = v.si & signN; // grab sign bit
    v.si ^= sign;             // clear sign bit from v
    sign >>= shiftSign;       // logical shift sign to fp16 position

    if (v.si <= maxZ)
    {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    }
    else if (v.si < minN)
    {
      // Handle denorms
      uint exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint vshift = 1 - exp16;
      uint significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    }
    else if (v.si <= maxN)
    {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    }
    else if (v.si <= infN)
    {
      v.si = infN;
    }
    else if (v.si < nanN)
    {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  // Same as above routine, except for addition of volatile keyword
  TVM_XINLINE uint16_t float2half(
      const volatile float &value) const volatile
  {
    Bits v;
    v.f = value;
    uint sign = v.si & signN; // grab sign bit
    v.si ^= sign;             // clear sign bit from v
    sign >>= shiftSign;       // logical shift sign to fp16 position

    if (v.si <= maxZ)
    {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    }
    else if (v.si < minN)
    {
      // Handle denorms
      uint exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint vshift = 1 - exp16;
      uint significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    }
    else if (v.si <= maxN)
    {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    }
    else if (v.si <= infN)
    {
      v.si = infN;
    }
    else if (v.si < nanN)
    {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  TVM_XINLINE float half2float(const uint16_t &value) const
  {
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
      const volatile uint16_t &value) const volatile
  {
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

  template <typename T>
  TVM_XINLINE void constructor(const T &value)
  {
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

TVM_XINLINE half __float2half_rn(const float a)
{
  return half(a);
}
#endif

// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y)
{
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Some fp16 math functions are not supported in cuda_fp16.h,
// so we define them here to make sure the generated CUDA code
// is valid.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
  static inline __device__ __host__ half HALF_MATH_NAME(half x, half y)   \
  {                                                                       \
    float tmp_x = __half2float(x);                                        \
    float tmp_y = __half2float(y);                                        \
    float result = FP32_MATH_NAME(tmp_x, tmp_y);                          \
    return __float2half(result);                                          \
  }

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
  static inline __device__ __host__ half HALF_MATH_NAME(half x)          \
  {                                                                      \
    float tmp_x = __half2float(x);                                       \
    float result = FP32_MATH_NAME(tmp_x);                                \
    return __float2half(result);                                         \
  }

CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY

#endif
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
#include <sm_61_intrinsics.h>
#endif

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 1
#else
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 0
#endif

#ifdef _WIN32
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;
using ushort = unsigned short;
using uint64_t = unsigned long long;
#else
#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short
#define int64_t long long
#define uint64_t unsigned long long
#endif
template <typename T1, typename T2>
__device__ void decode_i2s_to_i8s(T1 *_i2s, T2 *_i8s, const int N = 16)
{
  // convert 8 int2b_t to 8 int8b_t -> 2 int32
  uint *i8s = reinterpret_cast<uint *>(_i8s);

  // i2s = {e7,e6,e5,e4,e3,e2,e1,e0}
  // also require interleave {e7,e3,e6,e2,e5,e1,e4,e0}
  uint const i2s = *_i2s;

  // First, we extract the i4s and construct an intermediate fp16 number.
  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;     // 0b11101010
  static constexpr uint BOTTOM_MASK = 0x03030303;          // 0xf -> 0b11 select 0,3
  static constexpr uint I4s_TO_I8s_MAGIC_NUM = 0x00000000; // 1024

#pragma unroll
  for (int i = 0; i < (N / 2); i++)
  {
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(i8s[i])
                 : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(I4s_TO_I8s_MAGIC_NUM), "n"(immLut));
    i8s[i] = __vsubss4(i8s[i], 0x02020202);
  }
}

template <typename T1, typename T2>
__device__ void decode_u2s_to_u8s(T1 *_i2s, T2 *_i8s, const int N = 16)
{
  // convert 8 int2b_t to 8 int8b_t -> 2 int32
  uint *i8s = reinterpret_cast<uint *>(_i8s);

  // i2s = {e7,e6,e5,e4,e3,e2,e1,e0}
  // also require interleave {e7,e3,e6,e2,e5,e1,e4,e0}
  uint const i2s = *_i2s;

  // First, we extract the i4s and construct an intermediate fp16 number.
  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;     // 0b11101010
  static constexpr uint BOTTOM_MASK = 0x03030303;          // 0xf -> 0b11 select 0,3
  static constexpr uint I4s_TO_I8s_MAGIC_NUM = 0x00000000; // 1024

#pragma unroll
  for (int i = 0; i < (N / 2); i++)
  {
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(i8s[i])
                 : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(I4s_TO_I8s_MAGIC_NUM), "n"(immLut));
  }
}

extern "C" __global__ void main_kernel0(int8_t *__restrict__ B, int8_t *__restrict__ B_1, const int N = 16)
{
  // print B
  for (int i = 0; i < N / 4; i++)
  {
    printf("B[%d] = %x\n", i, (int)B[i]);
  }
  decode_i2s_to_i8s(reinterpret_cast<int *>(B), reinterpret_cast<int *>(B_1), 16);
  for (int i = 0; i < N; i++)
  {
    printf("B_1[%d] = %d\n", i, int(B_1[i]));
  }
}

void general_interleave_int8(int8_t *origin_arr, int8_t *interleaved, const int nbit, size_t size_in_bytes, bool verbose = false)
{
  // For fp16 example
  // i4s        {e7,e6,e5,e4,e3,e2,e1,e0}
  //            |-8b-||-8b-||-8b-||-8b-|
  // interleave {e7,e3,e6,e2,e5,e1,e4,e0}
  /*
    BOTTOM_MASK        0    0    0    f    0    0    0    f
    i4s                e7   e3   e6   e2   e5   e1   e4   e0
    selectedVal       0000  e3 0000  e2  0000   e1 0000   e0  // selectedVal = i4s & BOTTOM_MASK
    s[0]              0     e3   0    e2    0   e1   0    e0
  */

  //            |-----8b-------||-------8b----||----8b---||-----8b----|
  // i2s        {e15,e14,e13,e12,e11,e10,e9,e8,e7,e6,e5,e4,e3,e2,e1,e0}
  // interleave {e15,e11,e7,e3,e14,e10,e6,e2,e13,e9,e5,e1,e12,e8,e4,e0}

  //            |-------------8b----------------||--------------8b--------------||------------8b--------------||--------8b-----------|
  // i1s        {e31,e30,e29,e28,e27,e26,e25,e24,e23,e22,e21,e20,e19,e18,e17,e16,e15,e14,e13,e12,e11,e10,e9,e8,e7,e6,e5,e4,e3,e2,e1,e0}
  // interleave {e31,e27,e23,e19,e15,e11,e7,e3,e30,e26,e22,e18,e14,e10,e6,e2,e29,e25,e21,e17,e13,e9,e5,e1,e28,e24,e20,e16,e12,e8,e4,e0}
  // Assuming size is the number of int32 elements in origin_arr
  size_t size = size_in_bytes / sizeof(int32_t);
  int32_t *int32_origin = (int32_t *)origin_arr;
  int32_t *int32_interleaved = (int32_t *)interleaved;

  constexpr int bits_stride = 8;
  int elems_per_group = bits_stride / nbit;
  int mask = (1 << nbit) - 1;
  int num_groups = 32 / bits_stride;

  for (int idx = 0; idx < size; ++idx)
  {
    int32_t current_value = int32_origin[idx];
    int32_t new_value = 0;
    for (int i = 0; i < num_groups; ++i)
    {
      for (int j = 0; j < elems_per_group; ++j)
      {
        int offset = i * elems_per_group + j;
        int shift = (offset % num_groups) * bits_stride + (offset / num_groups) * nbit;
        int group_value = (current_value >> (nbit * (i * elems_per_group + j))) & mask;
        new_value |= group_value << shift;
        if (verbose)
          printf("put %d to %d\n", offset, shift);
      }
    }

    int32_interleaved[idx] = new_value;
  }

  // Convert back to int8_t if needed
  memcpy(interleaved, int32_interleaved, size * sizeof(int32_t));
}

void general_compress(const int8_t *lowbit, int8_t *compressed, const int nbit, const int N)
{
  const int nbit_per_byte = 8 / nbit;

  for (int i = 0; i < N / nbit_per_byte; i++)
  {
    for (int j = 0; j < nbit_per_byte; j++)
    {
      compressed[i] |= (lowbit[nbit_per_byte * i + j] << (nbit * j));
    }
  }
}
// nvcc -gencode arch=compute_80,code=sm_80 -lineinfo -O3 fast_decode_s2_s8.cu -o fast_decode_s2_s8; ./fast_decode_s2_s8
int main()
{
  const int N = 16;
  // create four int8_t values
  int8_t *i2s = new int8_t[16];
  i2s[0] = 3;
  i2s[1] = 3;
  i2s[2] = 2;
  i2s[3] = 1;
  i2s[4] = 1;
  i2s[5] = 1;
  i2s[6] = 0;
  i2s[7] = 2;
  i2s[8] = 1;
  i2s[9] = 2;
  i2s[10] = 1;
  i2s[11] = 1;
  i2s[12] = 1;
  i2s[13] = 2;
  i2s[14] = 1;
  i2s[15] = 1;
  // compressed_int8: cmpress 8 i4s to 4 i8s
  int8_t *i8s = new int8_t[4];
  general_compress(i2s, i8s, 2, 16);
  int8_t *interleaved = new int8_t[4];
  general_interleave_int8(i8s, interleaved, 2, 4 * sizeof(int8_t), true);

  printf("before interleave: ");
  printf("int-B = %x\n", reinterpret_cast<int *>(i8s)[0]);
  printf("after interleave: ");
  printf("int-B = %x\n", reinterpret_cast<int *>(interleaved)[0]);

  int8_t *B_local_decode = new int8_t[16];
  int8_t *i8s_gpu;
  int8_t *B_local_decode_gpu;

  cudaMalloc((void **)&i8s_gpu, 4 * sizeof(int8_t));
  cudaMalloc((void **)&B_local_decode_gpu, 4 * sizeof(half));
  cudaMemcpy(i8s_gpu, interleaved, 4 * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(B_local_decode_gpu, B_local_decode, 4 * sizeof(half), cudaMemcpyHostToDevice);
  // print the last error
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n",
           cudaGetErrorString(cudaerr));
  main_kernel0<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(i8s_gpu, B_local_decode_gpu);
  // print error
  cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n",
           cudaGetErrorString(cudaerr));
  cudaMemcpy(B_local_decode, B_local_decode_gpu, 4 * sizeof(half), cudaMemcpyDeviceToHost);

  return 0;
}
