tvm_rt_header = """
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/packed_func.h>
"""

cuda_default_header = """
#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <mma.h>

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 1
#else
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 0
#endif

#define uchar unsigned char

template <typename T1, typename T2>
__device__ void decode_i1s_to_i8s_l16(T1 *_i1s, T2 *_i8s, const int N = 16)
{
  int *i8s = reinterpret_cast<int *>(_i8s);
  int16_t i1s_i16 = *reinterpret_cast<int16_t *>(_i1s);
  // permutate: {e0,e4,e8,e12,e2,e6,e10,e14,e1,e5,e9,e13,e3,e7,e11,e15}
  // into: {e0,e4,e8,e12,x,x,x,x,e1,e5,e9,x,x,x,x,e13,e2,e6,e10,e14,e1,e5,e9,e13,e3,e7,e11,e15,x,x,x,x}
  int i1s = (i1s_i16 & 0x0f0f);
  i1s |= ((i1s_i16 & 0xf0f0) << 12); 
  // i1s        {0..,e15,e14,e13,e12,e11,e10,e9,e8,e7,e6,e5,e4,e3,e2,e1,e0}
  // interleave {0..,e15,e13,e11,e9,e7,e5,e3,e1,e14,e12,e10,e8,e6,e4,e2,e0}
  // First, we extract the i1s and construct an intermediate fp16 number.
  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa; // 0b11101010
  static constexpr uint BOTTOM_MASK = 0x01010101;      // 0x1 -> 0b01 select 0,1
  static constexpr uint I8s_MAGIC_NUM = 0x00000000;

  for (int i = 0; i < N / 4; i++)
  {
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                 : "=r"(i8s[i])
                 : "r"(i1s >> i), "n"(BOTTOM_MASK), "n"(I8s_MAGIC_NUM), "n"(immLut));
  }
}


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
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                 : "=r"(i8s[i])
                 : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(I4s_TO_I8s_MAGIC_NUM), "n"(immLut));
  }
}


template <typename T1, typename T2>
__device__ void decode_i4s_to_i8s(T1 *_i4s, T2 *_i8s, const int N = 16)
{
  uint *i8s = reinterpret_cast<uint *>(_i8s);
  uint *i4s = reinterpret_cast<uint *>(_i4s);
  // First, we extract the i4s and construct an intermediate fp16 number.
  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;     // 0b11101010
  static constexpr uint BOTTOM_MASK = 0x0f0f0f0f;          // 0xf -> 0b1111 select 0,4
  static constexpr uint I4s_TO_I8s_MAGIC_NUM = 0x00000000; // 1024
#pragma unroll
  for (int i = 0; i < (N / 8); i++)
  {
    // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                 : "=r"(i8s[i])
                 : "r"(i4s[0] >> (4 * i)), "n"(BOTTOM_MASK), "n"(I4s_TO_I8s_MAGIC_NUM), "n"(immLut));

    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
              : "=r"(i8s[i + 2])
              : "r"(i4s[1] >> (4 * i)), "n"(BOTTOM_MASK), "n"(I4s_TO_I8s_MAGIC_NUM), "n"(immLut));
  }
}
"""

cuda_fp16_header = """
#include <cuda_fp16.h>

namespace {

__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}

#define __int8_t_defined

#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
inline __device__ half HALF_MATH_NAME(half x, half y) {            \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
inline __device__ half HALF_MATH_NAME(half x) {                   \
  float tmp_x = __half2float(x);                                         \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2half(result);                                           \
}

CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY

  
// Pack two half values.
inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// There is no make_int8 in cuda, but TVM codegen seem to use it
inline __device__ longlong4 make_int8(int x0, int x1, int x2, int x3, int x4, int x5, int x6, int x7) {
  int2 i0 = make_int2(x0, x1);
  int2 i1 = make_int2(x2, x3);
  int2 i2 = make_int2(x4, x5);
  int2 i3 = make_int2(x6, x7);
  long long l0 = *(long long*)&i0;
  long long l1 = *(long long*)&i1;
  long long l2 = *(long long*)&i2;
  long long l3 = *(long long*)&i3;
  return make_longlong4(l0, l1, l2, l3);
}

template<int row_size, int col_size, int panel_width>
__device__ int rasterization2DRow(int idx) {
  const int block_size = row_size * col_size;
  const int panel_size = panel_width * col_size;
  const int block_offset = idx % block_size;
  const int block_idx = idx / block_size;
  const int panel_offset = block_offset % panel_size;
  const int panel_idx = block_offset / panel_size;
  const int total_panel = (block_size + panel_size - 1) / panel_size;
  const int stride = panel_idx + 1 < total_panel ? panel_width : (block_size - panel_idx * panel_size) / col_size;
  const int col_idx = (panel_idx & 1) ? col_size - 1 - panel_offset / stride : panel_offset / stride;
  const int row_idx = panel_offset % stride + panel_idx * panel_width;
  return block_idx * block_size + row_idx * col_size + col_idx;
}

template<int row_size, int col_size, int panel_width>
__device__ int rasterization2DColumn(int idx) {
  const int block_size = row_size * col_size;
  const int panel_size = panel_width * row_size;
  const int block_offset = idx % block_size;
  const int block_idx = idx / block_size;
  const int panel_offset = block_offset % panel_size;
  const int panel_idx = block_offset / panel_size;
  const int total_panel = (block_size + panel_size - 1) / panel_size;
  const int stride = panel_idx + 1 < total_panel ? panel_width : (block_size - panel_idx * panel_size) / row_size;
  const int row_idx = (panel_idx & 1) ? row_size - 1 - panel_offset / stride : panel_offset / stride;
  const int col_idx = panel_offset % stride + panel_idx * panel_width;
  return block_idx * block_size + row_idx * col_size + col_idx;
}

template <typename T1, typename T2>
__device__ void decode_i1s_to_f16(T1 *_i1s, T2 *B_local_decode, const int N = 32)
{
  uint *h = reinterpret_cast<uint *>(B_local_decode);

  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint BOTTOM_MASK = 0x00010001;
  static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
  uint const i1s = *reinterpret_cast<uint *>(_i1s);
#pragma unroll
  // decode 2 elems at one time.
  for (int i = 0; i < (N / 2); i++)
  {

    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                 : "=r"(h[i])
                 : "r"(i1s >> (1 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
    asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(FP16_TOP_MAGIC_NUM));
  }
}

template <typename T1, typename T2>
__device__ void decode_i2s_to_f16(T1 *_i2s, T2 *B_local_decode, const int N = 16)
{
 uint *h = reinterpret_cast<uint *>(B_local_decode);

  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint BOTTOM_MASK = 0x00030003;
  static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
  uint const i2s = *reinterpret_cast<uint *>(_i2s);
#pragma unroll
  // decode 2 elems at one time.
  for (int i = 0; i < (N / 2); i++)
  {

    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                 : "=r"(h[i])
                 : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
    asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(FP16_TOP_MAGIC_NUM));
  }
}

template <typename T1, typename T2>
__device__ void decode_i4s_to_f16(T1 *_i4s, T2 *B_local_decode, const int N = 8)
{
  uint *h = reinterpret_cast<uint *>(B_local_decode);

  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint BOTTOM_MASK = 0x000f000f;
  static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
  uint const i4s = *reinterpret_cast<uint *>(_i4s);
#pragma unroll
  // decode 2 elems at one time.
  for (int i = 0; i < (N / 2); i++)
  {

    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                 : "=r"(h[i])
                 : "r"(i4s >> (4 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
    asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(FP16_TOP_MAGIC_NUM));
  }
}


}
"""

rocm_default_header = """
#include <hip/hip_runtime.h>
#include <math.h>
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))
"""

rocm_fp16_header = """
#define half _Float16
#define __float2half_rn(x) half(x)

#define htanh tanhf
#define htan tanf
#define hatan atanf
#define herf erff
#include <hip/hcc_detail/hip_fp16_math_fwd.h>
#define hpow __ocml_pown_f16
#define hsqrt __ocml_sqrt_f16
#define hexp __ocml_exp_f16

// Pack two half values.
inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// There is no make_int8 in cuda, but TVM codegen seem to use it
inline __device__ longlong4 make_int8(int x0, int x1, int x2, int x3, int x4, int x5, int x6, int x7) {
  int2 i0 = make_int2(x0, x1);
  int2 i1 = make_int2(x2, x3);
  int2 i2 = make_int2(x4, x5);
  int2 i3 = make_int2(x6, x7);
  long long l0 = *(long long*)&i0;
  long long l1 = *(long long*)&i1;
  long long l2 = *(long long*)&i2;
  long long l3 = *(long long*)&i3;
  return make_longlong4(l0, l1, l2, l3);
}

using float16_t = _Float16;
using float16x2
 = __attribute__((__vector_size__(2 * sizeof(float16_t)))) float16_t;
using float16x4
 = __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
using float16x8
 = __attribute__((__vector_size__(8 * sizeof(float16_t)))) float16_t;
using float16x16
 = __attribute__((__vector_size__(16 * sizeof(float16_t)))) float16_t;
using int32x4
 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using float32x4
 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float32x16
 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
 
"""

cutlass_header = """
#include "cutlass/cutlass.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_sm70.h"

namespace cutlass {
namespace gemm {
namespace warp {

template<class MmaWarp, int KSize>
class MMAWarpWrapper {
public:
  typename MmaWarp::FragmentA frag_A[2];
  typename MmaWarp::FragmentB frag_B[2];
  typename MmaWarp::FragmentC accum;
  MmaWarp mma_op;
  typename MmaWarp::IteratorA iter_A;
  typename MmaWarp::IteratorB iter_B;
  const int warp_idx_m_, warp_idx_n_, lane_id_;

  using TensorRefA = typename MmaWarp::IteratorA::TensorRef;
  using TensorRefB = typename MmaWarp::IteratorB::TensorRef;
  static_assert(KSize % MmaWarp::Shape::kK == 0);
  static int constexpr kKgroups = KSize / MmaWarp::Shape::kK;

  CUTLASS_DEVICE
  MMAWarpWrapper(int warp_idx_m, int warp_idx_n, int lane_id)
  : warp_idx_m_(warp_idx_m), warp_idx_n_(warp_idx_n), lane_id_(lane_id), iter_A({nullptr, 0}, 0), iter_B({nullptr, 0}, 0) {
    accum.clear();
  }

  CUTLASS_DEVICE
  void prologue(const TensorRefA &ref_A, const TensorRefB &ref_B) {
    iter_A = typename MmaWarp::IteratorA(ref_A, lane_id_);
    iter_B = typename MmaWarp::IteratorB(ref_B, lane_id_);
    iter_A.add_tile_offset({warp_idx_m_, 0});
    iter_B.add_tile_offset({0, warp_idx_n_});
    iter_A.load(frag_A[0]);
    iter_B.load(frag_B[0]);
    ++iter_A;
    ++iter_B;
  }
  CUTLASS_DEVICE
  void body() {
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kKgroups - 1; ++k) {
      iter_A.load(frag_A[(k + 1) % 2]);
      iter_B.load(frag_B[(k + 1) % 2]);
      ++iter_A;
      ++iter_B;
      mma_op(accum, frag_A[k % 2], frag_B[k % 2], accum);
    }
    __syncthreads();
  }
  CUTLASS_DEVICE
  void epilogue() {
    mma_op(accum, frag_A[(kKgroups - 1) % 2], frag_B[(kKgroups - 1) % 2], accum);
  }
};

template <
  typename Shape,
  typename SMemLayoutA,
  typename SMemLayoutB
>
class GemmTensorOp {
public:
  using InstructionShape = GemmShape<16, 8, 16>;
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      InstructionShape,
      32,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::ColumnMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using MmaWarp = typename cutlass::gemm::warp::MmaTensorOp<
    GemmShape<Shape::kM, Shape::kN, InstructionShape::kK>,
    cutlass::half_t,
    SMemLayoutA,
    cutlass::half_t,
    SMemLayoutB,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    Policy
  >;
  using MMA = MMAWarpWrapper<MmaWarp, Shape::kK>;
  MMA mma;

  CUTLASS_DEVICE
  GemmTensorOp(int warp_idx_m, int warp_idx_n, int lane_id)
  : mma(warp_idx_m, warp_idx_n, lane_id) {}
  CUTLASS_DEVICE
  half& operator[](size_t i) const {
    return ((half*)mma.accum.data())[i];
  }
  CUTLASS_DEVICE
  half* operator+(size_t i) const {
    return (half*)mma.accum.data() + i;
  }
};

template <
  typename Shape,
  typename SMemLayoutA,
  typename LayoutA,
  typename SMemLayoutB,
  typename LayoutB,
  typename LayoutC
>
class VoltaGemmTensorOp {
public:
  using InstructionShape = GemmShape<16, 16, 4>;
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      InstructionShape,
      32,
      cutlass::half_t,
      LayoutA,
      cutlass::half_t,
      LayoutB,
      cutlass::half_t,
      LayoutC,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using MmaWarp = typename cutlass::gemm::warp::MmaVoltaTensorOp<
    GemmShape<Shape::kM, Shape::kN, InstructionShape::kK>,
    cutlass::half_t,
    SMemLayoutA,
    cutlass::half_t,
    SMemLayoutB,
    cutlass::half_t,
    LayoutC,
    Policy
  >;
  using MMA = MMAWarpWrapper<MmaWarp, Shape::kK>;
  MMA mma;

  CUTLASS_DEVICE
  VoltaGemmTensorOp(int warp_idx_m, int warp_idx_n, int lane_id)
  : mma(warp_idx_m, warp_idx_n, lane_id) {}
  CUTLASS_DEVICE
  half& operator[](size_t i) const {
    return ((half*)mma.accum.data())[i];
  }
  CUTLASS_DEVICE
  half* operator+(size_t i) const {
    return (half*)mma.accum.data() + i;
  }
};

template <
  typename Shape,
  typename SMemLayoutA,
  typename SMemLayoutB
>
class GemmI8TensorOp {
public:
  using InstructionShape = GemmShape<16, 8, 32>;
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      InstructionShape,
      32,
      int8_t,
      cutlass::layout::RowMajor,
      int8_t,
      cutlass::layout::ColumnMajor,
      int,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using MmaWarp = typename cutlass::gemm::warp::MmaTensorOp<
    GemmShape<Shape::kM, Shape::kN, InstructionShape::kK>,
    int8_t,
    SMemLayoutA,
    int8_t,
    SMemLayoutB,
    int,
    cutlass::layout::RowMajor,
    Policy
  >;
  using MMA = MMAWarpWrapper<MmaWarp, Shape::kK>;
  MMA mma;

  CUTLASS_DEVICE
  GemmI8TensorOp(int warp_idx_m, int warp_idx_n, int lane_id)
  : mma(warp_idx_m, warp_idx_n, lane_id) {}
  CUTLASS_DEVICE
  int& operator[](size_t i) const {
    return ((int*)mma.accum.data())[i];
  }
  CUTLASS_DEVICE
  int* operator+(size_t i) const {
    return (int*)mma.accum.data() + i;
  }
};

}}}

template<class TensorOp>
CUTLASS_DEVICE void call_cutlass_mma_body(TensorOp& op) {
  op.mma.body();
}

template<class TensorOp>
CUTLASS_DEVICE void call_cutlass_mma_epilogue(TensorOp& op) {
  op.mma.epilogue();
}

template<class TensorOp>
CUTLASS_DEVICE void call_cutlass_mma_prologue(TensorOp& op, void* pA, void* pB, int sA, int sB) {
  using TensorRefA = typename TensorOp::MMA::TensorRefA;
  using TensorRefB = typename TensorOp::MMA::TensorRefB;
  TensorRefA refA{(typename TensorRefA::Element*)pA, sA};
  TensorRefB refB{(typename TensorRefB::Element*)pB, sB};
  op.mma.prologue(refA, refB);
}

#define ALLOCATE_CUTLASS_OBJECT(var, ...) auto var = __VA_ARGS__;
"""
