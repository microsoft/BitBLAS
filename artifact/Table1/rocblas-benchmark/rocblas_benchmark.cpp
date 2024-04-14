#include "tensor.h"
#include <chrono>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <rocblas/internal/rocblas-beta.h>
#include <rocblas/rocblas.h>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>
#define CHECK_ROCBLAS_ERROR(error)                                             \
  if (error != rocblas_status_success) {                                       \
    std::stringstream ss;                                                      \
    ss << "rocBLAS error " << error << " at line " << __LINE__ << std::endl;   \
    throw std::runtime_error(ss.str());                                        \
  }

bool enable_tune = false;
// Vector saves m, n, k, a_t, b_t, enable_tune
std::vector<std::tuple<int, int, int, bool, bool, bool>> inference_server_set =
{
    std::make_tuple(16384, 16384, 16384, false, true, enable_tune),
};

template <typename T1, typename T2>
int time_gemm(Tensor<T1> A, Tensor<T1> B, Tensor<T2> C, bool a_t, bool b_t,
              rocblas_handle rocblas_handle, bool enable_tune = true) {

  int m = b_t ? B.dims()[0] : B.dims()[1];
  int n = a_t ? A.dims()[1] : A.dims()[0];
  int k = a_t ? A.dims()[0] : A.dims()[1];
  int warp_up_iters = 1;
  int numRepeats = 10;

  const int alpha = 1.f;
  const int beta = 1.f;
  rocblas_status stat;

  rocblas_operation transA =
      a_t ? rocblas_operation_transpose : rocblas_operation_none;
  rocblas_operation transB =
      b_t ? rocblas_operation_transpose : rocblas_operation_none;

  rocblas_datatype aType =
      rocblas_datatype_f32_r; // _r for real vs. _c for complex
  rocblas_datatype bType = rocblas_datatype_f32_r;
  rocblas_datatype cType = rocblas_datatype_f32_r;
  rocblas_datatype dType = rocblas_datatype_f32_r;
  rocblas_datatype computeType = rocblas_datatype_f32_r;
  rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
  int32_t solutionIndex = 0;
  uint32_t flags = 0;

  if (std::is_same<T1, half>::value) {
    aType = rocblas_datatype_f16_r;
    bType = rocblas_datatype_f16_r;
    cType = rocblas_datatype_f16_r;
    dType = rocblas_datatype_f16_r;
    computeType = rocblas_datatype_f32_r;
    if (std::is_same<T2, float>::value) {
      cType = rocblas_datatype_f32_r;
      dType = rocblas_datatype_f32_r;
      computeType = rocblas_datatype_f32_r;
    }
  }

  if (std::is_same<T1, uint8_t>::value) {
    aType = rocblas_datatype_i8_r; // _r for real vs. _c for complex
    bType = rocblas_datatype_i8_r;
    cType = rocblas_datatype_i8_r;
    dType = rocblas_datatype_i8_r;
    computeType = rocblas_datatype_i8_r;
    if (std::is_same<T2, uint32_t>::value) {
      cType = rocblas_datatype_i32_r;
      dType = rocblas_datatype_i32_r;
      computeType = rocblas_datatype_i32_r;
    }
  }

  if (enable_tune) {
    auto best_time = std::numeric_limits<double>::max();
    auto best_sol = 0;
    algo = rocblas_gemm_algo_standard;
    solutionIndex = 0;
    flags = rocblas_gemm_flags_none;
    // Get all solutions
    rocblas_int n_solutions;
    CHECK_ROCBLAS_ERROR(rocblas_gemm_ex_get_solutions(
        rocblas_handle, transA, transB, m, n, k, &alpha, A.begin(), aType,
        A.dims()[0], B.begin(), bType, B.dims()[0], &beta, C.begin(), cType,
        C.dims()[0], C.begin(), cType, C.dims()[0], computeType,
        rocblas_gemm_algo_solution_index, rocblas_gemm_flags_none, NULL,
        &n_solutions));

    std::vector<rocblas_int> solutions(n_solutions);
    CHECK_ROCBLAS_ERROR(rocblas_gemm_ex_get_solutions(
        rocblas_handle, transA, transB, m, n, k, &alpha, A.begin(), aType,
        A.dims()[0], B.begin(), bType, B.dims()[0], &beta, C.begin(), cType,
        C.dims()[0], C.begin(), cType, C.dims()[0], computeType,
        rocblas_gemm_algo_solution_index, rocblas_gemm_flags_none,
        solutions.data(), &n_solutions));

    for (auto sol : solutions) {
      // warmup
      for (rocblas_int c = 0; c < warp_up_iters; ++c) {
        // run with solutions
        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(
            rocblas_handle, transB, transA, m, n, k, &alpha, B.begin(), bType,
            B.dims()[1], A.begin(), aType, A.dims()[1], &beta, C.begin(), cType,
            m, C.begin(), cType, m, computeType, algo,
            solutionIndex, flags));
      }
      hipStream_t stream;
      CHECK_ROCBLAS_ERROR(rocblas_get_stream(rocblas_handle, &stream));
      auto start = std::chrono::steady_clock::now();

      // timing loop
      for (rocblas_int c = 0; c < numRepeats; ++c) {
        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(
            rocblas_handle, transB, transA, m, n, k, &alpha, B.begin(), bType,
            B.dims()[1], A.begin(), aType, A.dims()[1], &beta, C.begin(), cType,
            m, C.begin(), cType, m, computeType, algo,
            solutionIndex, flags));
      }

      auto end = std::chrono::steady_clock::now();

      auto time =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count();

      double avg_time = numRepeats ? (time / numRepeats) : 0;
      if (avg_time < best_time) {
        best_sol = sol;
        best_time = avg_time;
      }
    }
    solutionIndex = best_sol;
  }

  auto start = std::chrono::steady_clock::now();

  stat = rocblas_gemm_ex(rocblas_handle, transB, transA, m, n, k, &alpha,
                        B.begin(), bType, B.dims()[1], A.begin(), aType, 
                        A.dims()[1], &beta, C.begin(), cType, m, 
                        C.begin(), cType, m, computeType, algo,
                        solutionIndex, flags);
  if (stat != rocblas_status_success) {
    throw std::runtime_error("gemm failed");
  }

  hipDeviceSynchronize();

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < numRepeats; ++i) {
    stat = rocblas_gemm_ex(rocblas_handle, transB, transA, m, n, k, &alpha,
                          B.begin(), bType, B.dims()[1], A.begin(), aType,
                          A.dims()[1], &beta, C.begin(), cType, m,
                          C.begin(), cType, m, computeType, algo,
                          solutionIndex, flags);

    if (stat != rocblas_status_success) {
      throw std::runtime_error("gemm failed");
    }
  }
  hipDeviceSynchronize();

  auto end = std::chrono::steady_clock::now();

  return static_cast<int>(
      std::chrono::duration<double, std::micro>(end - start).count() /
      numRepeats);
}

int main(int argc, char **argv) {
  int deviceCount = 1;
  int inference = 1;

  for (int dev = 0; dev < deviceCount; ++dev) {
    hipSetDevice(dev);
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, dev);

    std::cout << "Device " << dev << ": " << deviceProp.name << std::endl;

    hiprandGenerator_t hiprand_gen;
    hiprandCreateGenerator(&hiprand_gen, HIPRAND_RNG_PSEUDO_DEFAULT);
    hiprandSetPseudoRandomGeneratorSeed(hiprand_gen, 123ULL);

    rocblas_handle rocblas_handle;
    rocblas_status status = rocblas_create_handle(&rocblas_handle);
    if (status != rocblas_status_success) {
      std::cout << "rocBLAS init failed" << std::endl;
    }

    std::cout << "m,n,k,a_t,b_t,f16-f16 "
                 "time (msec), int8-int32 time (msec)"
              << std::endl;

    int pad_kernels_count = 0;

    for (const auto &problem : inference_server_set) {
      int m, n, k;
      bool a_t, b_t, enable_tune;
      std::tie(m, n, k, a_t, b_t, enable_tune) = problem;
      int time_us;

      std::cout << m << ",";
      std::cout << n << ",";
      std::cout << k << ",";
      std::cout << (a_t ? "t" : "n") << ",";
      std::cout << (b_t ? "t" : "n") << ",";

      // fp16-f16 benchmark
      {
        auto a = rand<half>({a_t ? k : m, a_t ? m : k}, hiprand_gen);
        auto b = rand<half>({b_t ? n : k, b_t ? k : n}, hiprand_gen);
        auto c = zeros<half>({m, n});
        time_us = time_gemm<half, half>(a, b, c, a_t, b_t,
                                        rocblas_handle, enable_tune);
        std::cout << "," << std::setprecision(6) << time_us / 1000.0;
      }

      // int8-int32 benchmark
      {
        int pad_m;
        pad_m = m;
        if (pad_m % 4) {
          pad_kernels_count++;
          pad_dim(pad_m, 4);
        }

        auto a = rand<uint8_t>({a_t ? k : pad_m, a_t ? pad_m : k}, hiprand_gen);
        auto b = rand<uint8_t>({b_t ? n : k, b_t ? k : n}, hiprand_gen);
        auto c = zeros<uint32_t>({pad_m, n});
        time_us = time_gemm<uint8_t, uint32_t>(a, b, c, a_t, b_t,
                                               rocblas_handle, enable_tune);
        std::cout << "," << std::setprecision(6) << time_us / 1000.0;
      }

      std::cout << std::endl;
    }

    rocblas_destroy_handle(rocblas_handle);
    hiprandDestroyGenerator(hiprand_gen);
  }

  return 0;
}
