#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>

#include "tensor.h"

#ifndef PAD_KERNELS
#define PAD_KERNELS 1
#endif

// Vector saves m, n, k, a_t, b_t
std::vector<std::tuple<int, int, int, bool, bool>> inference_server_set = {

    // gemv
    std::make_tuple(1, 16384, 16384, false, true),
    std::make_tuple(1, 43008, 14336, false, true),
    std::make_tuple(1, 14336, 14336, false, true),
    std::make_tuple(1, 57344, 14336, false, true),
    std::make_tuple(1, 14336, 57344, false, true),
    std::make_tuple(1, 9216, 9216, false, true),
    std::make_tuple(1, 36864, 9216, false, true),
    std::make_tuple(1, 9216, 36864, false, true),
    std::make_tuple(1, 22016, 8192, false, true),
    std::make_tuple(1, 8192, 22016, false, true),
    std::make_tuple(1, 8192, 8192, false, true),
    std::make_tuple(1, 28672, 8192, false, true),
    std::make_tuple(1, 8192, 28672, false, true),

    // gemm
    std::make_tuple(16384, 16384, 16384, false, true),
    std::make_tuple(8192, 43008, 14336, false, true),
    std::make_tuple(8192, 14336, 14336, false, true),
    std::make_tuple(8192, 57344, 14336, false, true),
    std::make_tuple(8192, 14336, 57344, false, true),
    std::make_tuple(8192, 9216, 9216, false, true),
    std::make_tuple(8192, 36864, 9216, false, true),
    std::make_tuple(8192, 9216, 36864, false, true),
    std::make_tuple(8192, 22016, 8192, false, true),
    std::make_tuple(8192, 8192, 22016, false, true),
    std::make_tuple(8192, 8192, 8192, false, true),
    std::make_tuple(8192, 28672, 8192, false, true),
    std::make_tuple(8192, 8192, 28672, false, true),

};

/*
Usage:

The default precision is set based on the architecture and mode.

By default, the program runs the benchmark in training mode.

bin/gemm_bench

To run inference mode, use the following command:

bin/gemm_bench inference


To change the precision for training/inference, use:

bin/gemm_bench train <precision>
bin/gemm_bench inference <precision>

Supported precision types:

For Maxwell GPUS:
float for training and inference

For Pascal GPUS:
float, half for training
float, half, int8 for inference

*/

template <typename T1, typename T2>
int time_gemm(Tensor<T1> A, Tensor<T1> B, Tensor<T2> C, bool a_t, bool b_t,
              cublasHandle_t cublas_handle, bool use_tensor_core)
{
    const int alpha = 1.f;
    const int beta = 1.f;

    int m = b_t ? B.dims()[0] : B.dims()[1];
    int n = a_t ? A.dims()[1] : A.dims()[0];
    int k = a_t ? A.dims()[0] : A.dims()[1];

    int numRepeats;
    int minimal_repeat_ms = 100;
    cublasStatus_t stat;

    cudaDataType_t A_type = CUDA_R_32F;
    cudaDataType_t B_type = CUDA_R_32F;
    cudaDataType_t C_type = CUDA_R_32F;
    cudaDataType_t compute_type = CUDA_R_32F;
    cublasGemmAlgo_t algo;

    if (std::is_same<T1, uint16_t>::value || std::is_same<T1, half>::value)
    {
        A_type = CUDA_R_16F;
        B_type = CUDA_R_16F;
        C_type = CUDA_R_16F;
        compute_type = CUDA_R_16F;
    }

    if (std::is_same<T1, uint8_t>::value)
    {
        A_type = CUDA_R_8I;
        B_type = CUDA_R_8I;
        C_type = CUDA_R_32I;
        compute_type = CUDA_R_32I;
    }

    algo = use_tensor_core ? CUBLAS_GEMM_DFALT_TENSOR_OP : CUBLAS_GEMM_DFALT;
    auto warmup_start = std::chrono::steady_clock::now();

    stat =
        cublasGemmEx(cublas_handle,
                     b_t ? CUBLAS_OP_T : CUBLAS_OP_N, a_t ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha, B.begin(), B_type, B.dims()[1], A.begin(),
                     A_type, A.dims()[1], &beta,
                     C.begin(), C_type, m, compute_type, algo);

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error("sgemm failed");
    }
    cudaDeviceSynchronize();

    auto warmup_end = std::chrono::steady_clock::now();
    auto periter_duration = static_cast<int>(
                                std::chrono::duration<double, std::micro>(warmup_end - warmup_start).count()) *
                            1e3;

    numRepeats = std::max(5, int(minimal_repeat_ms / periter_duration));

    cudaDeviceSynchronize();

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < numRepeats; ++i)
    {
        stat = cublasGemmEx(cublas_handle,
                            b_t ? CUBLAS_OP_T : CUBLAS_OP_N, a_t ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha, B.begin(), B_type, B.dims()[1], A.begin(),
                            A_type, A.dims()[1], &beta,
                            C.begin(), C_type, m, compute_type, algo);

        if (stat != CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error("sgemm failed");
        }
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();

    return static_cast<int>(
        std::chrono::duration<double, std::micro>(end - start).count() /
        numRepeats);
}

int main(int argc, char **argv)
{
    // Get Device Number
    // int deviceCount = 0;
    // cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    // if (error_id != cudaSuccess)
    // {
    //     printf("cudaGetDeviceCount returned %d\n-> %s\n",
    //            static_cast<int>(error_id), cudaGetErrorString(error_id));
    //     printf("Result = FAIL\n");
    //     exit(EXIT_FAILURE);
    // }
    int deviceCount = 1;
    int inference = 1;
    if (argc > 1)
    {
        std::string inf = "inference";
        inference = argv[1] == inf ? 1 : 0;
    }

    if (inference)
    {
        std::cout << "Running inference benchmark " << std::endl;
    }
    else
    {
        std::cout << "Running training benchmark " << std::endl;
    }

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << ": " << deviceProp.name << std::endl;

        curandGenerator_t curand_gen;
        curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(curand_gen, 123ULL);

        cublasHandle_t cublas_handle;
        cublasStatus_t status = cublasCreate(&cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            std::cout << "CUBLAS init failed" << std::endl;
        }

        std::cout
            << "m,n,k,a_t,b_t,fp32 time (usec),fp16 time (usec),int8 time "
               "(usec),fp16 tensor core time (usec),int8 tensor core time (usec)"
            << std::endl;

        int pad_kernels_count = 0;

        for (const auto &problem : inference_server_set)
        {
            int m, n, k;
            bool a_t, b_t;
            std::tie(m, n, k, a_t, b_t) = problem;
            int time_us;

            std::cout << m << ",";
            std::cout << n << ",";
            std::cout << k << ",";
            std::cout << (a_t ? "t" : "n") << ",";
            std::cout << (b_t ? "t" : "n") << ",";

            // set cublas to not use tensor core
            status = cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                std::cout << "CUBLAS math mode failed" << std::endl;
            }

            // fp32 benchmark
            {
                auto a = rand<float>({a_t ? k : m, a_t ? m : k}, curand_gen);
                auto b = rand<float>({b_t ? n : k, b_t ? k : n}, curand_gen);
                auto c = zeros<float>({m, n});
                time_us =
                    time_gemm<float, float>(a, b, c, a_t, b_t, cublas_handle, false);
                std::cout << "," << std::setprecision(6) << time_us / 1000.0;
            }

            // fp16 benchmark
            {
                auto a = rand<uint16_t>({a_t ? k : m, a_t ? m : k}, curand_gen);
                auto b = rand<uint16_t>({b_t ? n : k, b_t ? k : n}, curand_gen);
                auto c = zeros<uint16_t>({m, n});
                time_us = time_gemm<uint16_t, uint16_t>(a, b, c, a_t, b_t,
                                                        cublas_handle, false);
                std::cout << "," << std::setprecision(6) << time_us / 1000.0;
            }

            // int8 benchmark
            {
                int pad_m;
                pad_m = m;
                if (pad_m % 4)
                {
                    pad_kernels_count++;
                    pad_dim(pad_m, 4);
                }

                auto a = rand<uint8_t>({a_t ? k : pad_m, a_t ? pad_m : k}, curand_gen);
                auto b = rand<uint8_t>({b_t ? n : k, b_t ? k : n}, curand_gen);
                auto c = zeros<int>({pad_m, n});
                time_us =
                    time_gemm<uint8_t, int>(a, b, c, a_t, b_t, cublas_handle, false);
                std::cout << "," << std::setprecision(6) << time_us / 1000.0;
            }

            // set cublas to use tensor core
            status = cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
            if (status != CUBLAS_STATUS_SUCCESS)
            {
                std::cout << "CUBLAS math mode failed" << std::endl;
            }

            // fp16 tensor core benchmark
            {
                auto a = rand<half>({a_t ? k : m, a_t ? m : k}, curand_gen);
                auto b = rand<half>({b_t ? n : k, b_t ? k : n}, curand_gen);
                auto c = zeros<half>({m, n});
                time_us = time_gemm<half, half>(a, b, c, a_t, b_t,
                                                cublas_handle, true);
                std::cout << "," << std::setprecision(6) << time_us / 1000.0;
            }

            // int8 tensor core benchmark
            {
                int pad_m;
                pad_m = m;
                if (pad_m % 4)
                {
                    pad_kernels_count++;
                    pad_dim(pad_m, 4);
                }

                auto a = rand<uint8_t>({a_t ? k : pad_m, a_t ? pad_m : k}, curand_gen);
                auto b = rand<uint8_t>({b_t ? n : k, b_t ? k : n}, curand_gen);
                auto c = zeros<int>({pad_m, n});
                time_us =
                    time_gemm<uint8_t, int>(a, b, c, a_t, b_t, cublas_handle, true);
                std::cout << "," << std::setprecision(6) << time_us / 1000.0;
            }

            std::cout << std::endl;
        }

        cublasDestroy(cublas_handle);
        curandDestroyGenerator(curand_gen);
    }

    return 0;
}