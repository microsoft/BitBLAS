#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <cublasLt.h>
#include <cuda.h>
#include <curand.h>

#include "tensor.h"

#ifndef PAD_KERNELS
#define PAD_KERNELS 1
#endif

#define CUDA_ERROR(x) \
    do { \
        throw std::runtime_error(std::string(__FILE__ ":") + std::to_string(__LINE__) +            \
                                 " in function " + __func__ + ": " + x);                           \
    } while (false)

inline void check_cublas_(cublasStatus_t status) {
    if ( status != CUBLAS_STATUS_SUCCESS ) {
        CUDA_ERROR("CUBLAS Error: " + std::string(cublasGetStatusString(status)));
    }
}

#define CUDA_CHECK_CUBLAS(ans) { check_cublas_(ans); }

// Vector saves m, n, k, a_t, b_t
std::vector<std::tuple<int, int, int, bool, bool>> inference_server_set = {
    // gemm
    std::make_tuple(16384, 16384, 16384, false, true),
    std::make_tuple(4096, 4096, 1024, false, true),
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

template <typename T1, typename T2, bool is_fp_e5m2=false, bool is_fp_e4m3=false>
int time_gemm(Tensor<T1> A, Tensor<T1> B, Tensor<T2> C, bool a_t, bool b_t,
              cublasLtHandle_t cublaslt_handle, bool use_tensor_core)
{
    const int alpha = 1.f;
    const int beta = 1.f;
    cublasOperation_t transa = a_t ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = b_t ? CUBLAS_OP_T : CUBLAS_OP_N;
    int m = a_t ? A.dims()[0] : A.dims()[1];
    int n = b_t ? B.dims()[1] : B.dims()[0];
    int k = a_t ? A.dims()[1] : A.dims()[0];

    int lda = a_t ? k : m;
    int ldb = b_t ? n : k;
    int ldc = m;

    void *workspace = nullptr;
    size_t workspaceSizeInBytes = 0;
    cublasLtMatmulDesc_t computeDesc = nullptr;
    cublasLtMatmulAlgo_t algo;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    int numRepeats;
    int minimal_repeat_ms = 100;
    cublasStatus_t stat;

    cudaDataType_t A_type = CUDA_R_8F_E5M2;
    cudaDataType_t B_type = CUDA_R_8F_E5M2;
    cudaDataType_t C_type = CUDA_R_16F;
    cudaDataType_t compute_type = CUDA_R_16F;
    cublasComputeType_t gemm_compute_type = CUBLAS_COMPUTE_32F;

    if (std::is_same<T1, uint16_t>::value || std::is_same<T1, half>::value)
    {
        A_type = CUDA_R_16F;
        B_type = CUDA_R_16F;
        C_type = CUDA_R_16F;
        compute_type = CUDA_R_16F;
        gemm_compute_type = CUBLAS_COMPUTE_16F;
    }
    else if (std::is_same<T1, uint8_t>::value)
    {
        A_type = CUDA_R_8I;
        B_type = CUDA_R_8I;
        C_type = CUDA_R_32I;
        compute_type = CUDA_R_32I;
        gemm_compute_type = CUBLAS_COMPUTE_32I;
        if (is_fp_e5m2)
        {
            A_type = CUDA_R_8F_E5M2;
            B_type = CUDA_R_8F_E4M3;
            C_type = CUDA_R_16F;
            compute_type = CUDA_R_32F;
            gemm_compute_type = CUBLAS_COMPUTE_32F;
            int8_t fastAccuMode = 1;
            cublasLtMatmulDescSetAttribute(computeDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccuMode, sizeof(fastAccuMode));
        }
        if (is_fp_e4m3)
        {
            A_type = CUDA_R_8F_E4M3;
            B_type = CUDA_R_8F_E4M3;
            C_type = CUDA_R_16F;
            compute_type = CUDA_R_32F;
            gemm_compute_type = CUBLAS_COMPUTE_32F;
        }
    }

    CUDA_CHECK_CUBLAS(cublasLtMatmulDescCreate(&computeDesc, gemm_compute_type, compute_type));
    CUDA_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(computeDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                     &transa, sizeof(transa)));
    CUDA_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(computeDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                     &transb, sizeof(transb)));

    CUDA_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, A_type,
                                                    transa == CUBLAS_OP_N ? m : k,
                                                    transa == CUBLAS_OP_N ? k : m,
                                                    lda));
    CUDA_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, B_type,
                                                    transb == CUBLAS_OP_N ? k : n,
                                                    transb == CUBLAS_OP_N ? n : k,
                                                    ldb));
    CUDA_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, C_type, m, n, ldc));


    CUDA_CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CUDA_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
                            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                            &workspaceSizeInBytes, sizeof(workspaceSizeInBytes)));
    int                             returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    CUDA_CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, computeDesc, Adesc, Bdesc, Cdesc,
                                                     Cdesc, preference, 1, &heuristicResult,
                                                     &returnedResults));
    if (returnedResults == 0) throw std::runtime_error("Unable to find any suitable algorithms");
    // check last error
    auto error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
    
    auto warmup_start = std::chrono::steady_clock::now();
    stat = cublasLtMatmul(
            cublaslt_handle,
            computeDesc,
            &alpha,
            A.begin(),
            Adesc,
            B.begin(),
            Bdesc,
            &beta,
            C.begin(),
            Cdesc,
            C.begin(),
            Cdesc,
            &heuristicResult.algo,
            workspace,
            workspaceSizeInBytes,
            0
        );

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS gemm failed the stat is " << stat << std::endl;
        // get status string
        std::cout << "CUBLAS gemm failed the stat is " << cublasGetStatusString(stat) << std::endl;
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
        stat = cublasLtMatmul(
            cublaslt_handle,
            computeDesc,
            &alpha,
            A.begin(),
            Adesc,
            B.begin(),
            Bdesc,
            &beta,
            C.begin(),
            Cdesc,
            C.begin(),
            Cdesc,
            &heuristicResult.algo,
            workspace,
            workspaceSizeInBytes,
            0
        );

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
    bool enable_fp8 = false;
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
        if (deviceProp.major > 8 || (deviceProp.major == 8 && deviceProp.minor >= 9)) {
            std::cout << "SM version is greater than 89" << std::endl;
            enable_fp8 = true;
        } else {
            std::cout << "SM version is not greater than 89" << std::endl;
        }
        curandGenerator_t curand_gen;
        curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(curand_gen, 123ULL);

        cublasLtHandle_t cublaslt_handle;
        cublasStatus_t status = cublasLtCreate(&cublaslt_handle);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            std::cout << "CUBLAS init failed" << std::endl;
        }

        std::cout
            << "m,n,k,a_t,b_t,fp32 time (usec),fp16 time (usec),int8 time "
               "(usec),fp16 tensor core time (usec),int8 tensor core time (usec)";
        if (enable_fp8){
            std::cout << "fp8_e5m2 tensor core time (usec), fp8_e4m3 tensor core time (usec)";
        }
        std::cout << std::endl;

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
            
            // fp16 benchmark
            {
                auto a = rand<uint16_t>({a_t ? k : m, a_t ? m : k}, curand_gen);
                auto b = rand<uint16_t>({b_t ? n : k, b_t ? k : n}, curand_gen);
                auto c = zeros<uint16_t>({m, n});
                time_us = time_gemm<uint16_t, uint16_t>(b, a, c, b_t, a_t, 
                                                        cublaslt_handle, false);
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
                    time_gemm<uint8_t, int>(b, a, c, b_t, a_t, cublaslt_handle, true);
                std::cout << "," << std::setprecision(6) << time_us / 1000.0;
            }

            // fp8_e5m2 tensor core benchmark
            {
        
                auto a = rand<uint8_t>({a_t ? k : m, a_t ? m : k}, curand_gen);
                auto b = rand<uint8_t>({b_t ? n : k, b_t ? k : n}, curand_gen);
                auto c = zeros<half>({m, n});
                time_us =
                    time_gemm<uint8_t, half, false, true>(b, a, c, b_t, a_t, cublaslt_handle, true);
                std::cout << "," << std::setprecision(6) << time_us / 1000.0;
            }

            std::cout << std::endl;
        }

        cublasLtDestroy(cublaslt_handle);
        curandDestroyGenerator(curand_gen);
    }

    return 0;
}