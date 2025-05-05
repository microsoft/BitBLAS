// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include <gtest/gtest.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "fast_decoding.hpp"

#define cudaCheckLastError(and)               \
    {                                         \
        gpuAssert((and), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#define REGISTER_GLOBAL_DEVICE_INVOKER(kernel, function) \
    template <typename... Args>                          \
    __global__ void kernel(Args... args)                 \
    {                                                    \
        function(args...);                               \
    }

REGISTER_GLOBAL_DEVICE_INVOKER(kernelWrapper_i2s_to_i4s, decode_i2s_to_i4s)
REGISTER_GLOBAL_DEVICE_INVOKER(kernelWrapper_i2u_to_i4s, decode_i2u_to_i4s)

// TEST(DecodeTest, DecodeInt4ToINT8)
// {
//     using target_dtype = int8_t;
//     constexpr int nbits = 2;
//     constexpr int N = 32 / nbits;
//     constexpr int QN = N / 8 * nbits;
//     constexpr bool isSigned = true;
//     constexpr int zero_point = isSigned ? ((1 << (nbits - 1)) - 1) : 0;

//     // create four int8_t values
//     int8_t in_data[N] = {
//         0,
//     };
//     // breed seed
//     srand(0);

//     // random initializations with nbits range
//     for (int i = 0; i < N; i++)
//     {
//         in_data[i] = (rand() % (1 << nbits)) - zero_point;
//     }

//     // print input data
//     printf("in_data \n");
//     for (int i = 0; i < N; i++)
//     {
//         printf("i:%d %d %x \n", i, in_data[i], in_data[i]);
//     }

//     int8_t *ins = new int8_t[QN];
//     for (int i = 0; i < QN; i++)
//     {
//         ins[i] = (in_data[i * 4] & 0x3) | ((in_data[i * 4 + 1] & 0x3) << 2) | ((in_data[i * 4 + 2] & 0x3) << 4) | ((in_data[i * 4 + 3] & 0x3) << 6);
//     }
//     // print input data
//     printf("ins \n");
//     for (int i = 0; i < QN; i++)
//     {
//         printf("i:%d %d %x b: ", i, ins[i], ins[i]);
//         for (int j = 7; j >= 0; j--)
//         {
//             printf("%d", (ins[i] >> j) & 1);
//         }
//         printf("\n");
//     }
//     printf("\n");
//     int8_t *interleaved = new int8_t[QN];
//     general_interleave_int4(ins, interleaved, 2, QN * sizeof(int8_t), true);
//     printf("interleaved \n");
//     for (int i = 0; i < QN; i++)
//     {
//         printf("i:%d %d %x b: ", i, interleaved[i], interleaved[i]);
//         for (int j = 7; j >= 0; j--)
//         {
//             printf("%d", (interleaved[i] >> j) & 1);
//         }
//         printf("\n");    
//     }
//     target_dtype *decoded = new target_dtype[N];
//     int8_t *ins_gpu;
//     target_dtype *decoded_gpu;

//     cudaCheckLastError(cudaMalloc((void **)&ins_gpu, QN * sizeof(int8_t)));
//     cudaCheckLastError(cudaMalloc((void **)&decoded_gpu, N * sizeof(target_dtype)));
//     cudaCheckLastError(cudaMemcpy(ins_gpu, interleaved, QN * sizeof(int8_t), cudaMemcpyHostToDevice));
//     cudaCheckLastError(cudaMemcpy(decoded_gpu, decoded, N * sizeof(target_dtype), cudaMemcpyHostToDevice));
//     cudaCheckLastError(cudaDeviceSynchronize());

//     kernelWrapper_i2s_to_i4s<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu, decoded_gpu);
//     cudaCheckLastError(cudaDeviceSynchronize());
//     cudaCheckLastError(cudaMemcpy(decoded, decoded_gpu, N * sizeof(target_dtype), cudaMemcpyDeviceToHost));
//     cudaCheckLastError(cudaFree(ins_gpu));
//     cudaCheckLastError(cudaFree(decoded_gpu));
//     printf("decoded \n");
//     for (int i = 0; i < (N / 2); i++)
//     {
//         printf("i %d %d %x \n", i, decoded[i], decoded[i]);
//     }
//     // output data int8
//     int8_t i8_out[N] = {
//         0,
//     };
//     for (int i = 0; i < N; i++)
//     {
//         i8_out[i] = (decoded[i / 2] >> (4 * (i % 2)) ) & 0xf;
//     }
//     printf("i8_out \n");
//     for (int i = 0; i < N; i++)
//     {
//         printf("i %d in_data: %d %x decode_data: %d %x \n", i, in_data[i], in_data[i], i8_out[i], i8_out[i]);
//     }
//     for (int i = 0; i < (N / 2); i++)
//     {
//         EXPECT_EQ(in_data[i], int(i8_out[i]));
//     }
//     free(ins);
//     free(interleaved);
//     free(decoded);
// }


// int32 -> 16 int2 -> 4 int8
// -> 16 int4 -> 8 int8
TEST(DecodeTest, DecodeUInt4ToINT8)
{
    using target_dtype = int8_t;
    constexpr int nbits = 2;
    constexpr int N = 32 / nbits;
    constexpr int QN = N / 8 * nbits;
    constexpr bool isSigned = false;
    constexpr int zero_point = isSigned ? ((1 << (nbits - 1)) - 1) : 0;

    // create four int8_t values
    int8_t in_data[N] = {
        0,
    };
    // breed seed
    srand(0);

    // random initializations with nbits range
    for (int i = 0; i < N; i++)
    {
        in_data[i] = (rand() % (1 << nbits)) - zero_point;
        // in_data[i] = (i % 2);
        // in_data[i] = 1;
    }

    // print input data
    for (int i = 0; i < N; i++)
    {
        printf("i:%d %d %x \n", i, in_data[i], in_data[i]);
    }

    int8_t *ins = new int8_t[QN];
    for (int i = 0; i < QN; i++)
    {
        ins[i] = (in_data[i * 4] & 0x3) | ((in_data[i * 4 + 1] & 0x3) << 2) | ((in_data[i * 4 + 2] & 0x3) << 4) | ((in_data[i * 4 + 3] & 0x3) << 6);
    }
    // print input data
    printf("ins \n");
    for (int i = 0; i < QN; i++)
    {
        printf("i:%d %d %x b: ", i, ins[i], ins[i]);
        for (int j = 7; j >= 0; j--)
        {
            printf("%d", (ins[i] >> j) & 1);
        }
        printf("\n");
    }
    printf("\n");
    int8_t *interleaved = new int8_t[QN];
    general_interleave_int4(ins, interleaved, 2, QN * sizeof(int8_t), true);
    printf("interleaved \n");
    for (int i = 0; i < QN; i++)
    {
        printf("i:%d %d %x b: ", i, interleaved[i], interleaved[i]);
        for (int j = 7; j >= 0; j--)
        {
            printf("%d", (interleaved[i] >> j) & 1);
        }
        printf("\n");    
    }
    target_dtype *decoded = new target_dtype[N];
    int8_t *ins_gpu;
    target_dtype *decoded_gpu;

    cudaCheckLastError(cudaMalloc((void **)&ins_gpu, QN * sizeof(int8_t)));
    cudaCheckLastError(cudaMalloc((void **)&decoded_gpu, N * sizeof(target_dtype)));
    cudaCheckLastError(cudaMemcpy(ins_gpu, interleaved, QN * sizeof(int8_t), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(decoded_gpu, decoded, N * sizeof(target_dtype), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaDeviceSynchronize());

    kernelWrapper_i2u_to_i4s<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu, decoded_gpu);
    cudaCheckLastError(cudaDeviceSynchronize());
    cudaCheckLastError(cudaMemcpy(decoded, decoded_gpu, N * sizeof(target_dtype), cudaMemcpyDeviceToHost));
    cudaCheckLastError(cudaFree(ins_gpu));
    cudaCheckLastError(cudaFree(decoded_gpu));
    printf("decoded \n");
    for (int i = 0; i < (N / 2); i++)
    {
        printf("i %d %d %x \n", i, decoded[i], decoded[i]);
    }
    // output data int8
    int8_t i8_out[N] = {
        0,
    };
    for (int i = 0; i < N; i++)
    {
        i8_out[i] = (decoded[i / 2] >> (4 * (i % 2)) ) & 0xf;
    }
    printf("i8_out \n");
    for (int i = 0; i < N; i++)
    {
        printf("i %d in_data: %d %x decode_data: %d %x \n", i, in_data[i], in_data[i], i8_out[i], i8_out[i]);
    }
    for (int i = 0; i < (N / 2); i++)
    {
        EXPECT_EQ(in_data[i], int(i8_out[i]));
    }
    free(ins);
    free(interleaved);
    free(decoded);
}
