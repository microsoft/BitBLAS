// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include <gtest/gtest.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "fast_decoding.hpp"

#define cudaCheckLastError(ans)               \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
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

REGISTER_GLOBAL_DEVICE_INVOKER(kernelWrapper_i4s_to_f16, decode_i4s_to_f16)
REGISTER_GLOBAL_DEVICE_INVOKER(kernelWrapper_i4u_to_f16, decode_i4u_to_f16)
REGISTER_GLOBAL_DEVICE_INVOKER(kernelWrapper_i2s_to_f16, decode_i2s_to_f16)
REGISTER_GLOBAL_DEVICE_INVOKER(kernelWrapper_i2u_to_f16, decode_i2u_to_f16)
REGISTER_GLOBAL_DEVICE_INVOKER(kernelWrapper_i1s_to_f16, decode_i1s_to_f16)
REGISTER_GLOBAL_DEVICE_INVOKER(kernelWrapper_i1u_to_f16, decode_i1u_to_f16)
REGISTER_GLOBAL_DEVICE_INVOKER(kernelWrapper_i4s_to_f16_scale, decode_i4s_to_f16_scale)
REGISTER_GLOBAL_DEVICE_INVOKER(kernelWrapper_i4u_to_f16_scale, decode_i4u_to_f16_scale)
REGISTER_GLOBAL_DEVICE_INVOKER(kernelWrapper_i2s_to_f16_scale, decode_i2s_to_f16_scale)
REGISTER_GLOBAL_DEVICE_INVOKER(kernelWrapper_i2u_to_f16_scale, decode_i2u_to_f16_scale)
REGISTER_GLOBAL_DEVICE_INVOKER(kernelWrapper_i1s_to_f16_scale, decode_i1s_to_f16_scale)
REGISTER_GLOBAL_DEVICE_INVOKER(kernelWrapper_i1u_to_f16_scale, decode_i1u_to_f16_scale)

TEST(DecodeTest, DecodeInt4ToFloat16)
{
    constexpr int nbits = 4;
    constexpr int N = 32 / nbits;
    constexpr int QN = N / 8 * nbits;
    constexpr bool isSigned = true;
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
    }

    int8_t *ins = new int8_t[QN];
    general_compress(in_data, ins, nbits, N, isSigned);

    int8_t *interleaved = new int8_t[QN];
    general_interleave_fp16(ins, interleaved, nbits, QN * sizeof(int8_t), false);
    half *decoded = new half[N];
    int8_t *ins_gpu;
    half *decoded_gpu;

    cudaCheckLastError(cudaMalloc((void **)&ins_gpu, QN * sizeof(int8_t)));
    cudaCheckLastError(cudaMalloc((void **)&decoded_gpu, N * sizeof(half)));
    cudaCheckLastError(cudaMemcpy(ins_gpu, interleaved, QN * sizeof(int8_t), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(decoded_gpu, decoded, N * sizeof(half), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaDeviceSynchronize());

    kernelWrapper_i4s_to_f16<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu, decoded_gpu);
    cudaCheckLastError(cudaDeviceSynchronize());
    cudaCheckLastError(cudaMemcpy(decoded, decoded_gpu, N * sizeof(half), cudaMemcpyDeviceToHost));
    cudaCheckLastError(cudaFree(ins_gpu));
    cudaCheckLastError(cudaFree(decoded_gpu));
    for (int i = 0; i < N; i++)
    {
        EXPECT_EQ(in_data[i], int(decoded[i]));
    }
    free(ins);
    free(interleaved);
    free(decoded);
}

TEST(DecodeTest, DecodeUInt4ToFloat16)
{
    constexpr int nbits = 4;
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
    }

    int8_t *ins = new int8_t[QN];
    general_compress(in_data, ins, nbits, N, isSigned);
    int8_t *interleaved = new int8_t[QN];
    general_interleave_fp16(ins, interleaved, nbits, QN * sizeof(int8_t), false);

    half *decoded = new half[N];
    int8_t *ins_gpu;
    half *decoded_gpu;

    cudaCheckLastError(cudaMalloc((void **)&ins_gpu, QN * sizeof(int8_t)));
    cudaCheckLastError(cudaMalloc((void **)&decoded_gpu, N * sizeof(half)));
    cudaCheckLastError(cudaMemcpy(ins_gpu, interleaved, QN * sizeof(int8_t), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(decoded_gpu, decoded, N * sizeof(half), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaDeviceSynchronize());

    kernelWrapper_i4u_to_f16<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu, decoded_gpu);

    cudaCheckLastError(cudaDeviceSynchronize());
    cudaCheckLastError(cudaMemcpy(decoded, decoded_gpu, N * sizeof(half), cudaMemcpyDeviceToHost));
    cudaCheckLastError(cudaFree(ins_gpu));
    cudaCheckLastError(cudaFree(decoded_gpu));
    for (int i = 0; i < N; i++)
    {
        EXPECT_EQ(in_data[i], int(decoded[i]));
    }

    free(ins);
    free(interleaved);
    free(decoded);
}

TEST(DecodeTest, DecodeInt2ToFloat16)
{
    constexpr int nbits = 2;
    constexpr int N = 32 / nbits;
    constexpr int QN = N / 8 * nbits;
    constexpr bool isSigned = true;
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
    }

    int8_t *ins = new int8_t[QN];
    general_compress(in_data, ins, nbits, N, isSigned);

    int8_t *interleaved = new int8_t[QN];
    general_interleave_fp16(ins, interleaved, nbits, QN * sizeof(int8_t), false);
    half *decoded = new half[N];
    int8_t *ins_gpu;
    half *decoded_gpu;

    cudaCheckLastError(cudaMalloc((void **)&ins_gpu, QN * sizeof(int8_t)));
    cudaCheckLastError(cudaMalloc((void **)&decoded_gpu, N * sizeof(half)));
    cudaCheckLastError(cudaMemcpy(ins_gpu, interleaved, QN * sizeof(int8_t), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(decoded_gpu, decoded, N * sizeof(half), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaDeviceSynchronize());

    kernelWrapper_i2s_to_f16<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu, decoded_gpu);
    kernelWrapper_i2s_to_f16<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu + QN / 2, decoded_gpu + N / 2);
    cudaCheckLastError(cudaDeviceSynchronize());
    cudaCheckLastError(cudaMemcpy(decoded, decoded_gpu, N * sizeof(half), cudaMemcpyDeviceToHost));
    cudaCheckLastError(cudaFree(ins_gpu));
    cudaCheckLastError(cudaFree(decoded_gpu));
    for (int i = 0; i < N; i++)
    {
        EXPECT_EQ(in_data[i], int(decoded[i]));
    }
    free(ins);
    free(interleaved);
    free(decoded);
}

TEST(DecodeTest, DecodeUInt2ToFloat16)
{
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
    }

    int8_t *ins = new int8_t[QN];
    general_compress(in_data, ins, nbits, N, isSigned);
    int8_t *interleaved = new int8_t[QN];
    general_interleave_fp16(ins, interleaved, nbits, QN * sizeof(int8_t), false);
    half *decoded = new half[N];
    int8_t *ins_gpu;
    half *decoded_gpu;

    cudaCheckLastError(cudaMalloc((void **)&ins_gpu, QN * sizeof(int8_t)));
    cudaCheckLastError(cudaMalloc((void **)&decoded_gpu, N * sizeof(half)));
    cudaCheckLastError(cudaMemcpy(ins_gpu, interleaved, QN * sizeof(int8_t), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(decoded_gpu, decoded, N * sizeof(half), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaDeviceSynchronize());

    kernelWrapper_i2u_to_f16<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu, decoded_gpu);
    kernelWrapper_i2u_to_f16<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu + QN / 2, decoded_gpu + N / 2);

    cudaCheckLastError(cudaDeviceSynchronize());
    cudaCheckLastError(cudaMemcpy(decoded, decoded_gpu, N * sizeof(half), cudaMemcpyDeviceToHost));
    cudaCheckLastError(cudaFree(ins_gpu));
    cudaCheckLastError(cudaFree(decoded_gpu));
    for (int i = 0; i < N; i++)
    {
        EXPECT_EQ(in_data[i], int(decoded[i]));
    }

    free(ins);
    free(interleaved);
    free(decoded);
}

TEST(DecodeTest, DecodeInt1ToFloat16)
{
    constexpr int nbits = 1;
    constexpr int N = 32 / nbits;
    constexpr int QN = N / 8 * nbits;
    constexpr bool isSigned = true;
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
    }

    int8_t *ins = new int8_t[QN];
    general_compress(in_data, ins, nbits, N, isSigned);

    int8_t *interleaved = new int8_t[QN];
    general_interleave_fp16(ins, interleaved, nbits, QN * sizeof(int8_t), false);
    half *decoded = new half[N];
    int8_t *ins_gpu;
    half *decoded_gpu;

    cudaCheckLastError(cudaMalloc((void **)&ins_gpu, QN * sizeof(int8_t)));
    cudaCheckLastError(cudaMalloc((void **)&decoded_gpu, N * sizeof(half)));
    cudaCheckLastError(cudaMemcpy(ins_gpu, interleaved, QN * sizeof(int8_t), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(decoded_gpu, decoded, N * sizeof(half), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaDeviceSynchronize());

    kernelWrapper_i1s_to_f16<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu, decoded_gpu);
    kernelWrapper_i1s_to_f16<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu + QN / 4, decoded_gpu + N / 4);
    kernelWrapper_i1s_to_f16<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu + QN / 2, decoded_gpu + N / 2);
    kernelWrapper_i1s_to_f16<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu + +QN / 2 + QN / 4, decoded_gpu + +N / 2 + N / 4);
    cudaCheckLastError(cudaDeviceSynchronize());
    cudaCheckLastError(cudaMemcpy(decoded, decoded_gpu, N * sizeof(half), cudaMemcpyDeviceToHost));
    cudaCheckLastError(cudaFree(ins_gpu));
    cudaCheckLastError(cudaFree(decoded_gpu));
    for (int i = 0; i < N; i++)
    {
        EXPECT_EQ(in_data[i], int(decoded[i]));
    }
    free(ins);
    free(interleaved);
    free(decoded);
}

TEST(DecodeTest, DecodeUInt1ToFloat16)
{
    constexpr int nbits = 1;
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
    }

    int8_t *ins = new int8_t[QN];
    general_compress(in_data, ins, nbits, N, isSigned);

    int8_t *interleaved = new int8_t[QN];
    general_interleave_fp16(ins, interleaved, nbits, QN * sizeof(int8_t), false);
    half *decoded = new half[N];
    int8_t *ins_gpu;
    half *decoded_gpu;

    cudaCheckLastError(cudaMalloc((void **)&ins_gpu, QN * sizeof(int8_t)));
    cudaCheckLastError(cudaMalloc((void **)&decoded_gpu, N * sizeof(half)));
    cudaCheckLastError(cudaMemcpy(ins_gpu, interleaved, QN * sizeof(int8_t), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(decoded_gpu, decoded, N * sizeof(half), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaDeviceSynchronize());

    kernelWrapper_i1u_to_f16<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu, decoded_gpu);
    kernelWrapper_i1u_to_f16<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu + QN / 4, decoded_gpu + N / 4);
    kernelWrapper_i1u_to_f16<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu + QN / 2, decoded_gpu + N / 2);
    kernelWrapper_i1u_to_f16<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu + +QN / 2 + QN / 4, decoded_gpu + +N / 2 + N / 4);
    cudaCheckLastError(cudaDeviceSynchronize());
    cudaCheckLastError(cudaMemcpy(decoded, decoded_gpu, N * sizeof(half), cudaMemcpyDeviceToHost));
    cudaCheckLastError(cudaFree(ins_gpu));
    cudaCheckLastError(cudaFree(decoded_gpu));
    for (int i = 0; i < N; i++)
    {
        EXPECT_EQ(in_data[i], int(decoded[i]));
    }
    free(ins);
    free(interleaved);
    free(decoded);
}

TEST(DecodeTest, DecodeInt4ToFloat16WithScaling)
{
    constexpr int nbits = 4;
    constexpr int N = 32 / nbits;
    constexpr int QN = N / 8 * nbits;
    constexpr bool isSigned = true;
    constexpr int zero_point = isSigned ? ((1 << (nbits - 1)) - 1) : 0;

    // create four int8_t values
    int8_t in_data[N] = {
        0,
    };
    half scale[1] = {__float2half(0.314)};
    // breed seed
    srand(0);

    // random initializations with nbits range
    for (int i = 0; i < N; i++)
    {
        in_data[i] = (rand() % (1 << nbits)) - zero_point;
    }

    int8_t *ins = new int8_t[QN];
    general_compress(in_data, ins, nbits, N, isSigned);

    int8_t *interleaved = new int8_t[QN];
    general_interleave_fp16(ins, interleaved, nbits, QN * sizeof(int8_t), false);
    half *decoded = new half[N];
    int8_t *ins_gpu;
    half *decoded_gpu, *scale_gpu;

    cudaCheckLastError(cudaMalloc((void **)&ins_gpu, QN * sizeof(int8_t)));
    cudaCheckLastError(cudaMalloc((void **)&decoded_gpu, N * sizeof(half)));
    cudaCheckLastError(cudaMalloc((void **)&scale_gpu, 1 * sizeof(half)));
    cudaCheckLastError(cudaMemcpy(ins_gpu, interleaved, QN * sizeof(int8_t), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(decoded_gpu, decoded, N * sizeof(half), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(scale_gpu, scale, 1 * sizeof(half), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaDeviceSynchronize());

    kernelWrapper_i4s_to_f16_scale<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu, decoded_gpu, scale_gpu);
    cudaCheckLastError(cudaDeviceSynchronize());
    cudaCheckLastError(cudaMemcpy(decoded, decoded_gpu, N * sizeof(half), cudaMemcpyDeviceToHost));
    cudaCheckLastError(cudaFree(ins_gpu));
    cudaCheckLastError(cudaFree(decoded_gpu));
    for (int i = 0; i < N; i++)
    {
        EXPECT_NEAR(in_data[i] * float(scale[0]), float(decoded[i]), 1e-2);
    }
    free(ins);
    free(interleaved);
    free(decoded);
}

TEST(DecodeTest, DecodeUInt4ToFloat16WithScaling)
{
    constexpr int nbits = 4;
    constexpr int N = 32 / nbits;
    constexpr int QN = N / 8 * nbits;
    constexpr bool isSigned = false;
    constexpr int zero_point = isSigned ? ((1 << (nbits - 1)) - 1) : 0;

    // create four int8_t values
    int8_t in_data[N] = {
        0,
    };
    half scale[1] = {__float2half(1.2)};
    // breed seed
    srand(0);

    // random initializations with nbits range
    for (int i = 0; i < N; i++)
    {
        in_data[i] = (rand() % (1 << nbits)) - zero_point;
    }

    int8_t *ins = new int8_t[QN];
    general_compress(in_data, ins, nbits, N, isSigned);

    int8_t *interleaved = new int8_t[QN];
    general_interleave_fp16(ins, interleaved, nbits, QN * sizeof(int8_t), false);
    half *decoded = new half[N];
    int8_t *ins_gpu;
    half *decoded_gpu, *scale_gpu;

    cudaCheckLastError(cudaMalloc((void **)&ins_gpu, QN * sizeof(int8_t)));
    cudaCheckLastError(cudaMalloc((void **)&decoded_gpu, N * sizeof(half)));
    cudaCheckLastError(cudaMalloc((void **)&scale_gpu, 1 * sizeof(half)));
    cudaCheckLastError(cudaMemcpy(ins_gpu, interleaved, QN * sizeof(int8_t), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(decoded_gpu, decoded, N * sizeof(half), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(scale_gpu, scale, 1 * sizeof(half), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaDeviceSynchronize());

    kernelWrapper_i4u_to_f16_scale<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu, decoded_gpu, scale_gpu);
    cudaCheckLastError(cudaDeviceSynchronize());
    cudaCheckLastError(cudaMemcpy(decoded, decoded_gpu, N * sizeof(half), cudaMemcpyDeviceToHost));
    cudaCheckLastError(cudaFree(ins_gpu));
    cudaCheckLastError(cudaFree(decoded_gpu));
    for (int i = 0; i < N; i++)
    {
        EXPECT_NEAR(in_data[i] * float(scale[0]), float(decoded[i]), 1e-2);
    }
    free(ins);
    free(interleaved);
    free(decoded);
}

TEST(DecodeTest, DecodeInt2ToFloat16WithScaling)
{
    constexpr int nbits = 2;
    constexpr int N = 32 / nbits;
    constexpr int QN = N / 8 * nbits;
    constexpr bool isSigned = true;
    constexpr int zero_point = isSigned ? ((1 << (nbits - 1)) - 1) : 0;

    // create four int8_t values
    int8_t in_data[N] = {
        0,
    };
    half scale[1] = {__float2half(0.314)};
    // breed seed
    srand(0);

    // random initializations with nbits range
    for (int i = 0; i < N; i++)
    {
        in_data[i] = (rand() % (1 << nbits)) - zero_point;
    }

    int8_t *ins = new int8_t[QN];
    general_compress(in_data, ins, nbits, N, isSigned);

    int8_t *interleaved = new int8_t[QN];
    general_interleave_fp16(ins, interleaved, nbits, QN * sizeof(int8_t), false);
    half *decoded = new half[N];
    int8_t *ins_gpu;
    half *decoded_gpu, *scale_gpu;

    cudaCheckLastError(cudaMalloc((void **)&ins_gpu, QN * sizeof(int8_t)));
    cudaCheckLastError(cudaMalloc((void **)&decoded_gpu, N * sizeof(half)));
    cudaCheckLastError(cudaMalloc((void **)&scale_gpu, 1 * sizeof(half)));
    cudaCheckLastError(cudaMemcpy(ins_gpu, interleaved, QN * sizeof(int8_t), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(decoded_gpu, decoded, N * sizeof(half), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(scale_gpu, scale, 1 * sizeof(half), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaDeviceSynchronize());

    kernelWrapper_i2s_to_f16_scale<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu, decoded_gpu, scale_gpu);
    kernelWrapper_i2s_to_f16_scale<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu + QN / 2, decoded_gpu + N / 2, scale_gpu);
    cudaCheckLastError(cudaDeviceSynchronize());
    cudaCheckLastError(cudaMemcpy(decoded, decoded_gpu, N * sizeof(half), cudaMemcpyDeviceToHost));
    cudaCheckLastError(cudaFree(ins_gpu));
    cudaCheckLastError(cudaFree(decoded_gpu));
    for (int i = 0; i < N; i++)
    {
        EXPECT_NEAR(in_data[i] * float(scale[0]), float(decoded[i]), 1e-2);
    }
    free(ins);
    free(interleaved);
    free(decoded);
}

TEST(DecodeTest, DecodeUInt2ToFloat16WithScaling)
{
    constexpr int nbits = 2;
    constexpr int N = 32 / nbits;
    constexpr int QN = N / 8 * nbits;
    constexpr bool isSigned = false;
    constexpr int zero_point = isSigned ? ((1 << (nbits - 1)) - 1) : 0;

    // create four int8_t values
    int8_t in_data[N] = {
        0,
    };
    half scale[1] = {__float2half(1.0)};
    // breed seed
    srand(0);

    // random initializations with nbits range
    for (int i = 0; i < N; i++)
    {
        in_data[i] = (rand() % (1 << nbits)) - zero_point;
    }

    int8_t *ins = new int8_t[QN];
    general_compress(in_data, ins, nbits, N, isSigned);

    int8_t *interleaved = new int8_t[QN];
    general_interleave_fp16(ins, interleaved, nbits, QN * sizeof(int8_t), false);
    half *decoded = new half[N];
    int8_t *ins_gpu;
    half *decoded_gpu, *scale_gpu;

    cudaCheckLastError(cudaMalloc((void **)&ins_gpu, QN * sizeof(int8_t)));
    cudaCheckLastError(cudaMalloc((void **)&decoded_gpu, N * sizeof(half)));
    cudaCheckLastError(cudaMalloc((void **)&scale_gpu, 1 * sizeof(half)));
    cudaCheckLastError(cudaMemcpy(ins_gpu, interleaved, QN * sizeof(int8_t), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(decoded_gpu, decoded, N * sizeof(half), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(scale_gpu, scale, 1 * sizeof(half), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaDeviceSynchronize());

    kernelWrapper_i2u_to_f16_scale<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu, decoded_gpu, scale_gpu);
    kernelWrapper_i2u_to_f16_scale<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu + QN / 2, decoded_gpu + N / 2, scale_gpu);
    cudaCheckLastError(cudaDeviceSynchronize());
    cudaCheckLastError(cudaMemcpy(decoded, decoded_gpu, N * sizeof(half), cudaMemcpyDeviceToHost));
    cudaCheckLastError(cudaFree(ins_gpu));
    cudaCheckLastError(cudaFree(decoded_gpu));
    for (int i = 0; i < N; i++)
    {
        EXPECT_NEAR(in_data[i] * float(scale[0]), float(decoded[i]), 1e-2);
    }
    free(ins);
    free(interleaved);
    free(decoded);
}

TEST(DecodeTest, DecodeInt1ToFloat16WithScaling)
{
    constexpr int nbits = 1;
    constexpr int N = 32 / nbits;
    constexpr int QN = N / 8 * nbits;
    constexpr bool isSigned = true;
    constexpr int zero_point = isSigned ? ((1 << (nbits - 1)) - 1) : 0;

    // create four int8_t values
    int8_t in_data[N] = {
        0,
    };
    half scale[1] = {__float2half(0.314)};
    // breed seed
    srand(0);

    // random initializations with nbits range
    for (int i = 0; i < N; i++)
    {
        in_data[i] = (rand() % (1 << nbits)) - zero_point;
    }

    int8_t *ins = new int8_t[QN];
    general_compress(in_data, ins, nbits, N, isSigned);

    int8_t *interleaved = new int8_t[QN];
    general_interleave_fp16(ins, interleaved, nbits, QN * sizeof(int8_t), false);
    half *decoded = new half[N];
    int8_t *ins_gpu;
    half *decoded_gpu, *scale_gpu;

    cudaCheckLastError(cudaMalloc((void **)&ins_gpu, QN * sizeof(int8_t)));
    cudaCheckLastError(cudaMalloc((void **)&decoded_gpu, N * sizeof(half)));
    cudaCheckLastError(cudaMalloc((void **)&scale_gpu, 1 * sizeof(half)));
    cudaCheckLastError(cudaMemcpy(ins_gpu, interleaved, QN * sizeof(int8_t), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(decoded_gpu, decoded, N * sizeof(half), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(scale_gpu, scale, 1 * sizeof(half), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaDeviceSynchronize());

    kernelWrapper_i1s_to_f16_scale<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu, decoded_gpu, scale_gpu);
    kernelWrapper_i1s_to_f16_scale<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu + QN / 4, decoded_gpu + N / 4, scale_gpu);
    kernelWrapper_i1s_to_f16_scale<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu + QN / 2, decoded_gpu + N / 2, scale_gpu);
    kernelWrapper_i1s_to_f16_scale<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu + QN / 2 + QN / 4, decoded_gpu + N / 2 + N / 4, scale_gpu);
    cudaCheckLastError(cudaDeviceSynchronize());
    cudaCheckLastError(cudaMemcpy(decoded, decoded_gpu, N * sizeof(half), cudaMemcpyDeviceToHost));
    cudaCheckLastError(cudaFree(ins_gpu));
    cudaCheckLastError(cudaFree(decoded_gpu));
    for (int i = 0; i < N; i++)
    {
        EXPECT_NEAR(in_data[i] * float(scale[0]), float(decoded[i]), 1e-2);
    }
    free(ins);
    free(interleaved);
    free(decoded);
}

TEST(DecodeTest, DecodeUInt1ToFloat16WithScaling)
{
    constexpr int nbits = 1;
    constexpr int N = 32 / nbits;
    constexpr int QN = N / 8 * nbits;
    constexpr bool isSigned = false;
    constexpr int zero_point = isSigned ? ((1 << (nbits - 1)) - 1) : 0;

    // create four int8_t values
    int8_t in_data[N] = {
        0,
    };
    half scale[1] = {__float2half(1.0)};
    // breed seed
    srand(0);

    // random initializations with nbits range
    for (int i = 0; i < N; i++)
    {
        in_data[i] = (rand() % (1 << nbits)) - zero_point;
    }

    int8_t *ins = new int8_t[QN];
    general_compress(in_data, ins, nbits, N, isSigned);

    int8_t *interleaved = new int8_t[QN];
    general_interleave_fp16(ins, interleaved, nbits, QN * sizeof(int8_t), false);
    half *decoded = new half[N];
    int8_t *ins_gpu;
    half *decoded_gpu, *scale_gpu;

    cudaCheckLastError(cudaMalloc((void **)&ins_gpu, QN * sizeof(int8_t)));
    cudaCheckLastError(cudaMalloc((void **)&decoded_gpu, N * sizeof(half)));
    cudaCheckLastError(cudaMalloc((void **)&scale_gpu, 1 * sizeof(half)));
    cudaCheckLastError(cudaMemcpy(ins_gpu, interleaved, QN * sizeof(int8_t), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(decoded_gpu, decoded, N * sizeof(half), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(scale_gpu, scale, 1 * sizeof(half), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaDeviceSynchronize());

    kernelWrapper_i1u_to_f16_scale<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu, decoded_gpu, scale_gpu);
    kernelWrapper_i1u_to_f16_scale<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu + QN / 4, decoded_gpu + N / 4, scale_gpu);
    kernelWrapper_i1u_to_f16_scale<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu + QN / 2, decoded_gpu + N / 2, scale_gpu);
    kernelWrapper_i1u_to_f16_scale<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu + QN / 2 + QN / 4, decoded_gpu + N / 2 + N / 4, scale_gpu);
    cudaCheckLastError(cudaDeviceSynchronize());
    cudaCheckLastError(cudaMemcpy(decoded, decoded_gpu, N * sizeof(half), cudaMemcpyDeviceToHost));
    cudaCheckLastError(cudaFree(ins_gpu));
    cudaCheckLastError(cudaFree(decoded_gpu));
    for (int i = 0; i < N; i++)
    {
        EXPECT_NEAR(in_data[i] * float(scale[0]), float(decoded[i]), 1e-2);
    }
    free(ins);
    free(interleaved);
    free(decoded);
}
