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

REGISTER_GLOBAL_DEVICE_INVOKER(kernelWrapper_i4s_to_i8s, decode_i4s_to_i8s)
REGISTER_GLOBAL_DEVICE_INVOKER(kernelWrapper_i4u_to_i8s, decode_i4u_to_i8s)
REGISTER_GLOBAL_DEVICE_INVOKER(kernelWrapper_i2s_to_i8s, decode_i2s_to_i8s)
REGISTER_GLOBAL_DEVICE_INVOKER(kernelWrapper_i2u_to_i8s, decode_i2u_to_i8s)
REGISTER_GLOBAL_DEVICE_INVOKER(kernelWrapper_i1s_to_i8s, decode_i1s_to_i8s)
REGISTER_GLOBAL_DEVICE_INVOKER(kernelWrapper_i1u_to_i8s, decode_i1u_to_i8s)

TEST(DecodeTest, DecodeInt4ToINT8)
{
    using target_dtype = int8_t;
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
    general_interleave_int8(ins, interleaved, nbits, QN * sizeof(int8_t), false);
    target_dtype *decoded = new target_dtype[N];
    int8_t *ins_gpu;
    target_dtype *decoded_gpu;

    cudaCheckLastError(cudaMalloc((void **)&ins_gpu, QN * sizeof(int8_t)));
    cudaCheckLastError(cudaMalloc((void **)&decoded_gpu, N * sizeof(target_dtype)));
    cudaCheckLastError(cudaMemcpy(ins_gpu, interleaved, QN * sizeof(int8_t), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(decoded_gpu, decoded, N * sizeof(target_dtype), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaDeviceSynchronize());

    kernelWrapper_i4s_to_i8s<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu, decoded_gpu);
    cudaCheckLastError(cudaDeviceSynchronize());
    cudaCheckLastError(cudaMemcpy(decoded, decoded_gpu, N * sizeof(target_dtype), cudaMemcpyDeviceToHost));
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

TEST(DecodeTest, DecodeUInt4ToINT8)
{
    using target_dtype = int8_t;
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
    general_interleave_int8(ins, interleaved, nbits, QN * sizeof(int8_t), false);
    target_dtype *decoded = new target_dtype[N];
    int8_t *ins_gpu;
    target_dtype *decoded_gpu;

    cudaCheckLastError(cudaMalloc((void **)&ins_gpu, QN * sizeof(int8_t)));
    cudaCheckLastError(cudaMalloc((void **)&decoded_gpu, N * sizeof(target_dtype)));
    cudaCheckLastError(cudaMemcpy(ins_gpu, interleaved, QN * sizeof(int8_t), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(decoded_gpu, decoded, N * sizeof(target_dtype), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaDeviceSynchronize());

    kernelWrapper_i4u_to_i8s<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu, decoded_gpu);
    cudaCheckLastError(cudaDeviceSynchronize());
    cudaCheckLastError(cudaMemcpy(decoded, decoded_gpu, N * sizeof(target_dtype), cudaMemcpyDeviceToHost));
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

TEST(DecodeTest, DecodeInt2ToINT8)
{
    using target_dtype = int8_t;
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
    general_interleave_int8(ins, interleaved, nbits, QN * sizeof(int8_t), false);
    target_dtype *decoded = new target_dtype[N];
    int8_t *ins_gpu;
    target_dtype *decoded_gpu;

    cudaCheckLastError(cudaMalloc((void **)&ins_gpu, QN * sizeof(int8_t)));
    cudaCheckLastError(cudaMalloc((void **)&decoded_gpu, N * sizeof(target_dtype)));
    cudaCheckLastError(cudaMemcpy(ins_gpu, interleaved, QN * sizeof(int8_t), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(decoded_gpu, decoded, N * sizeof(target_dtype), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaDeviceSynchronize());

    kernelWrapper_i2s_to_i8s<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu, decoded_gpu);
    cudaCheckLastError(cudaDeviceSynchronize());
    cudaCheckLastError(cudaMemcpy(decoded, decoded_gpu, N * sizeof(target_dtype), cudaMemcpyDeviceToHost));
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

TEST(DecodeTest, DecodeUInt2ToINT8)
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
    }

    int8_t *ins = new int8_t[QN];
    general_compress(in_data, ins, nbits, N, isSigned);

    int8_t *interleaved = new int8_t[QN];
    general_interleave_int8(ins, interleaved, nbits, QN * sizeof(int8_t), false);

    target_dtype *decoded = new target_dtype[N];
    int8_t *ins_gpu;
    target_dtype *decoded_gpu;

    cudaCheckLastError(cudaMalloc((void **)&ins_gpu, QN * sizeof(int8_t)));
    cudaCheckLastError(cudaMalloc((void **)&decoded_gpu, N * sizeof(target_dtype)));
    cudaCheckLastError(cudaMemcpy(ins_gpu, interleaved, QN * sizeof(int8_t), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(decoded_gpu, decoded, N * sizeof(target_dtype), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaDeviceSynchronize());

    kernelWrapper_i2u_to_i8s<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu, decoded_gpu);
    cudaCheckLastError(cudaDeviceSynchronize());
    cudaCheckLastError(cudaMemcpy(decoded, decoded_gpu, N * sizeof(target_dtype), cudaMemcpyDeviceToHost));
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

TEST(DecodeTest, DecodeInt1ToINT8)
{
    using target_dtype = int8_t;
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
    general_interleave_int8(ins, interleaved, nbits, QN * sizeof(int8_t), false);
    target_dtype *decoded = new target_dtype[N];
    int8_t *ins_gpu;
    target_dtype *decoded_gpu;

    cudaCheckLastError(cudaMalloc((void **)&ins_gpu, QN * sizeof(int8_t)));
    cudaCheckLastError(cudaMalloc((void **)&decoded_gpu, N * sizeof(target_dtype)));
    cudaCheckLastError(cudaMemcpy(ins_gpu, interleaved, QN * sizeof(int8_t), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(decoded_gpu, decoded, N * sizeof(target_dtype), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaDeviceSynchronize());

    kernelWrapper_i1s_to_i8s<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu, decoded_gpu);
    kernelWrapper_i1s_to_i8s<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu + QN / 2, decoded_gpu + N / 2);
    cudaCheckLastError(cudaDeviceSynchronize());
    cudaCheckLastError(cudaMemcpy(decoded, decoded_gpu, N * sizeof(target_dtype), cudaMemcpyDeviceToHost));
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

TEST(DecodeTest, DecodeUInt1ToINT8)
{
    using target_dtype = int8_t;
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
    general_interleave_int8(ins, interleaved, nbits, QN * sizeof(int8_t), false);

    target_dtype *decoded = new target_dtype[N];
    int8_t *ins_gpu;
    target_dtype *decoded_gpu;

    cudaCheckLastError(cudaMalloc((void **)&ins_gpu, QN * sizeof(int8_t)));
    cudaCheckLastError(cudaMalloc((void **)&decoded_gpu, N * sizeof(target_dtype)));
    cudaCheckLastError(cudaMemcpy(ins_gpu, interleaved, QN * sizeof(int8_t), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(decoded_gpu, decoded, N * sizeof(target_dtype), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaDeviceSynchronize());

    kernelWrapper_i1u_to_i8s<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu, decoded_gpu);
    kernelWrapper_i1u_to_i8s<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(ins_gpu + QN / 2, decoded_gpu + N / 2);
    cudaCheckLastError(cudaDeviceSynchronize());
    cudaCheckLastError(cudaMemcpy(decoded, decoded_gpu, N * sizeof(target_dtype), cudaMemcpyDeviceToHost));
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
