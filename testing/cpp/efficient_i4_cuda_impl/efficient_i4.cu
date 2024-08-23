// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include <gtest/gtest.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "i4matmul.hpp"

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

void general_compress(const int8_t *lowbit, int8_t *compressed, const int nbit, const int N, bool isSigned = false)
{
    int zero_point = isSigned ? ((1 << (nbit - 1)) - 1) : 0;
    const int nbit_per_byte = 8 / nbit;

    for (int i = 0; i < N / nbit_per_byte; i++)
    {
        compressed[i] = 0;
        for (int j = 0; j < nbit_per_byte; j++)
        {
            compressed[i] |= ((lowbit[nbit_per_byte * i + j] + zero_point) << (nbit * j));
        }
    }
}


// Helper function to interleave the perm array
std::vector<int> interleave_perms(const std::vector<int>& perm) {
    std::vector<int> interleaved_perm;
    std::array<int, 8> interleave = {0, 2, 4, 6, 1, 3, 5, 7};

    int num_rows = perm.size() / 8;
    for (int i = 0; i < num_rows; ++i) {
        std::array<int, 8> row;
        std::copy(perm.begin() + i * 8, perm.begin() + (i + 1) * 8, row.begin());
        for (int j : interleave) {
            interleaved_perm.push_back(row[j]);
        }
    }
    
    return interleaved_perm;
}


std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> get_perms() {
    std::vector<int> perm;

    for (int i = 0; i < 32; ++i) {
        std::vector<int> perm1;
        int col = i / 4;
        for (int block : {0, 1}) {
            for (int row : {
                    2 * (i % 4),
                    2 * (i % 4) + 1,
                    2 * (i % 4 + 4),
                    2 * (i % 4 + 4) + 1
            }) {
                perm1.push_back(16 * row + col + 8 * block);
            }
        }
        for (int j = 0; j < 4; ++j) {
            for (int p : perm1) {
                perm.push_back(p + 256 * j);
            }
        }
    }

    // Interleave the perm array
    perm = interleave_perms(perm);
    
    std::vector<int> scale_perm;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            scale_perm.push_back(i + 8 * j);
        }
    }

    std::vector<int> scale_perm_single;
    for (int i = 0; i < 4; ++i) {
        for (int j : {0, 1, 8, 9, 16, 17, 24, 25}) {
            scale_perm_single.push_back(2 * i + j);
        }
    }

    return std::make_tuple(perm, scale_perm, scale_perm_single);
}

void weight_pre_process(const int8_t *lowbit, int8_t *compressed, const int nbit, const int K, const int N)
{
    int8_t* tmp1 = new int8_t[K * N];
    const int maxq = 15;
    auto [perm, scale_perm, scale_perm_single] = get_perms();
    const int tile_size = 16;
    // transform the lowbit matrix to the compressed matrix
    for (int i = 0; i < (K / tile_size); i += 1)
    {
        for (int j = 0; j < (N / tile_size); j += 1)
        {
            for (int k = 0; k < tile_size; k++)
            {
                for (int l = 0; l < tile_size; l++)
                {
                    int idx_target = i * N * tile_size + j * tile_size * tile_size + k * tile_size + l;
                    int idx_source = (i * tile_size + k) * N + j * tile_size + l;
                    tmp1[idx_target] = lowbit[idx_source] +  (maxq + 1) / 2;
                }
            }
        }
    }
    // print the first 10 of tmp2
    printf("tmp1\n");
    for (int i = 0; i < 10; i++)
    {
        printf("%d ", tmp1[i]);
    }
    printf(" ... ");
    for (int i = K * N  - 10; i < K * N; i++)
    {
        printf("%d ", tmp1[i]);
    }
    printf("\n");
    // permute the matrix
    int32_t* tmp2 = new int32_t[K * N];
    const int perm_size = perm.size();
    for (int i = 0; i < (N * K / perm_size); i++)
    {
        for (int j = 0; j < perm_size; j++)
        {
            int idx_target = i * perm_size + j;
            int idx_source = i * perm_size + perm[j];
            tmp2[idx_target] = (int32_t)tmp1[idx_source];
        }
    }
    // print the first 10 of tmp2
    printf("tmp2\n");
    for (int i = 0; i < 10; i++)
    {
        printf("%d ", tmp2[i]);
    }
    printf(" ... ");
    for (int i = K * N / (32 / nbit) - 10; i < K * N / (32 / nbit); i++)
    {
        printf("%d ", tmp2[i]);
    }
    printf("\n");
    // compress
    int32_t* tmp3 = new int32_t[K * N / (32 / nbit)];
    // set zero
    for (int i = 0; i < K * N / (32 / nbit); i++)
    {
        tmp3[i] = 0;
    }
    for (int i = 0; i < (K / tile_size); i++)
    {
        for (int j = 0; j < (N * tile_size / 8); j++)
        {
            for (int k = 0; k < 8; k++)
            {
                int idx_target = i * N * tile_size / 8 + j;
                int idx_source = i * N * tile_size + j * 8 + k;
                tmp3[idx_target] |= (tmp2[idx_source] << (nbit * (k % 8)));
            }
        }
    }
    // print the first 10 of tmp3
    printf("tmp3\n");
    for (int i = 0; i < 10; i++)
    {
        printf("%d ", tmp3[i]);
    }
    printf(" ... ");
    for (int i = K * N / (32 / nbit) - 10; i < K * N / (32 / nbit); i++)
    {
        printf("%d ", tmp3[i]);
    }
    printf("\n");
    // copy tmp3 to compressed
    for (int i = 0; i < K * N / (32 / nbit); i++)
    {
        ((int32_t *)(compressed))[i] = tmp3[i];
    }
}

void scale_pre_process(const half *scale, half *scale_perm, const int K, const int N, int group_size)
{
    auto [perm, scale_perm_group, scale_perm_single] = get_perms();
    if (group_size == -1)
        group_size = K;
    if (group_size == K){
        const int perm_size = scale_perm_single.size();
        for (int i = 0; i < (N * K / group_size / perm_size); i++)
        {
            for (int j = 0; j < perm_size; j++)
            {
                int idx_target = i * perm_size + j;
                int idx_source = i * perm_size + scale_perm_single[j];
                if (idx_target < 10){
                    printf("idx_target = %d, idx_source = %d\n", idx_target, idx_source);
                }
                scale_perm[idx_target] = scale[idx_source];
            }
        }
    }
    else{
        const int perm_size = scale_perm_group.size();
        for (int i = 0; i < (N * K / group_size / perm_size); i++)
        {
            for (int j = 0; j < perm_size; j++)
            {
                int idx_target = i * perm_size + j;
                int idx_source = i * perm_size + scale_perm_group[j];
                scale_perm[idx_target] = scale[idx_source];
            }
        }
    }
    // print the first 10 of tmp2
    printf("scale_perm\n");
    for (int i = 0; i < 10; i++)
    {
        printf("%f ", (float)scale_perm[i]);
    }
    printf(" ... ");
    for (int i = K * N / group_size - 10; i < K * N / group_size; i++)
    {
        printf("%f ", (float)scale_perm[i]);
    }
}

TEST(EfficientI4MatmulTest, GEMVTest)
{
    const int prom_m = 1;
    const int prom_n = 256;
    const int prom_k = 256;
    const int bits = 4;
    const int group_size = prom_k;

    half* A = new half[prom_m * prom_k];
    int8_t* B = new int8_t[prom_k * prom_n];
    int8_t* qB_interleave = new int8_t[prom_k * prom_n / (8 / bits)];
    half* C = new half[prom_m * prom_n];
    half* s = new half[prom_n * (prom_k / group_size)];
    half* s_perm = new half[prom_n * (prom_k / group_size)];

    // Initialize A and B
    for (int i = 0; i < prom_m * prom_k; i++)
    {
        A[i] = __float2half(rand() / (float)RAND_MAX);
    }
    for (int i = 0; i < prom_k * prom_n; i++)
    {
        B[i] = rand() % 4 - 2;
    }
    for (int i = 0; i < prom_k * prom_n / group_size; i++)
    {
        // s[i] = __float2half(0.1);
        s[i] = __float2half(rand() / (float)RAND_MAX);
    }
    
    weight_pre_process(B, qB_interleave, bits, prom_k, prom_n);
    // print the first 10 elements and last 10 elements of C
    for (int i = 0; i < 10; i++)
    {
        printf("%d ", B[i]);
    }
    printf(" ... ");
    for (int i = prom_k * prom_n - 10; i < prom_k * prom_n; i++)
    {
        printf("%d ", B[i]);
    }
    // print interleave of B
    for (int i = 0; i < 10; i++)
    {
        printf("%d ", qB_interleave[i]);
    }
    printf(" ... ");
    for (int i = prom_k * prom_n / (8 / bits) - 10; i < prom_k * prom_n / (8 / bits); i++)
    {
        printf("%d ", qB_interleave[i]);
    }
    printf("\n");
    // print last 10 of qb_interleave
    for (int i = prom_k * prom_n / (8 / bits) - 10; i < prom_k * prom_n / (8 / bits); i++)
    {
        printf("%d ", qB_interleave[i]);
    }
    printf("\n");
    // print last 10 of B
    for (int i = prom_k * prom_n - 10; i < prom_k * prom_n; i++)
    {
        printf("%d ", B[i]);
    }
    printf("\n");
    // print last 10 of s
    for (int i = prom_n * (prom_k / group_size) - 10; i < prom_n * (prom_k / group_size); i++)
    {
        printf("%f ", __half2float(s[i]));
    }
    printf("\n");
    scale_pre_process(s, s_perm, prom_k, prom_n, group_size);
    // define cuda variables
    float* d_workspace = nullptr;
    cudaCheckLastError(cudaMalloc((void**)&d_workspace, prom_n * prom_k * 16 * sizeof(float)));

    half* d_A;
    int8_t* d_qB;
    half* d_C;
    half* d_s;
    cudaCheckLastError(cudaMalloc((void**)&d_A, prom_m * prom_k * sizeof(half)));
    cudaCheckLastError(cudaMalloc((void**)&d_qB, prom_k * prom_n / (8 / bits) * sizeof(int8_t)));
    cudaCheckLastError(cudaMalloc((void**)&d_C, prom_m * prom_n * sizeof(half)));
    cudaCheckLastError(cudaMalloc((void**)&d_s, prom_n * (prom_k / group_size) * sizeof(half)));
    // copy A and B to device
    cudaCheckLastError(cudaMemcpy(d_A, A, prom_m * prom_k * sizeof(half), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(d_qB, qB_interleave, prom_n * prom_k / (8 / bits) * sizeof(int8_t), cudaMemcpyHostToDevice));
    cudaCheckLastError(cudaMemcpy(d_s, s_perm, prom_n * (prom_k / group_size) * sizeof(half), cudaMemcpyHostToDevice));

    // allocate workspace
    // call the kernel
    int ret = marlin_cuda(d_A, d_qB, d_C, d_s, prom_m, prom_n, prom_k, d_workspace, group_size == prom_k? -1: group_size);
    printf("ret = %d\n", ret);

    // copy C back to host
    cudaCheckLastError(cudaMemcpy(C, d_C, prom_m * prom_n * sizeof(half), cudaMemcpyDeviceToHost));
    // print the first 10 elements and last 10 elements of C
    for (int i = 0; i < 10; i++)
    {
        printf("%f ", __half2float(C[i]));
    }
    printf(" ... ");
    for (int i = prom_m * prom_n - 10; i < prom_m * prom_n; i++)
    {
        printf("%f ", __half2float(C[i]));
    }
    printf("\n");

    // ref calculation
    float* ref_C = new float[prom_m * prom_n];
    // zero fill
    for (int i = 0; i < prom_m * prom_n; i++)
    {
        ref_C[i] = __float2half(0.0);
    }
    // 
    for (int i = 0; i < prom_m; i++)
    {
        for (int j = 0; j < prom_n; j++)
        {
            ref_C[i * prom_n + j] = __float2half(0.0);
            for (int k = 0; k < prom_k; k++)
            {
                ref_C[i * prom_n + j] += float(A[i * prom_k + k]) *  (float(B[k * prom_n + j]) * float(s[(k / group_size) * prom_n + j]));
            }
        }
    }
    for (int i = 0; i < 10; i++)
    {
        printf("%f ", __half2float(ref_C[i]));
    }
    printf(" ... ");
    for (int i = prom_m * prom_n - 10; i < prom_m * prom_n; i++)
    {
        printf("%f ", __half2float(ref_C[i]));
    }
    printf("\n");

    // check the result
    for (int i = 0; i < prom_m * prom_n; i++)
    {
        EXPECT_NEAR(__half2float(C[i]), __half2float(ref_C[i]), 1e-1);
    }

    // free memory
    delete[] A;
    delete[] B;
    delete[] C;
    cudaCheckLastError(cudaFree(d_A));
    cudaCheckLastError(cudaFree(d_qB));
    cudaCheckLastError(cudaFree(d_C));
}
