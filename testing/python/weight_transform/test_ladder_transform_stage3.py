# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import bitblas
import bitblas.testing
import torch
from bitblas.gpu.matmul_analysis import (get_ladder_stage3_map)

torch.manual_seed(0)


def compare_propagate_with_torch_iter_4_fp16(M, N, inner_m=16, inner_n=16):
    b = torch.randn(M // inner_m, N // inner_n, inner_m, inner_n, dtype=torch.float16)

    def shared_32x8_to_mma_32x8_layout(i, j):
        thread_id = (i % 8) * 4 + (j // 2)
        local_id = (i // 8) * 2 + (j % 2)
        return thread_id, local_id

    torch_transformed_b = torch.zeros((M // inner_m, N // inner_n, inner_m, inner_n),
                                      dtype=torch.float16)
    for i in range(M // inner_m):
        for j in range(N // inner_n):
            for ii in range(inner_m):
                for jj in range(inner_n):
                    dummy_ii = (ii * inner_n + jj) // 8
                    dummy_jj = (ii * inner_n + jj) % 8
                    new_dummy_ii, new_dummy_jj = shared_32x8_to_mma_32x8_layout(dummy_ii, dummy_jj)
                    new_ii = (new_dummy_ii * 8 + new_dummy_jj) // inner_n
                    new_jj = (new_dummy_ii * 8 + new_dummy_jj) % inner_n
                    torch_transformed_b[i, j, new_ii, new_jj] = b[i, j, ii, jj]

    ladder_stage3_map, ladder_stage3_map_inverse = get_ladder_stage3_map(dtype="float16")
    bitblas_transformed_b = bitblas.apply_transform_on_input(b, ladder_stage3_map_inverse)

    # assert cpu simulated and ladder compiled results are close
    torch.testing.assert_close(torch_transformed_b, bitblas_transformed_b, rtol=1e-2, atol=1e-2)

    torch_recovered_b = bitblas.apply_transform_on_input(bitblas_transformed_b, ladder_stage3_map)

    # assert recovered results are close to original
    torch.testing.assert_close(torch_recovered_b, b, rtol=1e-2, atol=1e-2)


def test_compare_propagate_with_torch_iter_4_fp16():
    compare_propagate_with_torch_iter_4_fp16(16, 16)
    compare_propagate_with_torch_iter_4_fp16(32, 32)
    compare_propagate_with_torch_iter_4_fp16(64, 64)


def compare_propagate_with_torch_iter_4_int8(M, N, inner_m=16, inner_n=32):
    b = torch.randint(-127, 127, (M // inner_m, N // inner_n, inner_m, inner_n), dtype=torch.int8)

    def shared_32x16_to_mma_32x16_layout(i, j):
        thread_id = (i % 8) * 4 + (j // 4)
        local_id = (i // 8) * 4 + (j % 4)
        return thread_id, local_id

    torch_transformed_b = torch.zeros((M // inner_m, N // inner_n, inner_m, inner_n),
                                      dtype=torch.int8)
    for i in range(M // inner_m):
        for j in range(N // inner_n):
            for ii in range(inner_m):
                for jj in range(inner_n):
                    dummy_ii = (ii * inner_n + jj) // 16
                    dummy_jj = (ii * inner_n + jj) % 16
                    new_dummy_ii, new_dummy_jj = shared_32x16_to_mma_32x16_layout(
                        dummy_ii, dummy_jj)
                    new_ii = (new_dummy_ii * 16 + new_dummy_jj) // inner_n
                    new_jj = (new_dummy_ii * 16 + new_dummy_jj) % inner_n
                    torch_transformed_b[i, j, new_ii, new_jj] = b[i, j, ii, jj]

    ladder_stage3_map, ladder_stage3_map_inverse = get_ladder_stage3_map(dtype="int8")
    bitblas_transformed_b = bitblas.apply_transform_on_input(b, ladder_stage3_map_inverse)

    # assert cpu simulated and ladder compiled results are close
    torch.testing.assert_close(torch_transformed_b, bitblas_transformed_b, rtol=1e-2, atol=1e-2)

    torch_recovered_b = bitblas.apply_transform_on_input(bitblas_transformed_b, ladder_stage3_map)

    # assert recovered results are close to original
    torch.testing.assert_close(torch_recovered_b, b, rtol=1e-2, atol=1e-2)


def test_compare_propagate_with_torch_iter_4_int8():
    compare_propagate_with_torch_iter_4_int8(16, 32)
    compare_propagate_with_torch_iter_4_int8(32, 64)
    compare_propagate_with_torch_iter_4_int8(64, 128)


if __name__ == "__main__":
    bitblas.testing.main()
