# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import bitblas
import bitblas.testing
from bitblas.ops.general_matmul_splitk import MatmulWithSplitK, MatmulConfigWithSplitK


def get_codegen_result(ops):
    code = ops.get_source()
    return code


# fmt: off
def matmul_codegen_default(M, N, K, A_dtype, W_dtype, accum_dtype, out_dtype, layout, with_bias,
                           group_size, with_scaling, with_zeros, zeros_mode):

    matmul_config = MatmulConfigWithSplitK(
        M=M,
        N=N,
        K=K,
        A_dtype=A_dtype,
        W_dtype=W_dtype,
        accum_dtype=accum_dtype,
        out_dtype=out_dtype,
        layout=layout,
        with_bias=with_bias,
        group_size=group_size,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        zeros_mode=zeros_mode,
        propagate_a=False,
        propagate_b=False,
    )
    matmul = MatmulWithSplitK(config=matmul_config, enable_tuning=False)
    assert get_codegen_result(matmul)


def test_matmul_codegen_default():
    matmul_codegen_default(1, 4096, 12800, "float16", "float16", "float16", "float16", "nt", False,
                           -1, False, False, None)
    matmul_codegen_default(16, 4096, 12800, "float16", "float16", "float16", "float16", "nt", False,
                           -1, False, False, None)


def matmul_torch_forward_consistent(SplitK, M, N, K, A_dtype, W_dtype, accum_dtype, out_dtype,
                                    layout, with_bias, group_size, with_scaling, with_zeros,
                                    zeros_mode):
    import torch
    torch.random.manual_seed(0)
    matmul_config = MatmulConfigWithSplitK(
        k_split=SplitK,
        M=M,
        N=N,
        K=K,
        A_dtype=A_dtype,
        W_dtype=W_dtype,
        accum_dtype=accum_dtype,
        out_dtype=out_dtype,
        layout=layout,
        with_bias=with_bias,
        group_size=group_size,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        zeros_mode=zeros_mode,
        propagate_a=False,
        propagate_b=False,
    )
    matmul = MatmulWithSplitK(config=matmul_config, enable_tuning=False)

    input_shape = (M, K)
    weight_shape = (N, K) if layout == "nt" else (K, N)
    inputs = []
    inputs.append(torch.rand(input_shape, dtype=torch.float16).cuda() - 0.5)
    inputs.append(torch.rand(weight_shape, dtype=torch.float16).cuda() - 0.5)

    output_bitblas = matmul.forward(*inputs)
    output_torch = torch.matmul(inputs[0], inputs[1].t() if layout == "nt" else inputs[1])
    bitblas.testing.torch_assert_close(output_torch, output_bitblas, rtol=1e-2, atol=1e-1, max_mismatched_ratio=1e-2)


def test_matmul_torch_forward_consistent():
    matmul_torch_forward_consistent(1, 1, 4096, 12800, "float16", "float16", "float16", "float16",
                                    "nt", False, -1, False, False, None)
    matmul_torch_forward_consistent(4, 1, 4096, 12800, "float16", "float16", "float16", "float16",
                                    "nt", False, -1, False, False, None)


def matmul_torch_forward_fp8e4m3(SplitK, M, N, K, A_dtype, W_dtype, accum_dtype, out_dtype, layout,
                                 with_bias, group_size, with_scaling, with_zeros, zeros_mode):
    import torch
    torch.random.manual_seed(0)
    matmul_config = MatmulConfigWithSplitK(
        k_split=SplitK,
        M=[1, 16],
        N=N,
        K=K,
        A_dtype=A_dtype,
        W_dtype=W_dtype,
        accum_dtype=accum_dtype,
        out_dtype=out_dtype,
        layout=layout,
        with_bias=with_bias,
        group_size=group_size,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        zeros_mode=zeros_mode,
        propagate_a=False,
        propagate_b=False,
    )
    matmul = MatmulWithSplitK(config=matmul_config, enable_tuning=False)

    input_shape = (M, K)
    weight_shape = (N, K) if layout == "nt" else (K, N)

    def map_torch_type(intype):

        typemap = {
            'e4m3_float8': torch.float8_e4m3fn,
            'e5m2_float8': torch.float8_e5m2,
        }
        if intype in typemap:
            return typemap[intype]
        else:
            return getattr(torch, intype)

    numpytype_a = map_torch_type(A_dtype)
    numpytype_b = map_torch_type(W_dtype)

    torch_a = torch.rand(M * K).uniform_(-1, 1).reshape(input_shape).type(numpytype_a).cuda()
    torch_b = torch.rand(N * K).uniform_(-1, 1).reshape(weight_shape).type(numpytype_b).cuda()
    ref_out = torch.matmul(torch_a.to(torch.float32),
                           torch_b.t().to(torch.float32)) if layout == "nt" else torch.matmul(
                               torch_a.to(torch.float32), torch_b.to(torch.float32))
    ref_out = ref_out.to(torch.float16)
    bitblas_out = torch.empty_like(ref_out)
    matmul.forward(torch_a, torch_b)
    print("torch_ref_out", ref_out)
    print("bitblas_out", bitblas_out)

    matmul.forward(torch_a, torch_b, output=bitblas_out)
    print("torch_ref_out", ref_out)
    print("bitblas_out", bitblas_out)

    matmul.forward(torch_a, torch_b, output=bitblas_out)
    print("torch_ref_out", ref_out)
    print("bitblas_out", bitblas_out)

    torch.testing.assert_close(bitblas_out, ref_out, rtol=1e0, atol=1e-1)


@bitblas.testing.requires_cuda_compute_version(8, 9)
def test_matmul_torch_forward_fp8e4m3():
    matmul_torch_forward_fp8e4m3(1, 16, 4096, 12800, "e4m3_float8", "e4m3_float8", "float32",
                                 "float16", "nt", False, -1, False, False, None)
    matmul_torch_forward_fp8e4m3(4, 16, 4096, 12800, "e4m3_float8", "e4m3_float8", "float32",
                                 "float16", "nt", False, -1, False, False, None)


# fmt: on
if __name__ == "__main__":
    bitblas.testing.main()
