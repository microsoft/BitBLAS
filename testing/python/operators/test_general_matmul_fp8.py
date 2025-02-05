import torch
import bitblas
import bitblas.testing
from bitblas import MatmulConfig, Matmul
import logging
from bitblas import set_log_level

set_log_level(logging.DEBUG)


def matmul_torch_forward(M, N, K, A_dtype, W_dtype, accum_dtype, out_dtype, layout, with_bias,
                         group_size, with_scaling, with_zeros, zeros_mode):
    torch.random.manual_seed(0)

    matmul_config = MatmulConfig(
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
    )
    matmul = Matmul(config=matmul_config, enable_tuning=False, backend="tl")

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
    numpytype_c = map_torch_type(out_dtype)

    torch_a = torch.rand(M * K).uniform_(-1, 1).reshape(input_shape).type(numpytype_a).cuda()
    torch_b = torch.rand(N * K).uniform_(-1, 1).reshape(weight_shape).type(numpytype_b).cuda()
    ref_out = torch.matmul(torch_a.to(torch.float32),
                           torch_b.t().to(torch.float32)) if layout == "nt" else torch.matmul(
                               torch_a.to(torch.float32), torch_b.to(torch.float32))
    ref_out = ref_out.to(numpytype_c)

    print("torch_ref_out", ref_out)
    new_torch_b = matmul.transform_weight(torch_b)
    bitblas_out = matmul(torch_a, new_torch_b)
    print("bitblas_out", bitblas_out)


@bitblas.testing.requires_cuda_compute_version(8, 9)
def test_matmul_torch_forward():
    matmul_torch_forward(1, 1024, 1024, "e4m3_float8", "e4m3_float8", "float32", "float32", "nt",
                         None, None, None, None, None)
    matmul_torch_forward(1024, 1024, 1024, "e4m3_float8", "e4m3_float8", "float32", "float32", "nt",
                         None, None, None, None, None)
    matmul_torch_forward(1, 1024, 1024, "e5m2_float8", "e5m2_float8", "float32", "float32", "nt",
                         None, None, None, None, None)
    matmul_torch_forward(1024, 1024, 1024, "e5m2_float8", "e5m2_float8", "float32", "float32", "nt",
                         None, None, None, None, None)


def matmul_torch_forward_weight_dequantize(M, N, K, A_dtype, W_dtype, accum_dtype, out_dtype,
                                           layout, with_bias, group_size, with_scaling, with_zeros,
                                           zeros_mode):
    torch.random.manual_seed(0)

    matmul_config = MatmulConfig(
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
    matmul = Matmul(config=matmul_config, enable_tuning=False, backend="tir")
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
    numpytype_c = map_torch_type(out_dtype)

    torch_a = torch.rand(M * K).uniform_(-1, 1).reshape(input_shape).type(numpytype_a).cuda()
    torch_b = torch.rand(N * K).uniform_(-1, 1).reshape(weight_shape).type(numpytype_b).cuda()

    torch_fp16_a = torch_a.to(torch.float16)
    torch_fp16_b = torch_b.t().to(torch.float16) if layout == "nt" else torch_b.to(torch.float16)
    scale_tensor = None
    if with_scaling:
        if group_size is None:
            group_size = -1
        if group_size == -1:
            group_size = K
        scale_tensor = torch.rand(N * K // group_size).uniform_(-1, 1).reshape(
            [N, K // group_size]).type(torch.float16).cuda()
        rescale_b = torch.zeros_like(torch_b).type(torch.float16)
        for i in range(K):
            rescale_b[:, i] = torch_b.to(torch.float16)[:, i] * scale_tensor[:, i // group_size]
        torch_fp16_b = rescale_b.t().to(torch.float16) if layout == "nt" else rescale_b.to(
            torch.float16)

    ref_out = torch.matmul(torch_fp16_a, torch_fp16_b)
    ref_out = ref_out.to(numpytype_c)

    permuted_inputs = []
    permuted_inputs.append(torch_a)
    permuted_inputs.append(matmul.transform_weight(torch_b))
    if with_scaling:
        permuted_inputs.append(scale_tensor)
    bitblas_out = matmul(*permuted_inputs)

    print("torch_ref_out", ref_out)
    print("bitblas_out", bitblas_out)

    torch.testing.assert_close(ref_out, bitblas_out, rtol=1e-1, atol=1e-1)


@bitblas.testing.requires_cuda_compute_version(8, 9)
def test_matmul_torch_forward_weight_dequantize():
    matmul_torch_forward_weight_dequantize(1, 1024, 1024, "float16", "e4m3_float8", "float16",
                                           "float16", "nt", None, None, None, None, None)
    matmul_torch_forward_weight_dequantize(1024, 1024, 1024, "float16", "e4m3_float8", "float16",
                                           "float16", "nt", None, None, None, None, None)
    matmul_torch_forward_weight_dequantize(1, 1024, 1024, "float16", "e4m3_float8", "float16",
                                           "float16", "nt", None, 32, True, None, None)
    matmul_torch_forward_weight_dequantize(1024, 1024, 1024, "float16", "e4m3_float8", "float16",
                                           "float16", "nt", None, 32, True, None, None)


if __name__ == "__main__":
    bitblas.testing.main()
