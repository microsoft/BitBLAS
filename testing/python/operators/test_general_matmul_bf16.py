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
        propagate_b=2,
    )
    matmul = Matmul(config=matmul_config, enable_tuning=True)

    input_shape = (M, K)
    weight_shape = (N, K) if layout == "nt" else (K, N)

    def map_torch_type(intype):
        return getattr(torch, intype)

    torch_type_a = map_torch_type(A_dtype)
    torch_type_b = map_torch_type(W_dtype)
    torch_type_c = map_torch_type(out_dtype)
    torch_a = torch.rand(M * K).uniform_(-1, 1).reshape(input_shape).type(torch_type_a).cuda()
    torch_b = torch.rand(N * K).uniform_(-1, 1).reshape(weight_shape).type(torch_type_b).cuda()
    ref_out = torch.matmul(torch_a.to(torch.float32),
                           torch_b.t().to(torch.float32)) if layout == "nt" else torch.matmul(
                               torch_a.to(torch.float32), torch_b.to(torch.float32))

    ref_out = ref_out.to(torch_type_c)

    print("torch_ref_out", ref_out)
    new_torch_b = matmul.transform_weight(torch_b)
    bitblas_out = matmul(torch_a, new_torch_b)
    print("bitblas_out", bitblas_out)


def matmul_torch_forward_weight_dequantize(M, N, K, A_dtype, W_dtype, accum_dtype, out_dtype,
                                           layout, with_bias, group_size, with_scaling, with_zeros,
                                           zeros_mode):
    import torch
    import numpy as np
    from bitblas.quantization import general_compress
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
        fast_decoding=False,
        propagate_a=False,
    )
    matmul = Matmul(config=matmul_config, enable_tuning=True)

    input_shape = (M, K)
    weight_shape = (N, K) if layout == "nt" else (K, N)
    output_shape = (M, N)
    inputs = []
    inputs.append(torch.rand(input_shape, dtype=getattr(torch, A_dtype)).cuda() - 0.5)
    source_format, bit = matmul.BITBLAS_TRICK_DTYPE_MAP[W_dtype]
    maxq = 2**(bit - 1)
    zeros = maxq
    if source_format == "uint":
        inputs.append(torch.randint(0, maxq, weight_shape, dtype=torch.int8).cuda())
    elif source_format == "int":
        inputs.append(torch.randint(-maxq, maxq, weight_shape, dtype=torch.int8).cuda())
    else:
        raise NotImplementedError

    inputs.append(torch.zeros(output_shape, dtype=getattr(torch, out_dtype)).cuda())

    intweight = inputs[1]
    intweight = intweight.cpu().to(torch.int8)
    if source_format == "int":
        intweight = intweight + maxq
    if with_zeros:
        inputs[1] = inputs[1] - zeros
    bias = torch.rand((output_shape[-1],), dtype=getattr(torch, out_dtype)).cuda()

    permuted_inputs = []
    permuted_inputs.append(inputs[0])
    if matmul.weight_transform is not None:
        permuted_inputs.append(matmul.weight_transform(intweight.cpu()).cuda())
    else:
        permuted_inputs.append(intweight)
    if with_scaling:
        if group_size == -1:
            group_size = K
        permuted_inputs.append(
            torch.randn((N, K // group_size), dtype=getattr(torch, A_dtype)).cuda())
    if with_zeros:
        if zeros_mode == "original":
            permuted_inputs.append(
                torch.ones([N, K // group_size], dtype=getattr(torch, A_dtype)).cuda() * zeros)
        elif zeros_mode == "rescale":
            original_zeros = torch.ones([N, K // group_size], dtype=getattr(torch,
                                                                            A_dtype)).cuda() * zeros
            scaled_zeros = original_zeros * permuted_inputs[-1]
            permuted_inputs.append(scaled_zeros)
        elif zeros_mode == "quantized":
            original_zeros = torch.ones([K // group_size, N], dtype=torch.int8).cuda() * zeros
            qzeros = general_compress(
                original_zeros.cpu().numpy(), source_bits=bit, storage_dtype=np.int8)
            permuted_inputs.append(torch.from_numpy(qzeros).cuda())
        else:
            raise NotImplementedError
    if with_bias:
        permuted_inputs.append(bias)
    permuted_inputs.append(inputs[2])
    matmul(*permuted_inputs[:-1], output=permuted_inputs[-1])

    args = [inputs[0]]
    b = inputs[1]
    if with_scaling:
        scale = permuted_inputs[2]
        rescale_b = torch.empty_like(b, dtype=torch.bfloat16)
        for i in range(N):
            for j in range(K):
                if with_zeros:
                    zeros = permuted_inputs[3]
                    if zeros_mode == "original":
                        rescale_b[i, j] = (b[i, j] - zeros[i, j // group_size]) * scale[i, j //
                                                                                        group_size]
                    elif zeros_mode == "rescale":
                        rescale_b[i, j] = (
                            b[i, j] * scale[i, j // group_size] + zeros[i, j // group_size])
                    else:
                        raise NotImplementedError
                else:
                    rescale_b[i, j] = b[i, j] * scale[i, j // group_size]
        args.append(rescale_b.t().cuda())
    else:
        args.append(b.t().cuda().to(getattr(torch, A_dtype)))
    ref_result = torch.matmul(*args).to(getattr(torch, out_dtype))
    print(permuted_inputs[-1])
    print(ref_result)
    # when scaling is not enabled, we should have some mismatch due to the scaling factor
    bitblas.testing.torch_assert_close(permuted_inputs[-1], ref_result, rtol=1e2, atol=1e0)


@bitblas.testing.requires_cuda_compute_version(8, 0)
def test_matmul_torch_forward_weight_dequantize():
    matmul_torch_forward_weight_dequantize(1, 1024, 1024, "bfloat16", "uint4", "float32", "float32",
                                           "nt", None, None, None, None, None)
    matmul_torch_forward_weight_dequantize(1024, 1024, 1024, "bfloat16", "uint4", "float32",
                                           "float32", "nt", None, None, None, None, None)
    matmul_torch_forward_weight_dequantize(1, 1024, 1024, "bfloat16", "uint4", "float32", "float32",
                                           "nt", None, 32, True, None, None)
    matmul_torch_forward_weight_dequantize(1024, 1024, 1024, "bfloat16", "uint4", "float32",
                                           "float32", "nt", None, 32, True, None, None)


if __name__ == "__main__":
    bitblas.testing.main()
