# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import bitblas
import bitblas.testing
from bitblas import MatmulConfig, Matmul
import logging
from bitblas import set_log_level

set_log_level(logging.DEBUG)


def get_codegen_result(ops):
    code = ops.get_source()
    return code


# fmt: off
def matmul_codegen_default(M, N, K, A_dtype, W_dtype, accum_dtype, out_dtype, layout, with_bias,
                           group_size, with_scaling, with_zeros, zeros_mode):

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
    matmul = Matmul(config=matmul_config, enable_tuning=False, backend="tl")
    assert get_codegen_result(matmul)


def matmul_finetune(M,
                    N,
                    K,
                    A_dtype,
                    W_dtype,
                    accum_dtype,
                    out_dtype,
                    layout,
                    with_bias,
                    group_size,
                    with_scaling,
                    with_zeros,
                    zeros_mode,
                    propagate_b=False):

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
        propagate_b=propagate_b,
    )
    matmul = Matmul(config=matmul_config, enable_tuning=False, backend="tl")
    matmul.hardware_aware_finetune(topk=20)
    print(matmul.get_source())
    assert get_codegen_result(matmul)


def matmul_torch_forward(M,
                         N,
                         K,
                         A_dtype,
                         W_dtype,
                         accum_dtype,
                         out_dtype,
                         layout,
                         with_bias,
                         group_size,
                         with_scaling,
                         with_zeros,
                         zeros_mode,
                         propagate_b=None):
    import torch
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
        propagate_b=propagate_b,
    )
    matmul = Matmul(config=matmul_config, enable_tuning=False, backend="tl")

    assert layout == "nt", "Only support nt layout"

    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, A_dtype))
    B = torch.rand(N, K, device="cuda", dtype=getattr(torch, W_dtype))

    LB = matmul.transform_weight(B)
    bitblas_output = matmul(A, LB)
    ref_output = torch.matmul(A, B.T).to(getattr(torch, out_dtype))
    torch.testing.assert_close(bitblas_output, ref_output, rtol=1e-1, atol=1e-1)


def matmul_torch_forward_dequant(M,
                                 N,
                                 K,
                                 A_dtype,
                                 W_dtype,
                                 accum_dtype,
                                 out_dtype,
                                 layout,
                                 with_bias=False,
                                 group_size=-1,
                                 with_scaling=False,
                                 with_zeros=False,
                                 zeros_mode="original",
                                 fast_decoding=True,
                                 propagate_b=None):
    import torch
    torch.random.manual_seed(0)
    import numpy as np
    from bitblas.quantization import general_compress

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
        fast_decoding=fast_decoding,
        propagate_a=False,
        propagate_b=propagate_b,
    )
    matmul = Matmul(config=matmul_config, enable_tuning=False, backend="tl")
    input_shape = (M, K)
    weight_shape = (N, K) if layout == "nt" else (K, N)
    output_shape = (M, N)

    A = torch.rand(input_shape, dtype=torch.float16).cuda() - 0.5

    source_format, bit = matmul.BITBLAS_TRICK_DTYPE_MAP[W_dtype]
    maxq = 2**(bit - 1)
    zeros = maxq
    if source_format == "uint":
        intweight = torch.randint(0, maxq, weight_shape, dtype=torch.int8).cuda()
    elif source_format == "int":
        intweight = torch.randint(-maxq, maxq, weight_shape, dtype=torch.int8).cuda()
    else:
        raise NotImplementedError

    ref_inputs = []
    ref_inputs.append(A)
    ref_inputs.append(intweight)

    if source_format == "int":
        intweight = intweight + maxq

    permuted_inputs = []
    permuted_inputs.append(A)
    if matmul.weight_transform is not None:
        permuted_inputs.append(matmul.weight_transform(intweight.cpu()).cuda())
    else:
        permuted_inputs.append(intweight)
    if with_scaling:
        if group_size == -1:
            group_size = K
        permuted_inputs.append(torch.rand([N, K // group_size], dtype=torch.float16).cuda())
        ref_inputs.append(permuted_inputs[-1])
    if with_zeros:
        if zeros_mode == "original":
            permuted_inputs.append(
                torch.ones([N, K // group_size], dtype=torch.float16).cuda() * zeros)
        elif zeros_mode == "rescale":
            original_zeros = torch.ones([N, K // group_size], dtype=torch.float16).cuda() * zeros
            scaled_zeros = original_zeros * permuted_inputs[-1]
            permuted_inputs.append(scaled_zeros)
        elif zeros_mode == "quantized":
            original_zeros = torch.ones([K // group_size, N], dtype=torch.int8).cuda() * zeros
            qzeros = general_compress(
                original_zeros.cpu().numpy(), source_bits=bit, storage_dtype=np.int8)
            permuted_inputs.append(torch.from_numpy(qzeros).cuda())
        else:
            raise NotImplementedError
        ref_inputs.append(permuted_inputs[-1])

    C = torch.zeros(output_shape, dtype=torch.float16).cuda()
    Bias = torch.rand((output_shape[-1],), dtype=torch.float16).cuda()

    if with_bias:
        permuted_inputs.append(Bias)
        ref_inputs.append(Bias)

    permuted_inputs.append(C)
    matmul(*permuted_inputs[:-1], output=permuted_inputs[-1])

    def ref_program(A, intweight, scale=None, zeros=None, Bias=None):
        import torch

        B = intweight
        _, K = B.shape

        if with_scaling:
            # Calculate group indices for each column (group_size determines the grouping)
            group_indices = torch.arange(K, device=B.device) // group_size

            # Broadcast zeros and scale to match the shape of B
            scale_expanded = scale[:, group_indices]  # Shape: [N, K]

            if with_zeros:
                if zeros_mode == "original":
                    zeros_expanded = zeros[:, group_indices]  # Shape: [N, K]
                    # Subtract zeros and then scale
                    rescale_b = (B - zeros_expanded) * scale_expanded
                elif zeros_mode == "rescale":
                    zeros_expanded = zeros[:, group_indices]  # Shape: [N, K]
                    # Scale first and then add zeros
                    rescale_b = B * scale_expanded - zeros_expanded
                elif zeros_mode == "quantized":
                    dequant_zeros = (
                        torch.zeros(zeros.shape[0], zeros.shape[1] * 8 // 4,
                                    dtype=torch.half).to(torch.half).to(zeros.device))
                    for i in range(dequant_zeros.shape[0]):
                        for j in range(dequant_zeros.shape[1]):
                            dequant_zeros[i][j] = ((zeros[i][j // 2] >> (4 * (j % 2))) & 0xF).to(torch.half)
                    zeros_expanded = dequant_zeros[group_indices, :]  # Shape: [N, K]
                    # Subtract zeros and then scale
                    rescale_b = (B - zeros_expanded) * scale_expanded
                else:
                    # Raise an error for unsupported zeros_mode
                    raise NotImplementedError(f"Unsupported zeros_mode: {zeros_mode}")
            else:
                # Apply scaling without zeros adjustment
                rescale_b = B * scale_expanded
        else:
            # If scaling is disabled, directly use B
            rescale_b = B

        C = torch.matmul(A.to(torch.float), rescale_b.T.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        if with_bias:
            C = C + Bias
        return C

    print("Ref result:")
    ref_result = ref_program(*ref_inputs)
    print(ref_result)
    print("Bitblas result:")
    print(permuted_inputs[-1])
    # print source and ir
    # print(matmul.get_source())
    # print(matmul.scheduled_ir_module)
    bitblas.testing.torch_assert_close(permuted_inputs[-1], ref_result, rtol=1e-2, atol=1e-2, max_mismatched_ratio=0.05)


def test_matmul_codegen_default():
    matmul_codegen_default(1, 768, 768, "float16", "float16", "float16", "float16", "nt", False, -1,
                           False, False, None),
    matmul_codegen_default(768, 768, 768, "float16", "float16", "float16", "float16", "nt", False,
                           -1, False, False, None),
    matmul_codegen_default(1, 768, 768, "int8", "int8", "int32", "int8", "nt", False, -1, False,
                           False, None),
    matmul_codegen_default(768, 768, 768, "int8", "int8", "int32", "int8", "nt", False, -1, False,
                           False, None),
    matmul_codegen_default(1, 768, 768, "float16", "uint4", "float16", "float16", "nt", False, -1,
                           False, False, None),
    matmul_codegen_default(1, 768, 768, "float16", "uint4", "float16", "float16", "nt", True, -1,
                           False, False, None),
    matmul_codegen_default(1, 768, 768, "float16", "uint4", "float16", "float16", "nt", False, -1,
                           True, False, None),
    matmul_codegen_default(1, 768, 768, "float16", "uint4", "float16", "float16", "nt", False, -1,
                           True, True, "original"),
    matmul_codegen_default(768, 768, 768, "float16", "uint4", "float16", "float16", "nt", False, -1,
                           False, False, None),
    matmul_codegen_default(768, 768, 768, "float16", "uint4", "float16", "float16", "nt", True, -1,
                           False, False, None),
    matmul_codegen_default(768, 768, 768, "float16", "uint4", "float16", "float16", "nt", False, -1,
                           True, False, None),
    matmul_codegen_default(768, 768, 768, "float16", "uint4", "float16", "float16", "nt", False, -1,
                           True, True, "original"),


def test_matmul_finetune():
    matmul_finetune(1024, 1024, 1024, "float16", "float16", "float16", "float16", "nt", False, -1,
                    False, False, None, False)
    # dynamic
    matmul_finetune([1, 128], 1024, 1024, "float16", "float16", "float16", "float16", "nt", False, -1,
                    False, False, None, False)

def test_matmul_torch_forward():
    matmul_torch_forward(1024, 1024, 1024, "float16", "float16", "float16", "float16", "nt", None,
                         None, None, None, None, False)
    matmul_torch_forward(1024, 1024, 1024, "float16", "float16", "float16", "float16", "nt", None,
                         None, None, None, None, True)


def test_matmul_torch_dequant_forward():
    # GEMV Test
    matmul_torch_forward_dequant(1, 256, 256, "float16", "uint4", "float16", "float16", "nt")
    matmul_torch_forward_dequant(1, 256, 256, "float16", "uint4", "float16", "float16", "nt", fast_decoding=False)
    matmul_torch_forward_dequant(1, 256, 256, "float16", "int4", "float16", "float16", "nt", group_size=-1, with_scaling=True)
    matmul_torch_forward_dequant(1, 256, 256, "float16", "int4", "float16", "float16", "nt", group_size=32, with_scaling=True)
    matmul_torch_forward_dequant(1, 256, 256, "float16", "uint4", "float16", "float16", "nt", group_size=32, with_scaling=True, with_zeros=True, zeros_mode="original")
    matmul_torch_forward_dequant(1, 256, 256, "float16", "uint4", "float16", "float16", "nt", group_size=32, with_scaling=True, with_zeros=True, zeros_mode="rescale")
    matmul_torch_forward_dequant(1, 256, 256, "float16", "uint4", "float16", "float16", "nt", group_size=32, with_scaling=True, with_zeros=True, zeros_mode="quantized")

    # GEMM Test
    matmul_torch_forward_dequant(256, 256, 256, "float16", "uint4", "float16", "float16", "nt", propagate_b=False)
    matmul_torch_forward_dequant(256, 256, 256, "float16", "int4", "float16", "float16", "nt", group_size=-1, with_scaling=True)
    matmul_torch_forward_dequant(256, 256, 256, "float16", "int4", "float16", "float16", "nt", group_size=32, with_scaling=True)
    matmul_torch_forward_dequant(256, 256, 256, "float16", "uint4", "float16", "float16", "nt", group_size=32, with_scaling=True, with_zeros=True, zeros_mode="original")
    matmul_torch_forward_dequant(256, 256, 256, "float16", "uint4", "float16", "float16", "nt", group_size=32, with_scaling=True, with_zeros=True, zeros_mode="rescale")
    matmul_torch_forward_dequant(256, 256, 256, "float16", "uint4", "float16", "float16", "nt", group_size=32, with_scaling=True, with_zeros=True, zeros_mode="quantized")


# fmt: on
if __name__ == "__main__":
    bitblas.testing.main()
