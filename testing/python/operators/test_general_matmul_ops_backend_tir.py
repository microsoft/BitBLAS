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
    )
    matmul = Matmul(config=matmul_config, enable_tuning=False, backend="tir")
    assert get_codegen_result(matmul)


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


def matmul_finetune(M, N, K, A_dtype, W_dtype, accum_dtype, out_dtype, layout, with_bias,
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
    )
    matmul = Matmul(config=matmul_config, enable_tuning=False, backend="tir")
    matmul.hardware_aware_finetune(topk=10)
    assert get_codegen_result(matmul)


def test_matmul_finetune():
    matmul_finetune(1, 768, 768, "float16", "float16", "float16", "float16", "nt", False, -1, False,
                    False, None),
    matmul_finetune(768, 768, 768, "float16", "float16", "float16", "float16", "nt", False, -1,
                    False, False, None),
    matmul_finetune(1, 768, 768, "int8", "int8", "int32", "int8", "nt", False, -1, False, False,
                    None),
    matmul_finetune(768, 768, 768, "int8", "int8", "int32", "int8", "nt", False, -1, False, False,
                    None),
    matmul_finetune(1, 768, 768, "float16", "uint4", "float16", "float16", "nt", False, -1, False,
                    False, None),
    matmul_finetune(1, 768, 768, "float16", "uint4", "float16", "float16", "nt", True, -1, False,
                    False, None),
    matmul_finetune(1, 768, 768, "float16", "uint4", "float16", "float16", "nt", False, -1, True,
                    False, None),
    matmul_finetune(1, 768, 768, "float16", "uint4", "float16", "float16", "nt", False, -1, True,
                    True, "original"),
    matmul_finetune(768, 768, 768, "float16", "uint4", "float16", "float16", "nt", False, -1, False,
                    False, None),
    matmul_finetune(768, 768, 768, "float16", "uint4", "float16", "float16", "nt", True, -1, False,
                    False, None),
    matmul_finetune(768, 768, 768, "float16", "uint4", "float16", "float16", "nt", False, -1, True,
                    False, None),
    matmul_finetune(768, 768, 768, "float16", "uint4", "float16", "float16", "nt", False, -1, True,
                    True, "original"),


def matmul_torch_forward(M, N, K, A_dtype, W_dtype, accum_dtype, out_dtype, layout, with_bias,
                         group_size, with_scaling, with_zeros, zeros_mode):
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
        propagate_a=False,
    )
    matmul = Matmul(config=matmul_config, enable_tuning=False, backend="tir")

    input_shape = (M, K)
    weight_shape = (N, K) if layout == "nt" else (K, N)
    output_shape = (M, N)
    inputs = []
    inputs.append(torch.rand(input_shape, dtype=torch.float16).cuda() - 0.5)
    source_format, bit = matmul.BITBLAS_TRICK_DTYPE_MAP[W_dtype]
    maxq = 2**(bit - 1)
    zeros = maxq
    if source_format == "uint":
        inputs.append(torch.randint(0, maxq, weight_shape, dtype=torch.int8).cuda())
    elif source_format == "int":
        inputs.append(torch.randint(-maxq, maxq, weight_shape, dtype=torch.int8).cuda())
    else:
        raise NotImplementedError

    inputs.append(torch.rand(output_shape, dtype=torch.float16).cuda())

    intweight = inputs[1]
    intweight = intweight.cpu().to(torch.int8)
    if source_format == "int":
        intweight = intweight + maxq
    if with_zeros:
        inputs[1] = inputs[1] - zeros
    bias = torch.rand((output_shape[-1],), dtype=torch.float16).cuda()
    ref_result = torch.matmul(inputs[0],
                              (inputs[1].t() if layout == "nt" else inputs[1]).to(torch.float16))
    if with_bias:
        ref_result = ref_result + bias
    permuted_inputs = []
    permuted_inputs.append(inputs[0])
    if matmul.weight_transform is not None:
        permuted_inputs.append(matmul.weight_transform(intweight.cpu()).cuda())
    else:
        permuted_inputs.append(intweight)
    if with_scaling:
        if group_size == -1:
            group_size = K
        permuted_inputs.append(torch.ones([N, K // group_size], dtype=torch.float16).cuda())
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
    if with_bias:
        permuted_inputs.append(bias)
    permuted_inputs.append(inputs[2])
    matmul(*permuted_inputs[:-1], output=permuted_inputs[-1])
    print(permuted_inputs[-1])
    print(ref_result)
    if zeros_mode == "rescale":
        torch.testing.assert_close(permuted_inputs[-1], ref_result, rtol=1e2, atol=1e0)
    else:
        torch.testing.assert_close(permuted_inputs[-1], ref_result, rtol=1e2, atol=1e0)


def test_matmul_torch_forward():
    matmul_torch_forward(1, 1024, 1024, "float16", "int4", "float16", "float16", "nt", None, None,
                         None, None, None)
    matmul_torch_forward(1, 768, 768, "float16", "uint4", "float16", "float16", "nt", False, -1,
                         False, False, None)
    matmul_torch_forward(1, 768, 768, "float16", "uint4", "float16", "float16", "nt", True, -1,
                         False, False, None),
    matmul_torch_forward(1, 768, 768, "float16", "uint4", "float16", "float16", "nt", False, -1,
                         True, False, None),
    matmul_torch_forward(1, 768, 768, "float16", "uint4", "float16", "float16", "nt", False, -1,
                         True, True, "original")
    matmul_torch_forward(768, 768, 768, "float16", "uint4", "float16", "float16", "nt", False, -1,
                         False, False, None)
    matmul_torch_forward(768, 768, 768, "float16", "uint4", "float16", "float16", "nt", True, -1,
                         False, False, None)
    matmul_torch_forward(768, 768, 768, "float16", "uint4", "float16", "float16", "nt", False, -1,
                         True, False, None)
    matmul_torch_forward(768, 768, 768, "float16", "uint4", "float16", "float16", "nt", False, -1,
                         True, True, "original")


def matmul_transform_weight(
    M,
    N,
    K,
    A_dtype,
    W_dtype,
    accum_dtype,
    out_dtype,
    with_bias,
):
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
        with_bias=with_bias,
    )
    matmul = Matmul(config=matmul_config, enable_tuning=False, backend="tir")

    input_shape = (M, K)
    weight_shape = (N, K)
    output_shape = (M, N)

    _, bit = matmul.BITBLAS_TRICK_DTYPE_MAP[W_dtype]
    maxq = 2**(bit - 1)

    input_tensor = torch.rand(input_shape, dtype=torch.float16).cuda()
    intweight_tensor = torch.randint(0, maxq, weight_shape, dtype=torch.int8).cuda()
    output_tensor = torch.rand(output_shape, dtype=torch.float16).cuda()

    bias = torch.rand((output_shape[-1],), dtype=torch.float16).cuda()
    ref_result = torch.matmul(input_tensor, intweight_tensor.t().to(torch.float16))
    if with_bias:
        ref_result = ref_result + bias

    bitblas_inputs = [input_tensor]
    intweight_tensor = matmul.transform_weight(intweight_tensor)
    bitblas_inputs.append(intweight_tensor)
    if with_bias:
        bitblas_inputs.append(bias)
    output_tensor = matmul(*bitblas_inputs)
    torch.testing.assert_close(output_tensor, ref_result, rtol=1e2, atol=1e0)


def test_matmul_transform_weight():
    matmul_transform_weight(1, 768, 768, "float16", "uint4", "float16", "float16", False)
    matmul_transform_weight(1, 768, 768, "float16", "int4", "float16", "float16", False)
    matmul_transform_weight(768, 768, 768, "float16", "uint4", "float16", "float16", False)
    matmul_transform_weight(768, 768, 768, "float16", "int4", "float16", "float16", False)


# fmt: on
if __name__ == "__main__":
    bitblas.testing.main()
