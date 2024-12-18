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


def matmul_torch_forward(M, N, K, A_dtype, W_dtype, accum_dtype, out_dtype):
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
    )
    matmul = Matmul(config=matmul_config, enable_tuning=False)
    print(matmul.scheduled_ir_module)

    input_shape = (M, K)
    weight_shape = (N, K)
    output_shape = (M, N)
    input_tensor = torch.rand(input_shape, dtype=torch.float16).cuda() - 0.5

    _, bit = matmul.BITBLAS_TRICK_DTYPE_MAP[W_dtype]
    maxq = 2**(bit - 1)
    weight_tensor = torch.randint(0, maxq, weight_shape, dtype=torch.int8).cuda()
    output_tensor = torch.empty(output_shape, dtype=torch.float16).cuda()

    intweight = weight_tensor
    lut = matmul.lut
    assert lut is not None
    ref_weight = torch.zeros_like(intweight, dtype=torch.float16)
    for j in range(intweight.shape[0]):
        for k in range(intweight.shape[1]):
            ref_weight[j, k] = lut[intweight[j, k]]

    intweight = intweight.cpu().to(torch.int8)
    ref_result = torch.matmul(input_tensor, ref_weight.t().to(torch.float16))
    permuted_inputs = []
    permuted_inputs.append(input_tensor)
    permuted_inputs.append(matmul.weight_transform(intweight.cpu()).cuda())

    permuted_inputs.append(output_tensor)
    matmul(*permuted_inputs[:-1], output=permuted_inputs[-1])
    print(permuted_inputs[-1])
    print(ref_result)
    torch.testing.assert_close(permuted_inputs[-1], ref_result, rtol=1e2, atol=1e0)


def test_matmul_torch_forward():
    matmul_torch_forward(1, 1024, 1024, "float16", "nf4", "float16", "float16")
    matmul_torch_forward(768, 768, 768, "float16", "nf4", "float16", "float16")


# fmt: on
if __name__ == "__main__":
    bitblas.testing.main()
