# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pytest
import tvm
import bitblas
from bitblas.ops.matmul import Matmul, MatmulConfig
from bitblas.utils import tvm_tensor_to_torch

target = tvm.target.Target("nvidia/nvidia-a100")


def get_codegen_result(ops, target):
    code = ops.get_source(target=target)
    return code


# fmt: off
@pytest.mark.parametrize(
    "M,N,K,in_dtype,out_dtype,accum_dtype,with_bias,propagate_a,propagate_b,layout,enable_tuning",
    [
        (16384, 16384, 16384, "float16", "float16", "float16", False, False, False, "nt", False),
        # dynamic shape
        ([1], 16384, 16384, "float16", "float16", "float16", False, False, False, "nt", False),
        ([1, 32], 16384, 16384, "float16", "float16", "float16", False, False, False, "nt", True),
    ],
)
def test_matmul_codegen_default(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    with_bias,
    propagate_a,
    propagate_b,
    layout,
    enable_tuning,
):

    matmul_config = MatmulConfig(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        with_bias=with_bias,
        propagate_a=propagate_a,
        propagate_b=propagate_b,
        layout=layout,
    )
    matmul = Matmul(
        config=matmul_config,
        target=target,
    )
    if enable_tuning:
        matmul.hardware_aware_finetune(topk=20)
    assert get_codegen_result(matmul, target)


@pytest.mark.parametrize(
    "M,N,K,in_dtype,out_dtype,accum_dtype,with_bias,propagate_a,propagate_b,layout",
    [
        (16384, 16384, 16384, "float16", "float16", "float16", False, False, False, "nt"),
        # dynamic shape
        ([1], 16384, 16384, "float16", "float16", "float16", False, False, False, "nt"),
    ],
)
def test_matmul_codegen_finetune(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    with_bias,
    propagate_a,
    propagate_b,
    layout,
):

    matmul_config = MatmulConfig(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        with_bias=with_bias,
        propagate_a=propagate_a,
        propagate_b=propagate_b,
        layout=layout,
    )
    matmul = Matmul(
        config=matmul_config,
        target=target,
    )
    matmul.hardware_aware_finetune(topk=20)
    assert get_codegen_result(matmul, target)


@pytest.mark.parametrize(
    "M,N,K,in_dtype,out_dtype,accum_dtype,with_bias,propagate_a,propagate_b,layout",
    [
        (1024, 1024, 1024, "float16", "float16", "float16", False, False, False, "nt"),
    ],
)
def test_matmul_profile_latency(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    with_bias,
    propagate_a,
    propagate_b,
    layout,
):
    matmul_config = MatmulConfig(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        with_bias=with_bias,
        propagate_a=propagate_a,
        propagate_b=propagate_b,
        layout=layout,
    )
    matmul = Matmul(
        config=matmul_config,
        target=target,
    )
    latency = matmul.profile_latency()
    assert latency


@pytest.mark.parametrize(
    "M,N,K,in_dtype,out_dtype,accum_dtype,with_bias,propagate_a,propagate_b,layout",
    [
        (256, 256, 256, "float16", "float16", "float16", False, False, False, "nt"),
        (256, 256, 256, "float16", "float16", "float16", False, False, True, "nt"),
    ],
)
def test_matmul_torch_forward(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    with_bias,
    propagate_a,
    propagate_b,
    layout,
):
    import torch

    matmul_config = MatmulConfig(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        with_bias=with_bias,
        propagate_a=propagate_a,
        propagate_b=propagate_b,
        layout=layout,
    )
    matmul = Matmul(
        config=matmul_config,
        target=target,
    )

    # convert tensors to torch
    input_shape = (M, K)
    weight_shape = (N, K) if layout == "nt" else (K, N)
    output_shape = (M, N)
    inputs = []
    inputs.append(torch.rand(input_shape, dtype=torch.float16).cuda())
    inputs.append(torch.rand(weight_shape, dtype=torch.float16).cuda())
    inputs.append(torch.rand(output_shape, dtype=torch.float16).cuda())
    ref_result = torch.matmul(inputs[0], inputs[1].t() if layout == "nt" else inputs[1])

    permuted_inputs = []
    if matmul.input_transform is not None:
        permuted_inputs.append(
            matmul.input_transform(inputs[0].cpu())
        ).cuda()
    else:
        permuted_inputs.append(inputs[0])
    if matmul.weight_transform is not None:
        permuted_inputs.append(
            matmul.weight_transform(inputs[1].cpu()).cuda()
        )
    else:
        permuted_inputs.append(inputs[1])
    permuted_inputs.append(inputs[2])
    matmul(*permuted_inputs)
    torch.testing.assert_close(permuted_inputs[-1], ref_result, rtol=1e-2, atol=1e-2)

# fmt: on

if __name__ == "__main__":
    bitblas.testing.main()
