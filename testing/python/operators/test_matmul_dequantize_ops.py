# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pytest
import tvm
import bitblas
from bitblas.utils import get_target_from_env
from bitblas.ops.matmul_dequantize import (
    MatmulWeightOnlyDequantize,
    MatmulWeightOnlyDequantizeConfig,
)

target = tvm.target.Target(get_target_from_env())


def get_codegen_result(ops, target):
    code = ops.codegen(target=target)
    return code


# fmt: off
@pytest.mark.parametrize(
    "M,N,K,in_dtype,out_dtype,accum_dtype,bit,storage_dtype,source_format,with_scaling,group_size,fast_decoding,with_bias,propagate_a,propagate_b,layout",
    [
        (1, 1024, 1024, "float16", "float16", "float16", 4, "int8", "uint", False, -1, False, False, False, False, "nt"),
    ],
)
def test_matmul_dequantize_codegen_default(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    bit,
    storage_dtype,
    source_format,
    with_scaling,
    group_size,
    fast_decoding,
    with_bias,
    propagate_a,
    propagate_b,
    layout,
):

    matmul_config = MatmulWeightOnlyDequantizeConfig(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        bit=bit,
        storage_dtype=storage_dtype,
        source_format=source_format,
        with_scaling=with_scaling,
        group_size=group_size,
        fast_decoding=fast_decoding,
        with_bias=with_bias,
        propagate_a=propagate_a,
        propagate_b=propagate_b,
        layout=layout,
    )
    matmul = MatmulWeightOnlyDequantize(
        config=matmul_config,
        target=target,
    )
    assert get_codegen_result(matmul, target)


@pytest.mark.parametrize(
    "M,N,K,in_dtype,out_dtype,accum_dtype,bit,storage_dtype,source_format,with_scaling,group_size,fast_decoding,with_bias,propagate_a,propagate_b,layout",
    [
        (1, 1024, 1024, "float16", "float16", "float16", 4, "int8", "uint", False, -1, False, False, False, False, "nt",),
    ],
)
def test_matmul_dequantize_codegen_finetune(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    bit,
    storage_dtype,
    source_format,
    with_scaling,
    group_size,
    fast_decoding,
    with_bias,
    propagate_a,
    propagate_b,
    layout,
):

    matmul_config = MatmulWeightOnlyDequantizeConfig(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        bit=bit,
        storage_dtype=storage_dtype,
        source_format=source_format,
        with_scaling=with_scaling,
        group_size=group_size,
        fast_decoding=fast_decoding,
        with_bias=with_bias,
        propagate_a=propagate_a,
        propagate_b=propagate_b,
        layout=layout,
    )
    matmul = MatmulWeightOnlyDequantize(
        config=matmul_config,
        target=target,
    )
    matmul.hardware_aware_finetune(topk=20)
    assert get_codegen_result(matmul, target)


@pytest.mark.parametrize(
    "M,N,K,in_dtype,out_dtype,accum_dtype,bit,storage_dtype,source_format,with_scaling,group_size,fast_decoding,with_bias,propagate_a,propagate_b,layout",
    [
        (1, 1024, 1024, "float16", "float16", "float16", 4, "int8", "uint", False, -1, False, False, False, False, "nt",),
        (1, 1024, 1024, "float16", "float16", "float16", 4, "int8", "af", False, -1, False, False, False, False, "nt",),
        (1024, 1024, 1024, "float16", "float16", "float16", 4, "int8", "af", False, -1, False, False, False, False, "nt",),
    ],
)
def test_matmul_dequantize_profile_latency(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    bit,
    storage_dtype,
    source_format,
    with_scaling,
    group_size,
    fast_decoding,
    with_bias,
    propagate_a,
    propagate_b,
    layout,
):

    matmul_config = MatmulWeightOnlyDequantizeConfig(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        bit=bit,
        storage_dtype=storage_dtype,
        source_format=source_format,
        with_scaling=with_scaling,
        group_size=group_size,
        fast_decoding=fast_decoding,
        with_bias=with_bias,
        propagate_a=propagate_a,
        propagate_b=propagate_b,
        layout=layout,
    )
    matmul = MatmulWeightOnlyDequantize(
        config=matmul_config,
        target=target,
    )
    # prim_func = matmul.prim_func
    # print(prim_func)
    # func = prim_func
    # arch = matmul.arch
    # from bitblas.base.roller import DefaultPolicy, TensorCorePolicy
    # from bitblas.gpu.matmul_analysis import get_tensorized_func_and_tags

    # policy = DefaultPolicy(func=func, arch=arch)
    # try:
    #     tensorized_func, tags = get_tensorized_func_and_tags(func, arch.target)
    # except:
    #     tags = None
    # # Tune with Tensor Core if possible
    # if tags:
    #     policy = TensorCorePolicy(func=tensorized_func, arch=arch, tags=tags)

    # configs = policy.emit_config(20)
    # # sch = bitblas.gpu.gemv.GEMV().apply_config(prim_func, configs[0])
    # sch = bitblas.gpu.matmul_mma_dequantize.MatmulTensorizationMMAWithDequantizeInfo().apply_config(prim_func, configs[0])
    # with open("debug/matmul_dequantize.py", "w") as f:
    #     f.write(str(sch.mod))

    # with tvm.transform.PassContext(
    #     config={"tir.use_async_copy": True}
    # ):
    #     rt_mod = tvm.build(sch.mod["main"], target=matmul.arch.target)
    # print(sch.mod)

    # with open("debug/matmul_dequantize.cu", "w") as f:
    #     f.write(rt_mod.imported_modules[0].get_source())

    # time_evaluator = rt_mod.time_evaluator(
    #             rt_mod.entry_name, matmul.arch.device, number=3
    #     )
    # profile_tensors = matmul.get_profile_tensors()
    # print(time_evaluator(*profile_tensors))
    
    matmul.hardware_aware_finetune(topk=20)
    latency = matmul.profile_latency()
    print(matmul.codegen())
    print(latency)
    # assert latency


@pytest.mark.parametrize(
    "M,N,K,in_dtype,out_dtype,accum_dtype,bit,storage_dtype,source_format,with_scaling,group_size,fast_decoding,with_bias,propagate_a,propagate_b,layout",
    [
        (1, 1024, 1024, "float16", "float16", "float16", 4, "int8", "uint", False, -1, False, False, False, False, "nt",),
        (1, 1024, 1024, "float16", "float16", "float16", 4, "int8", "uint", False, -1, True, False, False, False, "nt",),
        (1, 1024, 1024, "float16", "float16", "float16", 4, "int8", "int", False, -1, False, False, False, False, "nt",),
        (1, 1024, 1024, "float16", "float16", "float16", 4, "int8", "int", False, -1, True, False, False, False, "nt",),
        (1, 1024, 1024, "float16", "float16", "float16", 2, "int8", "int", False, -1, True, False, False, False, "nt",),
        (1, 1024, 1024, "float16", "float16", "float16", 2, "int8", "int", True, -1, True, False, False, False, "nt",),
        (1, 1024, 1024, "float16", "float16", "float16", 2, "int8", "int", True, 128, True, False, False, False, "nt",),
        (1024, 1024, 1024, "float16", "float16", "float16", 2, "int8", "int", True, 128, False, False, False, False, "nt",),
        (1024, 1024, 1024, "float16", "float16", "float16", 2, "int8", "int", True, 128, False, False, False, True, "nt",),
        (1024, 1024, 1024, "float16", "float16", "float16", 2, "int8", "int", True, 128, False, False, True, True, "nt",),
    ],
)
def test_matmul_dequantize_torch_forward(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    bit,
    storage_dtype,
    source_format,
    with_scaling,
    group_size,
    fast_decoding,
    with_bias,
    propagate_a,
    propagate_b,
    layout,
):
    import torch
    import numpy as np
    from bitblas.quantization.utils import general_compress

    matmul_config = MatmulWeightOnlyDequantizeConfig(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        bit=bit,
        storage_dtype=storage_dtype,
        source_format=source_format,
        with_scaling=with_scaling,
        group_size=group_size,
        fast_decoding=fast_decoding,
        with_bias=with_bias,
        propagate_a=propagate_a,
        propagate_b=propagate_b,
        layout=layout,
    )
    matmul = MatmulWeightOnlyDequantize(
        config=matmul_config,
        target=target,
    )
    matmul.hardware_aware_finetune(topk=20)
    input_shape = (M, K)
    weight_shape = (N, K) if layout == "nt" else (K, N)
    output_shape = (M, N)
    inputs = []
    inputs.append(torch.rand(input_shape, dtype=torch.float16).cuda())
    maxq = 2 ** (bit - 1) - 1
    if source_format == "uint":
        inputs.append(torch.randint(0, maxq, weight_shape, dtype=torch.int8).cuda())
    elif source_format == "int":
        inputs.append(torch.randint(-maxq, maxq, weight_shape, dtype=torch.int8).cuda())
    else:
        raise NotImplementedError

    inputs.append(torch.rand(output_shape, dtype=torch.float16).cuda())
    ref_result = torch.matmul(inputs[0], (inputs[1].t() if layout == "nt" else inputs[1]).to(torch.float16))
    
    intweight = inputs[1]
    intweight = intweight.cpu().numpy().astype(np.int8)
    if source_format == "int":
        intweight = intweight + maxq
    # quantize to 4bit
    qw_np = general_compress(
        intweight, source_bits=bit, storage_dtype=np.int8
    )
    qw_torch = torch.from_numpy(qw_np).cuda()
    permuted_inputs = []
    permuted_inputs.append(inputs[0])
    if matmul.weight_transform is not None:
        permuted_inputs.append(
            matmul.weight_transform(qw_torch.cpu()).cuda()
        )
    else:
        permuted_inputs.append(qw_torch)
    if with_scaling:
        if group_size == -1:
            group_size = K
        permuted_inputs.append(torch.ones([N, K // group_size], dtype=torch.float16).cuda())
    permuted_inputs.append(inputs[2])
    matmul(*permuted_inputs)
    torch.testing.assert_close(permuted_inputs[-1], ref_result, rtol=1e-2, atol=1e-2)

# fmt: on

if __name__ == "__main__":
    # bitblas.testing.main()
    test_matmul_dequantize_profile_latency(
        16384,
        16384,
        16384,
        "float16",
        "float16",
        "float16",
        4,
        "int8",
        "af",
        False,
        -1,
        False,
        False,
        True,
        True,
        "nt",
    )
