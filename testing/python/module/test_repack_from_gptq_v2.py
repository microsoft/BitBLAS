# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import bitblas
import torch

try:
    import gptqmodel  # noqa: F401
except ImportError as e:
    raise ImportError("Please install gptqmodel by running `pip install gptqmodel`") from e

from gptqmodel.nn_modules.qlinear.qlinear_exllama import ExllamaQuantLinear

torch.manual_seed(0)
bitblas.set_log_level("DEBUG")


def assert_output_with_gptq(m, in_features, out_features, group_size, sym=False):
    if group_size == -1:
        group_size = in_features
    _, linear, s, _ = bitblas.quantization.gen_quant4(in_features, out_features, group_size)

    zeros = torch.full((in_features // group_size, out_features), 8, dtype=torch.int32)

    old_linear = ExllamaQuantLinear(
        bits=4,
        group_size=group_size,
        infeatures=in_features,
        outfeatures=out_features,
        bias=False,
        sym=sym,
        desc_act=False,
    )
    old_linear._use_act_order = False
    old_linear.pack(linear, s.T, zeros.T, g_idx=None)
    old_linear = old_linear.to("cuda")
    old_linear.post_init()

    W_dtype = "uint4"
    with_zeros = True

    if sym:
        W_dtype = "int4"
        with_zeros = False

    bitblas_linear = bitblas.Linear(
        opt_M=m,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        A_dtype="float16",  # activation A dtype
        W_dtype=W_dtype,  # weight W dtype
        accum_dtype="float16",  # accumulation dtype
        out_dtype="float16",  # output dtype
        # configs for weight only quantization
        group_size=group_size,  # setting for grouped quantization
        with_scaling=True,  # setting for scaling factor
        with_zeros=with_zeros,  # setting for zeros
        zeros_mode="quantized",  # setting for how to calculating zeros
    )
    # Repack weights from CudaOldQuantLinear to BitBLAS linear module
    bitblas_linear.repack_from_gptq_v2(old_linear)

    # Prepare input data
    inp = torch.rand(m, in_features, dtype=torch.float16, device="cuda")

    # Move models to CUDA for execution
    old_linear = old_linear.to("cuda")
    bitblas_linear = bitblas_linear.to("cuda")

    # Perform inference without gradient calculations
    with torch.no_grad():
        res_cuda_old = old_linear(inp)
        res_bitblas = bitblas_linear(inp)

    # Verify the outputs are close within specified tolerances
    torch.testing.assert_close(res_bitblas, res_cuda_old, rtol=1e-0, atol=1e-1)


def test_assert_output_with_gptq():
    assert_output_with_gptq(1, 256, 256, 64, True)
    assert_output_with_gptq(1, 256, 256, -1, True)
    assert_output_with_gptq(1, 256, 256, 64, False)
    assert_output_with_gptq(1, 256, 256, -1, False)


if __name__ == "__main__":
    bitblas.testing.main()
