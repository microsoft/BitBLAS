# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import bitblas
from bitblas.ops.quant_compress import QuantCompressConfig, QuantCompress
from bitblas import tvm
import bitblas.quantization

target = tvm.target.Target("llvm")


# fmt: off
def quant_compress_profile_latency(
    N,
    K,
    input_dtype,
    storage_dtype,
    dequantize_bits,
):

    quant_compress_config = QuantCompressConfig(
        M=N,
        N=K,
        input_dtype=input_dtype,
        storage_dtype=storage_dtype,
        dequantize_bits=dequantize_bits,
    )
    quant_compress = QuantCompress(
        config=quant_compress_config,
        target=target,
    )
    latency = quant_compress.profile_latency()
    assert latency


def test_quant_compress_profile_latency():
    quant_compress_profile_latency(1024, 1024, "int8", "int8", 4)
    quant_compress_profile_latency(1024, 1024, "int8", "int8", 2)
    quant_compress_profile_latency(1024, 1024, "int8", "int8", 1)

def quant_compress_correctness(
    N,
    K,
    input_dtype,
    storage_dtype,
    dequantize_bits,
):

    quant_compress_config = QuantCompressConfig(
        M=N,
        N=K,
        input_dtype=input_dtype,
        storage_dtype=storage_dtype,
        dequantize_bits=dequantize_bits,
    )
    quant_compress = QuantCompress(
        config=quant_compress_config,
        target=target,
    )
    import torch
    maxq = 2 ** (dequantize_bits) - 1
    weight = torch.randint(0, maxq, (N, K), dtype=torch.int8)
    qweight = quant_compress(weight)
    ref_qweight = bitblas.quantization.general_compress(weight.numpy(), dequantize_bits)
    torch.testing.assert_close(qweight.numpy(), ref_qweight, rtol=1e-3, atol=1e-3)

def test_quant_compress_correctness():
    quant_compress_correctness(1024, 1024, "int8", "int8", 4)
    quant_compress_correctness(1024, 1024, "int8", "int8", 2)
    quant_compress_correctness(1024, 1024, "int8", "int8", 1)

# fmt: on

if __name__ == "__main__":
    bitblas.testing.main()
