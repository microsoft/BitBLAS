# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import bitblas
import bitblas.testing
import logging
from bitblas import set_log_level

set_log_level(logging.DEBUG)


def get_codegen_result(ops):
    code = ops.get_source()
    return code


def matmul_int4_torch_forward(M,
                              N,
                              K,
                              A_dtype,
                              W_dtype,
                              accum_dtype,
                              out_dtype,
                              layout,
                              propagate_b,
                              fast_decoding=False):
    import torch
    matmul_config = bitblas.MatmulConfig(
        M=M,  # M dimension
        N=N,  # N dimension
        K=K,  # K dimension
        A_dtype=A_dtype,  # activation A dtype
        W_dtype=W_dtype,  # weight W dtype
        accum_dtype=accum_dtype,  # accumulation dtype
        out_dtype=out_dtype,  # output dtype
        layout=layout,  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
        propagate_b=propagate_b,  # propagate B matrix
        fast_decoding=fast_decoding,
    )

    matmul = bitblas.Matmul(config=matmul_config, enable_tuning=False)

    # if finetuning is needed, uncomment the following line
    # matmul.hardware_aware_finetune(topk=20)

    print(matmul.get_source())
    storage_dtype = "int8"
    A = torch.randint(0, 4, (M, K), device="cuda", dtype=getattr(torch, storage_dtype))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, out_dtype))

    if W_dtype == "int4":
        B = torch.randint(0, 4, (N, K), device="cuda", dtype=getattr(torch, storage_dtype))
        compressed_A = (A[:, ::2] & 0x0F) + ((A[:, 1::2] & 0x0F) << 4)
        compressed_B = (B[:, ::2] & 0x0F) + ((B[:, 1::2] & 0x0F) << 4)
        if propagate_b:
            ladder_permutate_config = bitblas.ops.LadderPermutateConfig(
                M=N,
                N=(K // 2),
                datatype="int8",
                storage_dtype="int8",
                transform_kind=3,
                transpose_matrix=True,
            )

            ladder_permutate = bitblas.ops.LadderPermutate(ladder_permutate_config)
            LB = ladder_permutate(compressed_B.cpu()).cuda()
            matmul(compressed_A, LB, output=C)
        else:
            matmul(compressed_A, compressed_B, output=C)
    elif W_dtype == "int2":
        B = torch.randint(0, 2, (N, K), device="cuda", dtype=getattr(torch, storage_dtype))
        compressed_A = (A[:, ::2] & 0x0F) + ((A[:, 1::2] & 0x0F) << 4)
        compressed_B = (B[:, ::4] & 0x03) + ((B[:, 1::4] & 0x03) << 2) + (
            (B[:, 2::4] & 0x03) << 4) + ((B[:, 3::4] & 0x03) << 6)
        if propagate_b:
            compressed_B = (B[:, ::2] & 0x0F) + ((B[:, 1::2] & 0x0F) << 4)

            lop3_permutate_config = bitblas.ops.LOP3PermutateConfig(
                M=N,
                N=K,
                datatype="int4",
                dequantize_bits=2,
                storage_dtype="int8",
            )
            lop3_permutate = bitblas.ops.LOP3Permutate(
                config=lop3_permutate_config,
                target="llvm",
            )

            ladder_permutate_config = bitblas.ops.LadderPermutateConfig(
                M=N,
                N=(K // 2),
                datatype="int8",
                storage_dtype="int8",
                transform_kind=3,
                transpose_matrix=True,
            )

            ladder_permutate = bitblas.ops.LadderPermutate(ladder_permutate_config)

            compressed_B_ladder = ladder_permutate(compressed_B.cpu()).cuda()
            ladder_shape = compressed_B_ladder.shape
            int2_shape = (ladder_shape[:-1] + (ladder_shape[-1] // 2,))
            int2_tensor = torch.zeros(int2_shape, device="cuda", dtype=torch.int8)
            for i in range(int2_tensor.shape[-1]):
                int2_tensor[..., i] = (compressed_B_ladder[..., 2 * i] & 0x03) | (
                    (compressed_B_ladder[..., 2 * i] >> 4) & 0x03) << 2 | (
                        (compressed_B_ladder[..., 2 * i + 1] & 0x03) << 4) | (
                            (compressed_B_ladder[..., 2 * i + 1] >> 4) << 6)

            raw_tensor_shape = int2_tensor.shape
            print(f"{raw_tensor_shape=}")
            if fast_decoding:
                lop3_compressed_B = lop3_permutate(int2_tensor.cpu()).cuda()
                lop3_compressed_B = lop3_compressed_B.view(raw_tensor_shape)
            else:
                lop3_compressed_B = int2_tensor
            matmul(compressed_A, lop3_compressed_B, output=C)
        else:
            if fast_decoding:
                lop3_permutate_config = bitblas.ops.LOP3PermutateConfig(
                    M=N,
                    N=K,
                    datatype="int4",
                    dequantize_bits=2,
                    storage_dtype="int8",
                )
                lop3_permutate = bitblas.ops.LOP3Permutate(
                    config=lop3_permutate_config,
                    target="llvm",
                )
                lop3_compressed_B = lop3_permutate(compressed_B.cpu()).cuda()
                matmul(compressed_A, lop3_compressed_B, output=C)
            else:
                matmul(compressed_A, compressed_B, output=C)

    print(C)
    latency = matmul.profile_latency()
    print(latency)
    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, out_dtype))

    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


def test_matmul_torch_forward():
    matmul_int4_torch_forward(128, 128, 128, "int4", "int4", "int32", "int32", "nt", False)
    matmul_int4_torch_forward(128, 128, 128, "int4", "int4", "int32", "int32", "nt", True)
    matmul_int4_torch_forward(128, 128, 128, "int4", "int2", "int32", "int32", "nt", False, False)
    matmul_int4_torch_forward(128, 128, 128, "int4", "int2", "int32", "int32", "nt", False, True)
    matmul_int4_torch_forward(128, 128, 128, "int4", "int2", "int32", "int32", "nt", True, False)
    matmul_int4_torch_forward(128, 128, 128, "int4", "int2", "int32", "int32", "nt", True, True)


# fmt: on
if __name__ == "__main__":
    bitblas.testing.main()
