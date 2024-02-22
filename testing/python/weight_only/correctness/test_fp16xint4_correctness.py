import torch

import bitblas
import numpy as np

from bitblas.quantization.utils import general_compress, interleave_weight
from bitblas.ops.matmul import MatmulWeightOnlyDequantize

M = 1
N = 4096
K = 1024
bitblas_matmul = MatmulWeightOnlyDequantize(
        M=M,
        N=N,
        K=K,
        in_dtype="float16",
        out_dtype="float16",
        accum_dtype="float16",
        propagate_b=False,
        bit=4,
        storage_dtype="uint8",
        source_format="int",
        with_scaling=False,
        group_size=128,
        fast_decoding=False,
        with_bias=False,
)

torch_arrs = []
torch_arrs.append(torch.randint(0, 10, (M, K), dtype=torch.float16, device="cuda"))
torch_arrs.append(torch.randint(0, 7, (N, K), dtype=torch.float16, device="cuda"))
torch_arrs.append(torch.zeros((M, K), dtype=torch.float16, device="cuda"))

print("torch: {}".format(torch_arrs[-1]))

