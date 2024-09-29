# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm as tvm
from tvm import DataType
import tvm.tl.language as T
from typing import Optional
from bitblas.tl.utils import (
    get_mma_micro_size,
    make_swizzle_layout,
)

from bitblas.ops.base_scheduler import BaseScheduler

from dataclasses import dataclass


@dataclass
class MatmulFineGrainSIMTScheduler(BaseScheduler):
    # Fine-grained matrix multiplication scheduler
    # Allows for more detailed configuration.

    # Operation Configuration
    M: Optional[int] = None
    N: Optional[int] = None
    K: Optional[int] = None
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    trans_A: bool = False
    trans_B: bool = True
    accum_dtype: str = "float16"

    # Tensor Core Warp Configuration
    block_row_warps: int = 2
    block_col_warps: int = 2
    warp_row_tiles: int = 32
    warp_col_tiles: int = 32
    chunk: int = 32  # Usually determines the K-dimension split size

    # Tiling and Other Optimization Parameters
    num_stages: int = 2
    enable_rasterization: bool = False

    def with_default_config(self):
        raise NotImplementedError

    def apply_config(
        self,
    ):

        # M, N, K = self.M, self.N, self.K
        # trans_A, trans_B = self.trans_A, self.trans_B
        # in_dtype, out_dtype, accum_dtype = self.in_dtype, self.out_dtype, self.accum_dtype

        raise NotImplementedError


    def __post_init__(self):
        # Validate the matrix transpose settings
        assert self.trans_A is False, "Currently only support Matrix A not transposed"
        assert self.trans_B is True, "Currently only support Matrix B transposed"

        return
