# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import tvm.tl.language as T

from tvm import DataType
from tvm.runtime import convert
from .utils import (
    mma_store_index_map,
    get_ldmatrix_offset,
)

lift = convert


class TensorCorePTXMacroGenerator(object):
    """
    To eliminate Python syntax within TIR Macro.
    """

    M_DIM = 16
    N_DIM = 16
    WARP_SIZE = 32
    dtype_abbrv = {
        "float16": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "int8": "int8",
        "int32": "int32",
        "e4m3_float8": "e4m3",
        "e5m2_float8": "e5m2",
    }

    def __init__(
        self,
        a_dtype="float16",
        b_dtype="float16",
        accum_dtype="float16",
        a_transposed=False,
        b_transposed=False,
    ):
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.accum_dtype = accum_dtype
        self.a_transposed = a_transposed
        self.b_transposed = b_transposed
        self._initialize_k_dim(a_dtype)
        self._initialize_abbrev(a_dtype, b_dtype, accum_dtype)
        self._initialize_local_size(
            self.M_DIM, self.N_DIM, self.k_dim, self.WARP_SIZE
        )
        self._initialize_mma_prefix(self.k_dim, b_transposed)
        self._initialize_micro_size(self.M_DIM, self.N_DIM, self.k_dim)

    def _initialize_k_dim(self, a_dtype="float16"):
        self.k_dim = 256 // DataType(a_dtype).bits

    def _initialize_local_size(
        self, m_dim=16, n_dim=16, k_dim=16, warp_size=32
    ):
        self.local_size_a = (m_dim * k_dim) // warp_size
        self.local_size_b = (n_dim * k_dim) // warp_size
        self.local_size_out = (m_dim * n_dim) // warp_size

    def _initialize_abbrev(self, a_dtype, b_dtype, accum_dtype):
        self.a_dtype_abbrv = self.dtype_abbrv[a_dtype]
        self.b_dtype_abbrv = self.dtype_abbrv[b_dtype]
        self.accum_dtype_abbrv = self.dtype_abbrv[accum_dtype]

    def _initialize_mma_prefix(self, k_dim=16, b_transposed=False):
        if k_dim == 16:
            self.mma_prefix = "m16n8k16"
        elif k_dim == 32 and b_transposed:
            self.mma_prefix = "m16n8k32"
        elif k_dim == 32 and not b_transposed:
            self.mma_prefix = "m16n8k32"
        else:
            assert False, f"Unsupported k_dim {k_dim}"

    def _initialize_micro_size(self, m_dim=16, n_dim=16, k_dim=16):
        self.micro_size_x = m_dim
        self.micro_size_y = n_dim
        self.micro_size_k = k_dim

    @staticmethod
    @T.macro
    def MMA(inst, A_local_buf, B_local_buf, C_local_buf, warp_rows, warp_cols):
        for i, j in T.grid(warp_rows, warp_cols):
            T.ptx_mma(
                inst.accum_dtype,
                "m16n8k16",
                "row",
                "col",
                inst.a_dtype_abbrv,
                inst.b_dtype_abbrv,
                inst.accum_dtype_abbrv,
                A_local_buf.data,
                i * inst.local_size_a,
                B_local_buf.data,
                j * inst.local_size_b,
                C_local_buf.data,
                i * warp_cols * inst.local_size_out + j * inst.local_size_out,
                T.bool(False),
            )

            T.ptx_mma(
                inst.accum_dtype,
                "m16n8k16",
                "row",
                "col",
                inst.a_dtype_abbrv,
                inst.b_dtype_abbrv,
                inst.accum_dtype_abbrv,
                A_local_buf.data,
                i * inst.local_size_a,
                B_local_buf.data,
                j * inst.local_size_b + lift(inst.local_size_b) // 2,
                C_local_buf.data,
                i * warp_cols * inst.local_size_out
                + j * inst.local_size_out
                + lift(inst.local_size_out) // 2,
                T.bool(False),
            )

    @staticmethod
    @T.macro
    def LDMATRIX_A(
        inst,
        A_local_buf,
        A_shared_buf,
        tx,
        ty,
        ki,
        warp_rows,
        warp_row_tiles,
        stride,
    ):
        for i in T.serial(warp_rows):
            T.ptx_ldmatrix(
                "float16",
                T.bool(False),
                4,
                ".b16",
                A_local_buf.data,
                i * inst.local_size_a,
                T.address_of(
                    A_shared_buf[
                        ty * warp_row_tiles + i * inst.micro_size_x,
                        ki * inst.micro_size_k,
                    ]
                ),
                get_ldmatrix_offset("A", tx, 0, stride, inst.a_dtype, False),
            )

    @staticmethod
    @T.macro
    def LDMATRIX_B(
        inst,
        B_local_buf,
        B_shared_buf,
        tx,
        tz,
        ki,
        warp_cols,
        warp_col_tiles,
        stride,
    ):
        for j in T.serial(warp_cols):
            T.ptx_ldmatrix(
                "float16",
                T.bool(False),  # TODO(lei): should be optimized
                4,
                ".b16",
                B_local_buf.data,
                j * inst.local_size_b,
                T.address_of(
                    B_shared_buf[
                        tz * warp_col_tiles + j * inst.micro_size_y,
                        ki * micro_size_k,
                    ]
                ),
                get_ldmatrix_offset("B", tx, 0, stride, inst.b_dtype, True),
            )

    # STS
    # MMA Store must be in simulated instead of TVM Intrins
    # As TVM Intrins is like a hack that the threadIdx.x should be always
    # equal to the warp_size
    @staticmethod
    @T.macro
    def STMATRIX(
        inst, C_local_buf, C_shared_buf, tx, ty, tz, warp_rows, warp_cols
    ):
        for i, j in T.grid(warp_rows, warp_cols):
            for local_id in T.serial(inst.local_size_out):
                row, col = T.meta_var(mma_store_index_map(tx, local_id))
                C_shared_buf[
                    ty * warp_rows + i, tz * warp_cols + j, row, col
                ] = C_local_buf[
                    i * (warp_cols * inst.local_size_out)
                    + j * inst.local_size_out
                    + local_id
                ]
