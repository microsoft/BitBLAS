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
        block_row_warps=2,
        block_col_warps=2,
        warp_row_tiles=8,
        warp_col_tiles=8,
        chunk=16,
        threads=128,
    ):
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.accum_dtype = accum_dtype
        self.a_transposed = a_transposed
        self.b_transposed = b_transposed
        # Hint Information
        self.block_row_warps = block_row_warps
        self.block_col_warps = block_col_warps
        self.warp_row_tiles = warp_row_tiles
        self.warp_col_tiles = warp_col_tiles
        self.chunk = chunk
        self._initialize_k_dim(a_dtype)
        self._initialize_abbrev(a_dtype, b_dtype, accum_dtype)
        self._initialize_local_size(self.M_DIM, self.N_DIM, self.k_dim, self.WARP_SIZE)
        self._initialize_mma_prefix(self.k_dim)
        self._initialize_micro_size(self.M_DIM, self.N_DIM, self.k_dim)
        self.warp_rows = warp_row_tiles // self.micro_size_x
        self.warp_cols = warp_col_tiles // self.micro_size_y
        self.threads = threads

    def _initialize_k_dim(self, a_dtype="float16"):
        self.k_dim = 256 // DataType(a_dtype).bits

    def _initialize_local_size(self, m_dim=16, n_dim=16, k_dim=16, warp_size=32):
        self.local_size_a = (m_dim * k_dim) // warp_size
        self.local_size_b = (n_dim * k_dim) // warp_size
        self.local_size_out = (m_dim * n_dim) // warp_size

    def _initialize_abbrev(self, a_dtype, b_dtype, accum_dtype):
        self.a_dtype_abbrv = self.dtype_abbrv[a_dtype]
        self.b_dtype_abbrv = self.dtype_abbrv[b_dtype]
        self.accum_dtype_abbrv = self.dtype_abbrv[accum_dtype]

    def _initialize_mma_prefix(self, k_dim=16):
        if k_dim == 16:
            self.mma_prefix = "m16n8k16"
        elif k_dim == 32:
            self.mma_prefix = "m16n8k32"
        else:
            raise ValueError("Unsupported k_dim")

    def _initialize_micro_size(self, m_dim=16, n_dim=16, k_dim=16):
        self.micro_size_x = m_dim
        self.micro_size_y = n_dim
        self.micro_size_k = k_dim

    @staticmethod
    @T.macro
    def MMA(inst, A_local_buf, B_local_buf, C_local_buf):
        for i, j in T.grid(inst.warp_rows, inst.warp_cols):
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
                i * inst.warp_cols * inst.local_size_out + j * inst.local_size_out,
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
                i * inst.warp_cols * inst.local_size_out + j * inst.local_size_out +
                lift(inst.local_size_out) // 2,
                T.bool(False),
            )

    @staticmethod
    @T.macro
    def LDMATRIX_A(
        inst,
        A_local_buf,
        A_shared_buf,
        ki,
        thread_bindings,
    ):
        stride = inst.chunk
        tx = thread_bindings % inst.WARP_SIZE
        ty = (thread_bindings // inst.WARP_SIZE) % inst.block_row_warps
        # self.ty = (thread_bindings // warp_size) % block_row_warps
        # self.tz = thread_bindings // (warp_size * block_row_warps)
        for i in T.serial(inst.warp_rows):
            T.ptx_ldmatrix(
                "float16",
                T.bool(False),
                4,
                ".b16",
                A_local_buf.data,
                i * inst.local_size_a,
                T.address_of(A_shared_buf[ty * inst.warp_row_tiles + i * inst.micro_size_x,
                                          ki * inst.micro_size_k,]),
                get_ldmatrix_offset("A", tx, 0, stride, inst.a_dtype, False),
            )

    @staticmethod
    @T.macro
    def LDMATRIX_B(
        inst,
        B_local_buf,
        B_shared_buf,
        ki,
        thread_bindings,
    ):
        stride = inst.chunk
        tx = thread_bindings % inst.WARP_SIZE
        tz = thread_bindings // (inst.WARP_SIZE * inst.block_row_warps)
        for j in T.serial(inst.warp_cols):
            T.ptx_ldmatrix(
                "float16",
                T.bool(False),  # TODO(lei): should be optimized
                4,
                ".b16",
                B_local_buf.data,
                j * inst.local_size_b,
                T.address_of(B_shared_buf[tz * inst.warp_col_tiles + j * inst.micro_size_y,
                                          ki * inst.micro_size_k,]),
                get_ldmatrix_offset("B", tx, 0, stride, inst.b_dtype, True),
            )

    # STS
    # MMA Store must be in simulated instead of TVM Intrins
    # As TVM Intrins is like a hack that the threadIdx.x should be always
    # equal to the warp_size
    @staticmethod
    @T.macro
    def STMATRIX(inst, C_local_buf, C_shared_buf, thread_bindings):
        tx = thread_bindings % inst.WARP_SIZE
        ty = (thread_bindings // inst.WARP_SIZE) % inst.block_row_warps
        tz = thread_bindings // (inst.WARP_SIZE * inst.block_row_warps)
        for i, j in T.grid(inst.warp_rows, inst.warp_cols):
            for local_id in T.serial(inst.local_size_out):
                row, col = T.meta_var(mma_store_index_map(tx, local_id))
                C_shared_buf[ty * inst.warp_rows + i, tz * inst.warp_cols + j, row,
                             col] = C_local_buf[i * (inst.warp_cols * inst.local_size_out) +
                                                j * inst.local_size_out + local_id]

    # Allow GEMM from shared memory to local memory
    @staticmethod
    @T.macro
    def GEMM_SS(inst, A_shared_buf, B_shared_buf, C_local_buf, thread_bindings):
        A_local_buf = T.alloc_fragment((inst.warp_rows * inst.local_size),
                                       inst.a_dtype,
                                       scope="local")
        B_local_buf = T.alloc_fragment((inst.warp_cols * inst.local_size),
                                       inst.b_dtype,
                                       scope="local")
        for ki in T.serial(0, (inst.block_K // inst.micro_size_k)):
            inst.LDMATRIX_A(
                inst,
                A_local_buf,
                A_shared_buf,
                ki,
                thread_bindings=thread_bindings,
            )

            inst.LDMATRIX_B(
                inst,
                B_local_buf,
                B_shared_buf,
                ki,
                thread_bindings=thread_bindings,
            )

            inst.MMA(inst, A_local_buf, B_local_buf, C_local_buf)
