# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tvm.tl.language as T

from tvm import DataType
from tvm.runtime import convert
from .utils import (
    mfma_store_index_map,)

lift = convert


class MatrixCoreIntrinEmitter(object):
    """
    To eliminate Python syntax within TIR Macro.
    """

    M_DIM = 16
    N_DIM = 16
    WARP_SIZE = 64
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
        reduce_k=1,
        num_elems_per_byte=1,
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
        self._initialize_mfma_prefix(self.k_dim)
        self._initialize_micro_size(self.M_DIM, self.N_DIM, self.k_dim)
        self.warp_rows = warp_row_tiles // self.micro_size_x
        self.warp_cols = warp_col_tiles // self.micro_size_y
        self.reduce_k = reduce_k
        self.threads = (self.WARP_SIZE * (block_row_warps * block_col_warps) * reduce_k)
        self.num_elems_per_byte = num_elems_per_byte

    def _initialize_k_dim(self, a_dtype="float16"):
        if isinstance(a_dtype, str):
            a_dtype = DataType(a_dtype)
        if a_dtype.bits == 32:
            self.k_dim = 4
        elif a_dtype.bits in [16, 8]:
            self.k_dim = 16
        else:
            raise ValueError(f"Unsupported a_dtype = {a_dtype}")

    def _initialize_local_size(self, m_dim=16, n_dim=16, k_dim=16, warp_size=32):
        self.local_size_a = (m_dim * k_dim) // warp_size
        self.local_size_b = (n_dim * k_dim) // warp_size
        self.local_size_out = (m_dim * n_dim) // warp_size

    def _initialize_abbrev(self, a_dtype, b_dtype, accum_dtype):
        self.a_dtype_abbrv = self.dtype_abbrv[a_dtype]
        self.b_dtype_abbrv = self.dtype_abbrv[b_dtype]
        self.accum_dtype_abbrv = self.dtype_abbrv[accum_dtype]

    def _initialize_mfma_prefix(self, k_dim=16):
        in_dtype, out_dtype = self.a_dtype, self.accum_dtype
        M_DIM, N_DIM = self.M_DIM, self.N_DIM
        out_dtype_abbrv = {
            "float16": "f16",
            "float32": "f32",
            "int8": "i8",
            "int32": "i32"
        }[out_dtype]

        in_dtype_abbrv = {
            "float16": "f16",
            "float32": "f32",
            "int8": "i8",
            "int32": "i32"
        }[in_dtype]

        self.mfma_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}{in_dtype_abbrv}"

    def _initialize_micro_size(self, m_dim=16, n_dim=16, k_dim=16):
        self.micro_size_x = m_dim
        self.micro_size_y = n_dim
        self.micro_size_k = k_dim

    def get_ldmatrix_index_map(self, is_b=False):
        from .mfma_layout import (
            shared_16x4_to_local_64x1_layout_A,
            shared_4x16_to_local_64x1_layout_B,
            shared_16x16_to_local_64x4_layout_A,
            shared_16x16_to_local_64x4_layout_B,
            thread_id_shared_access_64x1_to_16x4_layout_A,
            thread_id_shared_access_64x1_to_4x16_layout_B,
            thread_id_shared_access_64x4_to_16x16_layout_A,
            thread_id_shared_access_64x4_to_16x16_layout_B,
        )

        k_dim = self.k_dim
        transposed = self.a_transposed if not is_b else self.b_transposed
        if k_dim == 4:
            index_map = shared_16x4_to_local_64x1_layout_A
            reverse_index_map = thread_id_shared_access_64x1_to_16x4_layout_A
            if is_b:
                index_map = shared_16x4_to_local_64x1_layout_A if transposed else shared_4x16_to_local_64x1_layout_B
                reverse_index_map = thread_id_shared_access_64x1_to_16x4_layout_A if transposed else thread_id_shared_access_64x1_to_4x16_layout_B
        elif k_dim == 16:
            index_map = shared_16x16_to_local_64x4_layout_B if transposed else shared_16x16_to_local_64x4_layout_A
            reverse_index_map = thread_id_shared_access_64x4_to_16x16_layout_B if transposed else thread_id_shared_access_64x4_to_16x16_layout_A

            if is_b:
                index_map = shared_16x16_to_local_64x4_layout_A if transposed else shared_16x16_to_local_64x4_layout_B
                reverse_index_map = thread_id_shared_access_64x4_to_16x16_layout_A if transposed else thread_id_shared_access_64x4_to_16x16_layout_B
        else:
            raise ValueError("k_dim must be 4 or 16 currently")

        return index_map, reverse_index_map

    def ldmatrix_a(self, A_local_buf, A_shared_buf, ki, thread_bindings, rk=0):
        WARP_SIZE = self.WARP_SIZE
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps
        warp_row_tiles = self.warp_row_tiles
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        is_transposed = self.a_transposed

        _, reverse_index_map = self.get_ldmatrix_index_map(is_b=False)

        @T.macro
        def _warp_ldmatrix_a(
            A_local_buf,
            A_shared_buf,
            ki,
            thread_bindings,
            rk=0,
        ):
            tx = thread_bindings % WARP_SIZE
            tz = (thread_bindings // (WARP_SIZE * block_col_warps)) % block_row_warps
            if is_transposed:
                for i in T.serial(warp_cols):
                    for local_id in T.vectorized(local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (rk * chunk + ki * micro_size_k,
                                tz * warp_row_tiles + i * micro_size_x)
                        A_local_buf[i * local_size_a + local_id] = A_shared_buf[l + row, r + col]
            else:
                for i in T.serial(warp_cols):
                    for local_id in T.vectorized(local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (tz * warp_row_tiles + i * micro_size_x,
                                rk * chunk + ki * micro_size_k)
                        A_local_buf[i * local_size_a + local_id] = A_shared_buf[l + row, r + col]

        return _warp_ldmatrix_a(A_local_buf, A_shared_buf, ki, thread_bindings, rk)

    def ldmatrix_b(self, B_local_buf, B_shared_buf, ki, thread_bindings, rk=0):

        WARP_SIZE = self.WARP_SIZE
        block_col_warps = self.block_col_warps
        warp_col_tiles = self.warp_col_tiles
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        is_transposed = self.b_transposed

        _, reverse_index_map = self.get_ldmatrix_index_map(is_b=True)

        @T.macro
        def _warp_ldmatrix_b(
            B_local_buf,
            B_shared_buf,
            ki,
            thread_bindings,
            rk=0,
        ):
            tx = thread_bindings % WARP_SIZE
            ty = (thread_bindings // WARP_SIZE) % block_col_warps

            if is_transposed:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            ty * warp_col_tiles + j * micro_size_y,
                            rk * chunk + ki * micro_size_k,
                        )
                        B_local_buf[j * local_size_b + local_id] = B_shared_buf[l + row, r + col]
            else:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            rk * chunk + ki * micro_size_k,
                            ty * warp_col_tiles + j * micro_size_y,
                        )
                        B_local_buf[j * local_size_b + local_id] = B_shared_buf[l + row, r + col]

        return _warp_ldmatrix_b(B_local_buf, B_shared_buf, ki, thread_bindings, rk)

    def mfma(self, A_local_buf, B_local_buf, C_local_buf):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        mfma_suffix = self.mfma_suffix
        a_dtype, b_dtype, out_dtype = self.a_dtype, self.b_dtype, self.accum_dtype
        compute_a_dtype = a_dtype if local_size_a == 1 else f"{a_dtype}x{local_size_a}"
        compute_b_dtype = b_dtype if local_size_b == 1 else f"{b_dtype}x{local_size_b}"
        compute_out_dtype = out_dtype if local_size_out == 1 else f"{out_dtype}x{local_size_out}"

        @T.macro
        def _warp_mma(A_local_buf, B_local_buf, C_local_buf):
            for i, j in T.grid(warp_rows, warp_cols):
                T.tvm_mfma(
                    mfma_suffix,
                    "row",
                    "row",
                    compute_a_dtype,
                    compute_b_dtype,
                    compute_out_dtype,
                    A_local_buf.data,
                    (i * local_size_a) // local_size_a,
                    B_local_buf.data,
                    (j * local_size_b) // local_size_b,
                    C_local_buf.data,
                    (i * warp_cols * local_size_out + j * local_size_out) // local_size_out,
                    dtype=compute_out_dtype,
                )

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)

    def stmatrix(self, C_local_buf, C_buf, thread_bindings, pid_m=None, pid_n=None):
        WARP_SIZE = self.WARP_SIZE
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_out = self.local_size_out

        is_global = pid_m is not None and pid_n is not None
        BLOCK_M = block_row_warps * warp_rows
        BLOCK_N = block_col_warps * warp_cols
        M_DIM, N_DIM = self.M_DIM, self.N_DIM

        # STS
        # MMA Store must be in simulated instead of TVM Intrins
        # As TVM Intrins is like a hack that the threadIdx.x should be always
        # equal to the warp_size
        @T.macro
        def _warp_stmatrix_shared(C_local_buf, C_buf, thread_bindings):
            tx = thread_bindings % WARP_SIZE
            ty = (thread_bindings // WARP_SIZE) % block_row_warps
            tz = (thread_bindings // (WARP_SIZE * block_row_warps)) % block_col_warps

            for i, j in T.grid(warp_rows, warp_cols):
                for local_id in T.serial(local_size_out):
                    row, col = T.meta_var(mfma_store_index_map(tx, local_id))
                    C_buf[ty * warp_rows + i, tz * warp_cols + j, row,
                          col] = C_local_buf[i * warp_cols * local_size_out + j * local_size_out +
                                             local_id]

        @T.macro
        def _warp_stmatrix_global(C_local_buf, C_buf, thread_bindings):
            tx = thread_bindings % WARP_SIZE
            ty = (thread_bindings // WARP_SIZE) % block_row_warps
            tz = (thread_bindings // (WARP_SIZE * block_row_warps)) % block_col_warps

            for i, j in T.grid(warp_rows, warp_cols):
                for local_id in T.serial(local_size_out):
                    row, col = T.meta_var(mfma_store_index_map(tx, local_id))
                    C_buf[(pid_m * BLOCK_M + tz * warp_rows + i) * M_DIM + row,
                          (pid_n * BLOCK_N + ty * warp_cols + j) * N_DIM +
                          col] = C_local_buf[i * warp_cols * local_size_out + j * local_size_out +
                                             local_id]

        return _warp_stmatrix_global(C_local_buf, C_buf,
                                     thread_bindings) if is_global else _warp_stmatrix_shared(
                                         C_local_buf, C_buf, thread_bindings)
