# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tvm.tl.language as T
from typing import Union, Tuple, Optional
from bitblas.base.operator_common import TransformKind
from tvm import DataType
from tvm.tir import PrimExpr
from tvm.runtime import convert
from .utils import (
    mma_store_index_map,
    get_ldmatrix_offset,
)

lift = convert


class WMMAIntrinEmitter(object):
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

    # Represent the thread binding in the form of (tx, warp_n, warp_m)
    is_m_first = False

    def __init__(
        self,
        a_dtype: str = "float16",
        b_dtype: str = "float16",
        accum_dtype: str = "float16",
        a_transposed: bool = False,
        b_transposed: bool = False,
        block_row_warps: int = 2,
        block_col_warps: int = 2,
        warp_row_tiles: int = 8,
        warp_col_tiles: int = 8,
        chunk: int = 16,
        reduce_k: int = 1,
        num_elems_per_byte: int = 1,
        is_m_first: Optional[bool] = False,
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
        self._initialize_is_m_first(is_m_first)

        self.warp_rows = warp_row_tiles // self.micro_size_x
        self.warp_cols = warp_col_tiles // self.micro_size_y
        self.reduce_k = reduce_k
        self.threads = self.WARP_SIZE * (block_row_warps * block_col_warps) * reduce_k
        self.num_elems_per_byte = num_elems_per_byte

    def _initialize_k_dim(self, a_dtype="float16"):
        if isinstance(a_dtype, str):
            a_dtype = DataType(a_dtype)
        self.k_dim = 256 // a_dtype.bits

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

    def _initialize_is_m_first(self, is_m_first: Optional[bool] = False):
        if is_m_first is not None:
            self.is_m_first = is_m_first

    def extract_thread_binding(self,
                               thread_id,
                               is_m_first=None) -> Tuple[PrimExpr, PrimExpr, PrimExpr]:
        """
        is_m_first: True if the thread binding is in the form of (tx, warp_n, warp_m)
        which represents [warp_size, block_row_warps (split n), block_col_warps (split m)]
        Otherwise, it is in the form of [warp_size, block_col_warps (split m), block_row_warps (split n)]
        """
        WARP_SIZE = self.WARP_SIZE
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps

        # if is_m_first is None, then use the default value
        if is_m_first is None:
            is_m_first = self.is_m_first

        if is_m_first:
            lane_id, warp_n, warp_m = (
                thread_id % WARP_SIZE,
                (thread_id // WARP_SIZE) % block_col_warps,
                (thread_id // (WARP_SIZE * block_col_warps)) % block_row_warps,
            )
            return lane_id, warp_n, warp_m
        else:
            lane_id, warp_m, warp_n = (
                thread_id % WARP_SIZE,
                (thread_id // WARP_SIZE) % block_row_warps,
                (thread_id // (WARP_SIZE * block_row_warps)) % block_col_warps,
            )
            return lane_id, warp_n, warp_m

    ######## WMMA intrinsics ########
    def get_wmma_fragment_index(self, buffer, stride, m_dim, n_dim):
        """Compute wmma fragment index using elem_offset of the buffer"""
        frag_index_m = buffer.elem_offset // stride // m_dim
        frag_index_n = buffer.elem_offset % stride // n_dim

        num_fragments_per_row = stride // n_dim
        return frag_index_m * num_fragments_per_row + frag_index_n

    def fill(self, C_local_buf, value:float=0):
        m_dim = 16
        n_dim = 16
        k_dim = 16
        @T.macro
        def _wmma_fill(C_local_buf):
            block_row_warps = self.block_row_warps
            block_col_warps = self.block_col_warps
            warp_rows = self.warp_rows
            warp_cols = self.warp_cols
            local_size_out = self.local_size_out

            T.evaluate(
                T.tvm_fill_fragment(
                    C_local_buf.data,
                    m_dim,
                    n_dim,
                    k_dim,
                    self.get_wmma_fragment_index(
                        C_local_buf, 16, m_dim, n_dim
                    ),
                    T.float32(value),
                    dtype="handle",
                )
            )

        return _wmma_fill(C_local_buf)

    def load_matrix_sync_a(self, A_local_buf, A_shared_buf, ki, thread_bindings, rk=0):
        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        a_dtype = self.a_dtype
        a_transposed = self.a_transposed

        @T.macro
        def _warp_ldmatrix_a(
            A_local_buf,
            A_shared_buf,
            ki,
            thread_bindings,
            rk=0,
        ):
            stride = A_shared_buf.shape[-1]
            tx, _, warp_m = self.extract_thread_binding(thread_bindings)
            for i in T.serial(warp_rows):
                T.ptx_ldmatrix(
                    a_dtype,
                    T.bool(False),
                    4,
                    ".b16",
                    A_local_buf.data,
                    i * local_size_a,
                    T.address_of(A_shared_buf[
                        warp_m * warp_row_tiles + i * micro_size_x,
                        rk * chunk + ki * micro_size_k,
                    ]),
                    get_ldmatrix_offset("A", tx, 0, stride, a_dtype, a_transposed),
                )

        return _warp_ldmatrix_a(A_local_buf, A_shared_buf, ki, thread_bindings, rk)

    def load_matrix_sync_b(self, B_local_buf, B_shared_buf, ki, thread_bindings, rk=0):
        warp_col_tiles = self.warp_col_tiles
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        b_dtype = self.b_dtype
        b_transposed = self.b_transposed

        @T.macro
        def _warp_ldmatrix_b(
            B_local_buf,
            B_shared_buf,
            ki,
            thread_bindings,
            rk=0,
        ):
            stride = B_shared_buf.shape[-1]
            tx, warp_n, _ = self.extract_thread_binding(thread_bindings)

            for j in T.serial(warp_cols):
                # Assign B_shared_elem
                ri, rj = (
                    warp_n * warp_col_tiles + j * micro_size_y,
                    rk * chunk + ki * micro_size_k,
                )
                B_shared_elem = B_shared_buf[ri, rj]

                T.ptx_ldmatrix(
                    b_dtype,
                    T.bool(False),  # TODO(lei): should be optimized
                    4,
                    ".b16",
                    B_local_buf.data,
                    j * local_size_b,
                    T.address_of(B_shared_elem),
                    get_ldmatrix_offset("B", tx, 0, stride, b_dtype, b_transposed),
                )

        return _warp_ldmatrix_b(B_local_buf, B_shared_buf, ki, thread_bindings, rk)

    def sync(self, A_local_buf, B_local_buf, C_local_buf):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        accum_dtype = self.accum_dtype
        accum_dtype_abbrv = self.accum_dtype_abbrv
        mma_prefix = self.mma_prefix

        @T.macro
        def _warp_mma(A_local_buf, B_local_buf, C_local_buf):
            for i, j in T.grid(warp_rows, warp_cols):
                T.ptx_mma(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    accum_dtype_abbrv,
                    A_local_buf.data,
                    i * local_size_a,
                    B_local_buf.data,
                    j * local_size_b,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + j * local_size_out,
                    T.bool(False),
                )

                T.ptx_mma(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    accum_dtype_abbrv,
                    A_local_buf.data,
                    i * local_size_a,
                    B_local_buf.data,
                    j * local_size_b + lift(local_size_b) // 2,
                    C_local_buf.data,
                    i * warp_cols * local_size_out + j * local_size_out + lift(local_size_out) // 2,
                    T.bool(False),
                )

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)

    def stmatrix(self, C_local_buf, C_buf, thread_bindings, pid_m=None, pid_n=None):
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
            tx, warp_n, warp_m = self.extract_thread_binding(thread_bindings)
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id_o in T.serial(local_size_out // 2):
                    for local_id_i in T.vectorized(2):
                        local_id = local_id_o * 2 + local_id_i
                        row, col = T.meta_var(mma_store_index_map(tx, local_id))
                        C_buf[warp_m * warp_rows + i, warp_n * warp_cols + j, row,
                              col] = C_local_buf[i * (warp_cols * local_size_out) +
                                                 j * local_size_out + local_id]

        @T.macro
        def _warp_stmatrix_global(C_local_buf, C_buf, thread_bindings):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_bindings)
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id_o in T.serial(local_size_out // 2):
                    for local_id_i in T.vectorized(2):
                        local_id = local_id_o * 2 + local_id_i
                        row, col = T.meta_var(mma_store_index_map(tx, local_id))
                        C_buf[
                            (pid_m * BLOCK_M + warp_m * warp_rows + i) * M_DIM + row,
                            (pid_n * BLOCK_N + warp_n * warp_cols + j) * N_DIM + col,
                        ] = C_local_buf[i * warp_cols * local_size_out + j * local_size_out +
                                        local_id]

        return (_warp_stmatrix_global(C_local_buf, C_buf, thread_bindings)
                if is_global else _warp_stmatrix_shared(C_local_buf, C_buf, thread_bindings))
