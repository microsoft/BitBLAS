# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas import tilelang as tilelang
import tilelang.language as T
from typing import Tuple, Optional
from tvm import DataType
from tvm.tir import PrimExpr
from tvm.runtime import convert

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

    # Not implemented yet
    def fill(self, C_local_buf, value: float = 0):
        m_dim = 16
        n_dim = 16
        k_dim = 16

        @T.macro
        def _wmma_fill(C_local_buf):

            T.evaluate(
                T.tvm_fill_fragment(
                    C_local_buf.data,
                    m_dim,
                    n_dim,
                    k_dim,
                    self.get_wmma_fragment_index(C_local_buf, 16, m_dim, n_dim),
                    T.float32(value),
                    dtype="handle",
                ))

        return _wmma_fill(C_local_buf)

    def load_matrix_sync_a(self, A_local_buf, A_shared_buf, ki, thread_bindings, rk=0):
        raise NotImplementedError

    def load_matrix_sync_b(self, B_local_buf, B_shared_buf, ki, thread_bindings, rk=0):
        raise NotImplementedError

    def sync(self, A_local_buf, B_local_buf, C_local_buf):
        raise NotImplementedError

    def stmatrix(self, C_local_buf, C_buf, thread_bindings, pid_m=None, pid_n=None):
        raise NotImplementedError
