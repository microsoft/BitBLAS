# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tvm.tl.language as T

from typing import Union
from bitblas.ops.common import TransformKind
from tvm import DataType
from tvm.runtime import convert
from .utils import (
    mma_store_index_map,
    get_ldmatrix_offset,
)

lift = convert


class TensorCoreIntrinEmitter(object):
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
        self._initialize_local_size(
            self.M_DIM, self.N_DIM, self.k_dim, self.WARP_SIZE
        )
        self._initialize_mma_prefix(self.k_dim)
        self._initialize_micro_size(self.M_DIM, self.N_DIM, self.k_dim)
        self.warp_rows = warp_row_tiles // self.micro_size_x
        self.warp_cols = warp_col_tiles // self.micro_size_y
        self.reduce_k = reduce_k
        self.threads = (
            self.WARP_SIZE * (block_row_warps * block_col_warps) * reduce_k
        )
        self.num_elems_per_byte = num_elems_per_byte

    def _initialize_k_dim(self, a_dtype="float16"):
        if isinstance(a_dtype, str):
            a_dtype = DataType(a_dtype)
        self.k_dim = 256 // a_dtype.bits

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

    def ldmatrix_a(self, A_local_buf, A_shared_buf, ki, thread_bindings, rk=0):
        WARP_SIZE = self.WARP_SIZE
        block_row_warps = self.block_row_warps
        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        a_dtype = self.a_dtype
        a_transposed = self.a_transposed
        local_size_a = self.local_size_a

        @T.macro
        def _warp_ldmatrix_a(
            A_local_buf,
            A_shared_buf,
            ki,
            thread_bindings,
            rk=0,
        ):
            stride = A_shared_buf.shape[-1]
            tx = thread_bindings % WARP_SIZE
            ty = (thread_bindings // WARP_SIZE) % block_row_warps

            for i in T.serial(warp_rows):
                T.ptx_ldmatrix(
                    a_dtype,
                    T.bool(False),
                    4,
                    ".b16",
                    A_local_buf.data,
                    i * local_size_a,
                    T.address_of(
                        A_shared_buf[
                            ty * warp_row_tiles + i * micro_size_x,
                            rk * chunk + ki * micro_size_k,
                        ]
                    ),
                    get_ldmatrix_offset(
                        "A", tx, 0, stride, a_dtype, a_transposed
                    ),
                )

        return _warp_ldmatrix_a(
            A_local_buf, A_shared_buf, ki, thread_bindings, rk
        )

    def ldmatrix_b(self, B_local_buf, B_shared_buf, ki, thread_bindings, rk=0):

        WARP_SIZE = self.WARP_SIZE
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps
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
            tx = thread_bindings % WARP_SIZE
            tz = (
                thread_bindings // (WARP_SIZE * block_row_warps)
            ) % block_col_warps

            for j in T.serial(warp_cols):
                # Assign B_shared_elem
                ri, rj = (
                    tz * warp_col_tiles + j * micro_size_y,
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
                    get_ldmatrix_offset(
                        "B", tx, 0, stride, b_dtype, b_transposed
                    ),
                )

        return _warp_ldmatrix_b(
            B_local_buf, B_shared_buf, ki, thread_bindings, rk
        )

    def mma(self, A_local_buf, B_local_buf, C_local_buf):
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
                    i * warp_cols * local_size_out
                    + j * local_size_out
                    + lift(local_size_out) // 2,
                    T.bool(False),
                )

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)

    def stmatrix(self, C_local_buf, C_shared_buf, thread_bindings):
        WARP_SIZE = self.WARP_SIZE
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_out = self.local_size_out

        # STS
        # MMA Store must be in simulated instead of TVM Intrins
        # As TVM Intrins is like a hack that the threadIdx.x should be always
        # equal to the warp_size
        @T.macro
        def _warp_stmatrix(C_local_buf, C_shared_buf, thread_bindings):
            tx = thread_bindings % WARP_SIZE
            ty = (thread_bindings // WARP_SIZE) % block_row_warps
            tz = (
                thread_bindings // (WARP_SIZE * block_row_warps)
            ) % block_col_warps
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id_o in T.serial(local_size_out // 2):
                    for local_id_i in T.vectorized(2):
                        local_id = local_id_o * 2 + local_id_i
                        row, col = T.meta_var(mma_store_index_map(tx, local_id))
                        C_shared_buf[
                            ty * warp_rows + i, tz * warp_cols + j, row, col
                        ] = C_local_buf[
                            i * (warp_cols * local_size_out)
                            + j * local_size_out
                            + local_id
                        ]

        return _warp_stmatrix(C_local_buf, C_shared_buf, thread_bindings)


class TensorCoreIntrinEmitterWithLadderTransform(TensorCoreIntrinEmitter):
    """
    To eliminate Python syntax within TIR Macro.
    With Ladder Transform Plugin.
    """

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
        transform_kind_a: Union[int, TransformKind] = 0,
        transform_kind_b: Union[int, TransformKind] = 0,
        num_elems_per_byte=1,
    ):
        super().__init__(
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            accum_dtype=accum_dtype,
            a_transposed=a_transposed,
            b_transposed=b_transposed,
            block_row_warps=block_row_warps,
            block_col_warps=block_col_warps,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=chunk,
            reduce_k=reduce_k,
            num_elems_per_byte=num_elems_per_byte,
        )
        self._initialize_transform_kind(transform_kind_a, transform_kind_b)

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

    def _initialize_transform_kind(self, transform_kind_a, transform_kind_b):
        if isinstance(transform_kind_a, int):
            self.transform_kind_a = TransformKind(transform_kind_a)
        elif isinstance(transform_kind_a, TransformKind):
            self.transform_kind_a = transform_kind_a
        else:
            raise ValueError("Unsupported transform_kind_a")

        if isinstance(transform_kind_b, int):
            self.transform_kind_b = TransformKind(transform_kind_b)
        elif isinstance(transform_kind_b, TransformKind):
            self.transform_kind_b = transform_kind_b
        else:
            raise ValueError("Unsupported transform_kind_b")

        if self.transform_kind_a != TransformKind.NonTransform:
            raise ValueError("TransformKind A is not supported yet")

        assert transform_kind_b in [0, 3], "Currently only support 0 and 3"

    def ldmatrix_b(self, B_local_buf, B_shared_buf, ki, thread_bindings, rk=0):
        @T.macro
        def _warp_ldmatrix_b(
            inst,
            B_local_buf,
            B_shared_buf,
            ki,
            thread_bindings,
            rk=0,
        ):
            WARP_SIZE = inst.WARP_SIZE
            block_row_warps = inst.block_row_warps
            block_col_warps = inst.block_col_warps
            warp_col_tiles = inst.warp_col_tiles
            warp_cols = inst.warp_cols
            chunk = inst.chunk
            micro_size_y = inst.micro_size_y
            micro_size_k = inst.micro_size_k
            local_size_b = inst.local_size_b
            b_dtype = inst.b_dtype
            transform_kind_b = inst.transform_kind_b
            b_transposed = inst.b_transposed
            num_elems_per_byte = inst.num_elems_per_byte

            stride = B_shared_buf.shape[-1]
            tx = thread_bindings % WARP_SIZE
            tz = (
                thread_bindings // (WARP_SIZE * block_row_warps)
            ) % block_col_warps

            if transform_kind_b < TransformKind.LDMatrixTransform:
                for j in T.serial(warp_cols):
                    # Assign B_shared_elem
                    ri, rj = (
                        tz * warp_col_tiles + j * micro_size_y,
                        rk * chunk + ki * micro_size_k,
                    )
                    ni, nj, nii, njj = (
                        (ri) // micro_size_y,
                        (rj) // micro_size_k,
                        (ri) % micro_size_y,
                        (rj) % micro_size_k,
                    )
                    args = (
                        (ni, nj, nii, njj) if transform_kind_b > 0 else (ri, rj)
                    )
                    B_shared_elem = B_shared_buf[args]

                    T.ptx_ldmatrix(
                        b_dtype,
                        T.bool(False),  # TODO(lei): should be optimized
                        4,
                        ".b16",
                        B_local_buf.data,
                        j * local_size_b,
                        T.address_of(B_shared_elem),
                        get_ldmatrix_offset(
                            "B", tx, 0, stride, b_dtype, b_transposed
                        ),
                    )
            else:
                local_size_dequantize = local_size_b // num_elems_per_byte
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(local_size_dequantize):
                        # Assign B_shared_elem
                        ri, rj = (
                            tz * warp_cols + j,
                            rk * (chunk // micro_size_k) + ki,
                        )
                        rii, rjj = (tx * local_size_dequantize + local_id) // (
                            micro_size_k // num_elems_per_byte
                        ), (tx * local_size_dequantize + local_id) % (
                            micro_size_k // num_elems_per_byte
                        )
                        B_local_buf[j * local_size_dequantize + local_id] = (
                            B_shared_buf[ri, rj, rii, rjj]
                        )

        return _warp_ldmatrix_b(
            B_local_buf, B_shared_buf, ki, thread_bindings, rk
        )

    def mma(self, A_local_buf, B_local_buf, C_local_buf):
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
                    i * warp_cols * local_size_out
                    + j * local_size_out
                    + lift(local_size_out) // 2,
                    T.bool(False),
                )

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)
