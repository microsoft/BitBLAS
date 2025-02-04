# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm as tvm
from bitblas import tilelang as tilelang
from tvm.runtime import convert
from tvm.tir.expr import Cast, IntImm
from tvm.tir.function import TensorIntrin
from tvm.script import tir as T
from tvm._ffi import register_func
from typing import Dict, Tuple, Optional, List
from typing_extensions import Literal
from tvm.tir.function import PrimFunc

lift = convert

WARP_SIZE = 64
M_DIM = 16
N_DIM = 16


def shared_16x4_to_local_64x1_layout_A(i, j):
    thread_id = (j * 16 + i)
    return thread_id, convert(0)


def thread_id_shared_access_64x1_to_16x4_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = thread_id // 16
    return i, j


def shared_4x16_to_local_64x1_layout_B(i, j):
    thread_id = (i * 16 + j)
    return thread_id, convert(0)


def thread_id_shared_access_64x1_to_4x16_layout_B(thread_id, local_id):
    i = thread_id // 16
    j = thread_id % 16
    return i, j


def shared_16x16_to_local_64x4_layout_C(i, j):
    thread_id = j + (i // 4) * 16
    local = (i % 4)
    return thread_id, local


@register_func("tir.index_map.shared_16x16_to_ldmatrix_64x4_layout")
def shared_16x16_to_ldmatrix_64x4_layout(ind):
    i, j = ind[0], ind[1]
    thread_id, local_id = shared_16x16_to_local_64x4_layout_C(i, j)
    return convert([thread_id, local_id])


def thread_id_shared_access_64x4_to_16x16_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = (thread_id // 16) * 4 + local_id
    return i, j


def shared_16x16_to_local_64x4_layout_A(i, j):
    thread_id = i + 16 * (j // 4)
    local = (j % 4)
    return thread_id, local


def thread_id_shared_access_64x4_to_16x16_layout_B(thread_id, local_id):
    i = local_id + (thread_id // 16) * 4
    j = thread_id % 16
    return i, j


def shared_16x16_to_local_64x4_layout_B(i, j):
    thread_id = j + (i // 4) * 16
    local = (i % 4)
    return thread_id, local


def thread_id_shared_access_64x4_to_16x16_layout_C(thread_id, local_id):
    i = local_id + (thread_id // 16) * 4
    j = thread_id % 16
    return i, j


def get_mma_fill_intrin(dtype, local_size):
    zero = IntImm("int32", 0).astype(dtype)

    # Assume M = N = 16
    index_map = shared_16x16_to_local_64x4_layout_C

    @T.prim_func
    def mma_fill_desc(a: T.handle) -> None:
        C_warp = T.match_buffer(a, [WARP_SIZE, local_size], dtype=dtype, scope="warp")

        with T.block("root"):
            T.reads()
            T.writes(C_warp[0:WARP_SIZE, 0:local_size])
            for i0, i1 in T.grid(M_DIM, N_DIM):
                with T.block("C_warp"):
                    i, j = T.axis.remap("SS", [i0, i1])
                    thread_id, local_id = T.meta_var(index_map(i, j))
                    T.reads()
                    T.writes(C_warp[thread_id, local_id])
                    C_warp[thread_id, local_id] = zero

    @T.prim_func
    def mma_fill_impl(a: T.handle) -> None:
        C_warp = T.match_buffer(
            a, [WARP_SIZE, local_size], dtype=dtype, scope="warp", offset_factor=1)

        with T.block("root"):
            T.reads()
            T.writes(C_warp[0:WARP_SIZE, 0:local_size])
            for tx in T.thread_binding(WARP_SIZE, "threadIdx.x"):
                for local_id in T.serial(0, local_size):
                    C_warp[tx, local_id] = zero

    return mma_fill_desc, mma_fill_impl


def get_mfma_load_intrin(
    k_dim=4,
    dtype="float32",
    scope="shared",
    is_b=False,
    transposed=False,
):

    local_size = (M_DIM * k_dim) // WARP_SIZE if not is_b else (N_DIM * k_dim) // WARP_SIZE
    memory_shape = (M_DIM, k_dim)
    if is_b:
        memory_shape = (N_DIM, k_dim) if transposed else (k_dim, N_DIM)

    row_dim, col_dim = memory_shape

    if k_dim == 4:
        index_map = shared_16x4_to_local_64x1_layout_A
        reverse_index_map = thread_id_shared_access_64x1_to_16x4_layout_A
        if is_b:
            index_map = shared_16x4_to_local_64x1_layout_A if transposed else shared_4x16_to_local_64x1_layout_B
            reverse_index_map = thread_id_shared_access_64x1_to_16x4_layout_A if transposed else thread_id_shared_access_64x1_to_4x16_layout_B
    elif k_dim == 16:
        index_map = shared_16x16_to_local_64x4_layout_A
        reverse_index_map = thread_id_shared_access_64x4_to_16x16_layout_A

        if is_b:
            index_map = shared_16x16_to_local_64x4_layout_A if transposed else shared_16x16_to_local_64x4_layout_B
            reverse_index_map = thread_id_shared_access_64x4_to_16x16_layout_A if transposed else thread_id_shared_access_64x4_to_16x16_layout_B
    else:
        raise ValueError("k_dim must be 4 or 16 currently")

    @T.prim_func
    def mfma_load_desc(reg_handle: T.handle, memory_handle: T.handle) -> None:
        memory = T.match_buffer(
            memory_handle,
            memory_shape,
            dtype,
            offset_factor=1,
            scope=scope,
        )
        reg = T.match_buffer(
            reg_handle, (WARP_SIZE, local_size), dtype, offset_factor=1, scope="warp")

        with T.block("root"):
            T.reads(memory[0:row_dim, 0:col_dim])
            T.writes(reg[0:WARP_SIZE, 0:local_size])

            for ax0, ax1 in T.grid(row_dim, col_dim):
                with T.block("memory_reg"):
                    v0, v1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(memory[v0, v1])

                    thread_id, local_id = T.meta_var(index_map(v0, v1))
                    T.writes(reg[thread_id, local_id])
                    reg[thread_id, local_id] = memory[v0, v1]

    @T.prim_func
    def mfma_load_impl(reg_handle: T.handle, memory_handle: T.handle) -> None:
        s0 = T.int32()
        s1 = T.int32()

        memory = T.match_buffer(
            memory_handle,
            memory_shape,
            dtype,
            align=64,
            offset_factor=1,
            scope=scope,
            strides=[s0, s1],
        )
        reg = T.match_buffer(
            reg_handle, (WARP_SIZE, local_size), dtype, align=64, offset_factor=1, scope="warp")

        with T.block("root"):
            T.reads(memory[0:row_dim, 0:col_dim])
            T.writes(reg[0:WARP_SIZE, 0:local_size])
            for tx in T.thread_binding(WARP_SIZE, "threadIdx.x"):
                for local_id in T.serial(local_size):
                    row, col = T.meta_var(reverse_index_map(tx, local_id))
                    reg[tx, local_id] = memory[row, col]

    return mfma_load_desc, mfma_load_impl


def get_mfma_intrin(k_dim, in_dtype="float32", out_dtype="float32", b_transposed=False):

    local_size = (M_DIM * k_dim) // WARP_SIZE
    local_size_out = (M_DIM * N_DIM) // WARP_SIZE
    compute_in_dtype = in_dtype if local_size == 1 else f"{in_dtype}x{local_size}"
    compute_out_dtype = out_dtype if local_size_out == 1 else f"{out_dtype}x{local_size_out}"

    if k_dim == 4:
        index_map_A = shared_16x4_to_local_64x1_layout_A
        index_map_B = shared_4x16_to_local_64x1_layout_B
        index_map_C = shared_16x16_to_local_64x4_layout_C
    elif k_dim == 16:
        index_map_A = shared_16x16_to_local_64x4_layout_A
        index_map_B = shared_16x16_to_local_64x4_layout_A if b_transposed else shared_16x16_to_local_64x4_layout_B
        index_map_C = shared_16x16_to_local_64x4_layout_C
    else:
        raise ValueError("k_dim must be 4 or 16 currently")

    out_dtype_abbrv = {"float16": "f16", "float32": "f32", "int8": "i8", "int32": "i32"}[out_dtype]

    in_dtype_abbrv = {"float16": "f16", "float32": "f32", "int8": "i8", "int32": "i32"}[in_dtype]

    mfma_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}{in_dtype_abbrv}"

    def maybe_cast(v):
        if out_dtype != in_dtype:
            return Cast(out_dtype, v)
        return v

    def maybe_swap(i, j):
        if b_transposed:
            return j, i
        return i, j

    @T.prim_func
    def mfma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (WARP_SIZE, local_size), in_dtype, offset_factor=16, scope="warp")
        B = T.match_buffer(b, (WARP_SIZE, local_size), in_dtype, offset_factor=16, scope="warp")
        C = T.match_buffer(
            c, (WARP_SIZE, local_size_out), out_dtype, offset_factor=16, scope="warp")

        with T.block("root"):
            T.reads(
                C[0:WARP_SIZE, 0:local_size_out],
                A[0:WARP_SIZE, 0:local_size],
                B[0:WARP_SIZE, 0:local_size],
            )
            T.writes(C[0:WARP_SIZE, 0:local_size_out])

            for i, j, k in T.grid(M_DIM, N_DIM, k_dim):
                with T.block("C"):
                    i, j, k = T.axis.remap("SSR", [i, j, k])
                    b_row_ind, b_col_ind = T.meta_var(maybe_swap(k, j))

                    thread_id_C, local_id_C = T.meta_var(index_map_C(i, j))
                    thread_id_A, local_id_A = T.meta_var(index_map_A(i, k))
                    thread_id_B, local_id_B = T.meta_var(index_map_B(b_row_ind, b_col_ind))

                    T.reads(
                        C[thread_id_C, local_id_C],
                        A[thread_id_A, local_id_A],
                        B[thread_id_B, local_id_B],
                    )
                    T.writes(C[thread_id_C, local_id_C])

                    C[thread_id_C, local_id_C] += maybe_cast(
                        A[thread_id_A, local_id_A]) * maybe_cast(B[thread_id_B, local_id_B])

    @T.prim_func
    def mfma_sync_impl_float(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (WARP_SIZE, local_size), in_dtype, offset_factor=16, scope="warp")
        B = T.match_buffer(b, (WARP_SIZE, local_size), in_dtype, offset_factor=16, scope="warp")
        C = T.match_buffer(
            c, (WARP_SIZE, local_size_out), out_dtype, offset_factor=16, scope="warp")

        with T.block("root"):
            T.reads(
                A[0:WARP_SIZE, 0:local_size],
                B[0:WARP_SIZE, 0:local_size],
                C[0:WARP_SIZE, 0:local_size_out],
            )
            T.writes(C[0:WARP_SIZE, 0:local_size_out])
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, WARP_SIZE)
            T.evaluate(
                T.tvm_mfma(
                    mfma_suffix,
                    "row",
                    "row",
                    compute_in_dtype,
                    compute_in_dtype,
                    compute_out_dtype,
                    A.data,
                    A.elem_offset // (WARP_SIZE * local_size_out),
                    B.data,
                    B.elem_offset // (WARP_SIZE * local_size_out),
                    C.data,
                    C.elem_offset // (WARP_SIZE * local_size_out),
                    dtype=compute_out_dtype,
                ))

    @T.prim_func
    def mfma_sync_impl_integer(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (WARP_SIZE, local_size), in_dtype, offset_factor=1, scope="warp")
        B = T.match_buffer(b, (WARP_SIZE, local_size), in_dtype, offset_factor=1, scope="warp")
        C = T.match_buffer(c, (WARP_SIZE, local_size_out), out_dtype, offset_factor=1, scope="warp")

        with T.block("root"):
            T.reads(
                A[0:WARP_SIZE, 0:local_size],
                B[0:WARP_SIZE, 0:local_size],
                C[0:WARP_SIZE, 0:local_size_out],
            )
            T.writes(C[0:WARP_SIZE, 0:local_size_out])
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, WARP_SIZE)

            T.evaluate(
                T.tvm_mfma(
                    mfma_suffix,
                    "row",
                    "row",
                    compute_in_dtype,
                    compute_in_dtype,
                    compute_out_dtype,
                    T.call_intrin("int32", "tir.reinterpret", A.data),
                    A.elem_offset,
                    T.call_intrin("int32", "tir.reinterpret", B.data),
                    B.elem_offset,
                    C.data,
                    C.elem_offset // (WARP_SIZE * local_size_out),
                    dtype=compute_out_dtype,
                ))

    return (mfma_sync_desc,
            mfma_sync_impl_integer) if in_dtype == "int8" else (mfma_sync_desc,
                                                                mfma_sync_impl_float)


def get_mfma_store_intrin(local_size=4, dtype="float32", scope="global"):

    index_map = shared_16x16_to_local_64x4_layout_C

    @T.prim_func
    def mfma_store_desc(a: T.handle, c: T.handle) -> None:
        C_warp = T.match_buffer(a, [WARP_SIZE, local_size], dtype=dtype, scope="warp")
        C = T.match_buffer(c, [M_DIM, N_DIM], dtype=dtype, scope=scope)

        with T.block("root"):
            T.reads(C_warp[0:WARP_SIZE, 0:local_size])
            T.writes(C[0:M_DIM, 0:N_DIM])
            for i0, i1 in T.grid(M_DIM, N_DIM):
                with T.block("C_warp"):
                    v0, v1 = T.axis.remap("SS", [i0, i1])
                    thread_id, local_id = T.meta_var(index_map(v0, v1))
                    T.reads(C_warp[thread_id, local_id])
                    T.writes(C[v0, v1])
                    C[v0, v1] = C_warp[thread_id, local_id]

    @T.prim_func
    def mfma_store_impl(a: T.handle, c: T.handle) -> None:
        s0 = T.var("int32")
        s1 = T.var("int32")
        C_warp = T.match_buffer(
            a, [WARP_SIZE, local_size], dtype=dtype, scope="warp", offset_factor=16)
        C = T.match_buffer(
            c, [M_DIM, N_DIM], dtype=dtype, scope=scope, offset_factor=1, strides=[s0, s1])

        with T.block("root"):
            T.reads(C_warp[0:WARP_SIZE, 0:local_size])
            T.writes(C[0:M_DIM, 0:N_DIM])
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, WARP_SIZE)
            T.evaluate(
                T.tvm_mfma_store(
                    M_DIM,
                    N_DIM,
                    C.access_ptr("w"),
                    C_warp.data,
                    C_warp.elem_offset // (WARP_SIZE),
                    s0,
                    dtype=dtype,
                ))

    return mfma_store_desc, mfma_store_impl


HIP_MFMA_fill_16x16_f32_INTRIN = "HIP_mfma_fill_16x16_f32"
TensorIntrin.register(HIP_MFMA_fill_16x16_f32_INTRIN, *get_mma_fill_intrin("float32", 4))

HIP_MFMA_fill_16x16_i32_INTRIN = "HIP_mfma_fill_16x16_i32"
TensorIntrin.register(HIP_MFMA_fill_16x16_i32_INTRIN, *get_mma_fill_intrin("int", 4))

HIP_MFMA_LOAD_16x16_A_SHARED_s8_INTRIN = "hip_mfma_load_16x16_a_shared_s8"
TensorIntrin.register(HIP_MFMA_LOAD_16x16_A_SHARED_s8_INTRIN,
                      *get_mfma_load_intrin(16, "int8", "shared"))
HIP_MFMA_LOAD_16x16_B_SHARED_s8_INTRIN = "hip_mfma_load_b_16x16_shared_s8"
TensorIntrin.register(HIP_MFMA_LOAD_16x16_B_SHARED_s8_INTRIN,
                      *get_mfma_load_intrin(16, "int8", "shared", is_b=True))

HIP_MFMA_LOAD_16x16_A_SHARED_f16_INTRIN = "hip_mfma_load_16x16_a_shared_f16"
TensorIntrin.register(HIP_MFMA_LOAD_16x16_A_SHARED_f16_INTRIN,
                      *get_mfma_load_intrin(16, "float16", "shared"))
HIP_MFMA_LOAD_16x16_B_SHARED_f16_INTRIN = "hip_mfma_load_b_16x16_shared_f16"
TensorIntrin.register(HIP_MFMA_LOAD_16x16_B_SHARED_f16_INTRIN,
                      *get_mfma_load_intrin(16, "float16", "shared", is_b=True))
HIP_MFMA_LOAD_16x16_B_TRANS_SHARED_f16_INTRIN = "hip_mfma_load_b_trans_16x16_shared_f16"
TensorIntrin.register(HIP_MFMA_LOAD_16x16_B_TRANS_SHARED_f16_INTRIN,
                      *get_mfma_load_intrin(16, "float16", "shared", is_b=True, transposed=True))

HIP_MFMA_LOAD_16x4_A_SHARED_f32_INTRIN = "hip_mfma_load_16x4_a_shared_f32"
TensorIntrin.register(HIP_MFMA_LOAD_16x4_A_SHARED_f32_INTRIN,
                      *get_mfma_load_intrin(4, "float32", "shared"))
HIP_MFMA_LOAD_16x4_B_SHARED_f32_INTRIN = "hip_mfma_load_b_16x4_shared_f32"
TensorIntrin.register(HIP_MFMA_LOAD_16x4_B_SHARED_f32_INTRIN,
                      *get_mfma_load_intrin(4, "float32", "shared", is_b=True))
HIP_MFMA_LOAD_16x4_B_TRANS_SHARED_f32_INTRIN = "hip_mfma_load_b_trans_16x4_shared_f32"
TensorIntrin.register(HIP_MFMA_LOAD_16x4_B_TRANS_SHARED_f32_INTRIN,
                      *get_mfma_load_intrin(4, "float32", "shared", is_b=True, transposed=True))

HIP_MFMA_f32f32f32_INTRIN = "hip_mfma_f32f32f32"
TensorIntrin.register(HIP_MFMA_f32f32f32_INTRIN, *get_mfma_intrin(4, "float32", "float32"))

HIP_MFMA_f16f16f32_INTRIN = "hip_mfma_f16f16f32"
TensorIntrin.register(HIP_MFMA_f16f16f32_INTRIN, *get_mfma_intrin(16, "float16", "float32"))

HIP_MFMA_f16f16f32_TRANS_INTRIN = "hip_mfma_f16f16f32_trans"
TensorIntrin.register(HIP_MFMA_f16f16f32_TRANS_INTRIN,
                      *get_mfma_intrin(16, "float16", "float32", b_transposed=True))

HIP_MFMA_s8s8s32_INTRIN = "hip_mfma_s8s8s32"
TensorIntrin.register(HIP_MFMA_s8s8s32_INTRIN, *get_mfma_intrin(16, "int8", "int32"))

HIP_MFMA_STORE_16x16_s32_INTRIN = "hip_mfma_store_16x16_s32"
TensorIntrin.register(HIP_MFMA_STORE_16x16_s32_INTRIN, *get_mfma_store_intrin(4, "int32", "global"))

HIP_MFMA_STORE_16x16_f32_INTRIN = "hip_mfma_store_16x16_f32"
TensorIntrin.register(HIP_MFMA_STORE_16x16_f32_INTRIN,
                      *get_mfma_store_intrin(4, "float32", "global"))


def get_mfma_intrin_group(
    load_scope: Literal["shared", "shared.dyn"] = "shared",
    store_scope: Literal["global", "shared", "shared.dyn"] = "global",
    a_dtype: Literal["float16", "int8", "bfloat16", "e4m3_float8", "e5m2_float8"] = "float16",
    b_dtype: Literal["float16", "int8", "bfloat16", "e4m3_float8", "e5m2_float8"] = "float16",
    out_dtype: Literal["float16", "float32", "int32"] = "float16",
    trans_a: bool = False,
    trans_b: bool = False,
    not_use_mfma_store_intrinic: bool = True,
    store_to_smem_dtype: Optional[Literal["float16", "float32", "int32"]] = None,
) -> Dict[str, str]:
    """Get a group of intrinsics for mma tensor core with the given configurations

    Parameters
    ----------
    load_scope : Literal["shared", "shared.dyn"]
        The memory scope of the input buffer.

    store_scope : Literal["global", "shared", "shared.dyn"]
        The memory scope of the result buffer.

    a_dtype : str
        The dtype of the input matrix A.

    b_dtype : str
        The dtype of the input matrix B.

    out_dtype : str
        The output data dtype.

    trans_b : bool
        Whether the input matrix B is transposed.

    not_use_mma_store_intrinic : bool
        Whether to not use the mma_store intrinsic. If True, use BufferStore stmts to store the
        result of mma. Otherwise, use mfma_store intrinsic.

        This is because if we use mfma_store intrinsic, during swizzling shared memory visits, our
        rearrangement scheme will involve areas accessed by different mma_store calls. This makes
        swizzling quite complex. But BufferStore will not face this problem.

    store_to_smem_dtype : Optional[Literal["float16", "float32", "int32"]]
        The dtype that we use to store from register to shared memory. By default it is out_dtype.

    Returns
    -------
    ret : Dict[str, str]
        A group of tensor intrinsics.
    """
    assert load_scope in ["shared", "shared.dyn"]
    assert store_scope in ["global", "shared", "shared.dyn"]
    assert a_dtype in ["float16", "bfloat16", "int8", "e4m3_float8", "e5m2_float8"]
    assert b_dtype in ["float16", "bfloat16", "int8", "e4m3_float8", "e5m2_float8"]
    assert out_dtype in ["float16", "float32", "int32"]

    shape = "16x16"

    dtype_mapping = {
        "float16": "f16",
        "bfloat16": "bf16",
        "float32": "f32",
        "int8": "i8",
        "e4m3_float8": "e4m3",
        "e5m2_float8": "e5m2",
        "int32": "i32",
    }
    a_dtype = dtype_mapping[a_dtype]
    b_dtype = dtype_mapping[b_dtype]
    out_dtype = dtype_mapping[out_dtype]

    # e.g. HIP_mfma_fill_16x16_f32
    init_intrin = f"HIP_mfma_fill_{shape}_{out_dtype}"

    # TODO change format
    # e.g. hip_mfma_load_16x4_a_shared_f32, hip_mfma_load_16x16_a_shared_s8
    # trans_a = "_trans" if trans_a else ""
    # trans_b = "_trans" if trans_b else ""

    load_a_intrin = f"hip_mfma_load_{shape}_a_shared_{a_dtype}"
    # hip_mfma_load_b_trans_16x16_shared_f16
    load_b_intrin = f"hip_mfma_load_b_{shape}_shared_{b_dtype}" if trans_b is False else f"hip_mfma_load_b_trans_{shape}_shared_{b_dtype}"

    # e.g. hip_mfma_f32f32f32, hip_mfma_f16f16f32_trans
    compute_intrin = (f"hip_mfma_{a_dtype}{b_dtype}{out_dtype}") if trans_b is False else (
        f"hip_mfma_{a_dtype}{b_dtype}{out_dtype}_trans")

    # e.g. hip_mfma_store_global_16x16_s32
    store_scope = store_scope.replace(".", "_")
    store_to_smem_dtype = dtype_mapping[store_to_smem_dtype] if store_to_smem_dtype else out_dtype
    store_intrin = f"hip_mfma_store_{shape}_{store_to_smem_dtype}"

    index_map_c = shared_16x16_to_local_64x4_layout_C
    if a_dtype in ["f16", "bf16"]:
        index_map_a = shared_16x16_to_local_64x4_layout_A
        index_map_b = shared_16x16_to_local_64x4_layout_B if trans_b is False else shared_16x16_to_local_64x4_layout_A
    elif a_dtype in ["i8", "e4m3", "e5m2"]:
        index_map_a = shared_16x4_to_local_64x1_layout_A
        index_map_b = shared_4x16_to_local_64x1_layout_B
    else:
        raise ValueError(f"Unsupported in_dtype: {a_dtype}")

    # micro kernel size, the order is [m, n, k]
    micro_kernel: List[int]
    if a_dtype in ["f16", "bf16"]:
        micro_kernel = [16, 16, 16]
    elif a_dtype in ["i8", "e4m3", "e5m2"]:
        micro_kernel = [16, 16, 32]
    else:
        raise ValueError(f"Unsupported in_dtype: {a_dtype}")

    return {
        "init": init_intrin,
        "load_a": load_a_intrin,
        "load_b": load_b_intrin,
        "compute": compute_intrin,
        "store": store_intrin,
        "index_map": [index_map_a, index_map_b, index_map_c],
        "micro_kernel": micro_kernel,
    }


######## ROCrocwmma intrinsics ########


def get_rocwmma_fragment_index(buffer, stride, m_dim, n_dim):
    """Compute rocwmma fragment index using elem_offset of the buffer"""
    frag_index_m = buffer.elem_offset // stride // m_dim
    frag_index_n = buffer.elem_offset % stride // n_dim

    num_fragments_per_row = stride // n_dim
    return frag_index_m * num_fragments_per_row + frag_index_n


def get_rocwmma_load_intrin(
    m_dim: int,
    n_dim: int,
    k_dim: int,
    dtype: str,
    shared_scope: str,
    is_b: bool,
    is_col_major: bool,
) -> Tuple[PrimFunc, PrimFunc]:
    """Generator of rocwmma_load intrins"""
    rocwmma_fragment_scope = "wmma.matrix_{}".format("b" if is_b else "a")
    layout = "col_major" if is_col_major else "row_major"

    @T.prim_func
    def rocwmma_load_desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (m_dim, n_dim), dtype, align=64, offset_factor=16, scope=shared_scope)
        C = T.match_buffer(
            c, (m_dim, n_dim), dtype, align=64, offset_factor=16, scope=rocwmma_fragment_scope)
        with T.block("root"):
            T.reads(A[0:m_dim, 0:n_dim])
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j in T.grid(m_dim, n_dim):
                with T.block("load"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = A[vii, vjj]

    @T.prim_func
    def rocwmma_load_impl(a: T.handle, c: T.handle) -> None:
        s1 = T.int32()
        s0 = T.int32()
        d1 = T.int32()
        d0 = T.int32()
        A = T.match_buffer(
            a,
            (m_dim, n_dim),
            dtype,
            align=64,
            offset_factor=16,
            scope=shared_scope,
            strides=[s1, s0],
        )
        C = T.match_buffer(
            c,
            (m_dim, n_dim),
            dtype,
            align=64,
            offset_factor=16,
            scope=rocwmma_fragment_scope,
            strides=[d1, d0],
        )
        with T.block("root"):
            T.reads(A[0:m_dim, 0:n_dim])
            T.writes(C[0:m_dim, 0:n_dim])
            T.evaluate(
                T.tvm_load_matrix_sync(
                    C.data,
                    m_dim,
                    n_dim,
                    k_dim,
                    get_rocwmma_fragment_index(C, d1, m_dim, n_dim),
                    A.access_ptr("r"),
                    s1,
                    layout,
                    dtype="handle",
                ))

    return rocwmma_load_desc, rocwmma_load_impl


def get_rocwmma_fill_intrin(m_dim: int, n_dim: int, k_dim: int,
                            dtype: str) -> Tuple[PrimFunc, PrimFunc]:
    """Generator of rocwmma_fill intrins"""
    zero = IntImm("int32", 0).astype(dtype)

    @T.prim_func
    def rocwmma_fill_desc(c: T.handle) -> None:
        C = T.match_buffer(
            c, (m_dim, n_dim), dtype, align=64, offset_factor=16, scope="wmma.accumulator")
        with T.block("root"):
            T.reads()
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j in T.grid(m_dim, n_dim):
                with T.block("init"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = zero

    @T.prim_func
    def rocwmma_fill_impl(c: T.handle) -> None:
        d1 = T.int32()
        d0 = T.int32()
        C = T.match_buffer(
            c,
            (m_dim, n_dim),
            dtype,
            align=64,
            offset_factor=16,
            scope="wmma.accumulator",
            strides=[d1, d0],
        )
        with T.block("root"):
            T.reads()
            T.writes(C[0:m_dim, 0:n_dim])
            T.evaluate(
                T.tvm_fill_fragment(
                    C.data,
                    m_dim,
                    n_dim,
                    k_dim,
                    get_rocwmma_fragment_index(C, d1, m_dim, n_dim),
                    zero,
                    dtype="handle",
                ))

    return rocwmma_fill_desc, rocwmma_fill_impl


def get_rocwmma_store_intrin(m_dim: int, n_dim: int, k_dim: int, dtype: str,
                             scope: str) -> Tuple[PrimFunc, PrimFunc]:
    """Generator of rocwmma_store intrins"""

    @T.prim_func
    def rocwmma_store_desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (m_dim, n_dim), dtype, align=64, offset_factor=16, scope="wmma.accumulator")
        C = T.match_buffer(c, (m_dim, n_dim), dtype, align=64, offset_factor=16, scope=scope)
        with T.block("root"):
            T.reads(A[0:m_dim, 0:n_dim])
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j in T.grid(m_dim, n_dim):
                with T.block("store"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = A[vii, vjj]

    @T.prim_func
    def rocwmma_store_impl(a: T.handle, c: T.handle) -> None:
        s1 = T.int32()
        s0 = T.int32()
        d1 = T.int32()
        d0 = T.int32()
        A = T.match_buffer(
            a,
            (m_dim, n_dim),
            dtype,
            align=64,
            offset_factor=16,
            scope="wmma.accumulator",
            strides=[d1, d0],
        )
        C = T.match_buffer(
            c, (m_dim, n_dim), dtype, align=64, offset_factor=16, scope=scope, strides=[s1, s0])
        with T.block("root"):
            T.reads(A[0:m_dim, 0:n_dim])
            T.writes(C[0:m_dim, 0:n_dim])
            T.evaluate(
                T.tvm_store_matrix_sync(
                    A.data,
                    m_dim,
                    n_dim,
                    k_dim,
                    get_rocwmma_fragment_index(A, d1, m_dim, n_dim),
                    C.access_ptr("w"),
                    s1,
                    "row_major",
                    dtype="handle",
                ))

    return rocwmma_store_desc, rocwmma_store_impl


def get_rocwmma_sync_intrin(m_dim: int, n_dim: int, k_dim: int, in_dtype: str, out_dtype: str,
                            b_transposed: bool) -> Tuple[PrimFunc, PrimFunc]:
    """Generator of rocwmma_sync intrins"""

    def maybe_cast(v):
        if in_dtype != out_dtype:
            return Cast(out_dtype, v)
        return v

    def maybe_swap(i, j):
        if b_transposed:
            return j, i
        return i, j

    b_shape_0, b_shape_1 = maybe_swap(k_dim, n_dim)

    @T.prim_func
    def rocwmma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (m_dim, k_dim), in_dtype, align=64, offset_factor=16, scope="wmma.matrix_a")
        B = T.match_buffer(
            b,
            maybe_swap(k_dim, n_dim),
            in_dtype,
            align=64,
            offset_factor=16,
            scope="wmma.matrix_b",
        )
        C = T.match_buffer(
            c, (m_dim, n_dim), out_dtype, align=64, offset_factor=16, scope="wmma.accumulator")

        with T.block("root"):
            T.reads(C[0:m_dim, 0:n_dim], A[0:m_dim, 0:k_dim], B[0:b_shape_0, 0:b_shape_1])
            T.writes(C[0:m_dim, 0:n_dim])
            for i, j, k in T.grid(m_dim, n_dim, k_dim):
                with T.block(""):
                    vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                    B_index_0, B_index_1 = maybe_swap(vkk, vjj)
                    C[vii,
                      vjj] = C[vii,
                               vjj] + maybe_cast(A[vii, vkk]) * maybe_cast(B[B_index_0, B_index_1])

    @T.prim_func
    def rocwmma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
        a1 = T.int32()
        a0 = T.int32()
        b1 = T.int32()
        b0 = T.int32()
        c1 = T.int32()
        c0 = T.int32()

        A = T.match_buffer(
            a,
            (m_dim, k_dim),
            in_dtype,
            align=64,
            offset_factor=16,
            scope="wmma.matrix_a",
            strides=[a1, a0],
        )
        B = T.match_buffer(
            b,
            maybe_swap(k_dim, n_dim),
            in_dtype,
            align=64,
            offset_factor=16,
            scope="wmma.matrix_b",
            strides=[b1, b0],
        )
        C = T.match_buffer(
            c,
            (m_dim, n_dim),
            out_dtype,
            align=64,
            offset_factor=16,
            scope="wmma.accumulator",
            strides=[c1, c0],
        )

        with T.block("root"):
            T.reads(C[0:m_dim, 0:n_dim], A[0:m_dim, 0:k_dim], B[0:b_shape_0, 0:b_shape_1])
            T.writes(C[0:m_dim, 0:n_dim])
            T.evaluate(
                T.tvm_mma_sync(
                    C.data,
                    get_rocwmma_fragment_index(C, c1, m_dim, n_dim),
                    A.data,
                    get_rocwmma_fragment_index(A, a1, m_dim, k_dim),
                    B.data,
                    get_rocwmma_fragment_index(B, b1, b_shape_0, b_shape_1),
                    C.data,
                    get_rocwmma_fragment_index(C, c1, m_dim, n_dim),
                    dtype="handle",
                ))

    return rocwmma_sync_desc, rocwmma_sync_impl


ROCWMMA_SYNC_16x16x16_f16f16f32_INTRIN = "rocwmma_sync_16x16x16_f16f16f32"
TensorIntrin.register(
    ROCWMMA_SYNC_16x16x16_f16f16f32_INTRIN,
    *get_rocwmma_sync_intrin(16, 16, 16, "float16", "float32", False),
)

ROCWMMA_SYNC_16x16x16_f16f16f32_TRANS_INTRIN = "rocwmma_sync_16x16x16_f16f16f32_trans"
TensorIntrin.register(
    ROCWMMA_SYNC_16x16x16_f16f16f32_TRANS_INTRIN,
    *get_rocwmma_sync_intrin(16, 16, 16, "float16", "float32", True),
)

ROCWMMA_SYNC_16x16x16_f16f16f16_INTRIN = "rocwmma_sync_16x16x16_f16f16f16"
TensorIntrin.register(
    ROCWMMA_SYNC_16x16x16_f16f16f16_INTRIN,
    *get_rocwmma_sync_intrin(16, 16, 16, "float16", "float16", False),
)

ROCWMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN = "rocwmma_sync_16x16x16_f16f16f16_trans"
TensorIntrin.register(
    ROCWMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN,
    *get_rocwmma_sync_intrin(16, 16, 16, "float16", "float16", True),
)

ROCWMMA_SYNC_16x16x16_s8s8s32_INTRIN = "rocwmma_sync_16x16x16_s8s8s32"
TensorIntrin.register(
    ROCWMMA_SYNC_16x16x16_s8s8s32_INTRIN,
    *get_rocwmma_sync_intrin(16, 16, 16, "int8", "int32", False),
)

ROCWMMA_SYNC_16x16x16_s8s8s32_TRANS_INTRIN = "rocwmma_sync_16x16x16_s8s8s32_trans"
TensorIntrin.register(
    ROCWMMA_SYNC_16x16x16_s8s8s32_TRANS_INTRIN,
    *get_rocwmma_sync_intrin(16, 16, 16, "int8", "int32", True),
)

ROCWMMA_LOAD_16x16x16_F16_A_INTRIN = "rocwmma_load_16x16x16_f16_a_shared"
TensorIntrin.register(
    ROCWMMA_LOAD_16x16x16_F16_A_INTRIN,
    *get_rocwmma_load_intrin(16, 16, 16, "float16", "shared", False, False),
)

ROCWMMA_LOAD_16x16x16_F16_A_DYN_INTRIN = "rocwmma_load_16x16x16_f16_a_shared_dyn"
TensorIntrin.register(
    ROCWMMA_LOAD_16x16x16_F16_A_DYN_INTRIN,
    *get_rocwmma_load_intrin(16, 16, 16, "float16", "shared.dyn", False, False),
)

ROCWMMA_LOAD_16x16x16_F16_B_INTRIN = "rocwmma_load_16x16x16_f16_b_shared"
TensorIntrin.register(
    ROCWMMA_LOAD_16x16x16_F16_B_INTRIN,
    *get_rocwmma_load_intrin(16, 16, 16, "float16", "shared", True, False),
)

ROCWMMA_LOAD_16x16x16_F16_B_DYN_INTRIN = "rocwmma_load_16x16x16_f16_b_shared_dyn"
TensorIntrin.register(
    ROCWMMA_LOAD_16x16x16_F16_B_DYN_INTRIN,
    *get_rocwmma_load_intrin(16, 16, 16, "float16", "shared.dyn", True, False),
)

ROCWMMA_LOAD_16x16x16_F16_A_TRANS_INTRIN = "rocwmma_load_16x16x16_f16_a_trans_shared"
TensorIntrin.register(
    ROCWMMA_LOAD_16x16x16_F16_A_TRANS_INTRIN,
    *get_rocwmma_load_intrin(16, 16, 16, "float16", "shared", False, True),
)

ROCWMMA_LOAD_16x16x16_F16_A_TRANS_DYN_INTRIN = "rocwmma_load_16x16x16_f16_a_trans_shared_dyn"
TensorIntrin.register(
    ROCWMMA_LOAD_16x16x16_F16_A_TRANS_DYN_INTRIN,
    *get_rocwmma_load_intrin(16, 16, 16, "float16", "shared.dyn", False, True),
)

ROCWMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN = "rocwmma_load_16x16x16_f16_b_trans_shared"
TensorIntrin.register(
    ROCWMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN,
    *get_rocwmma_load_intrin(16, 16, 16, "float16", "shared", True, True),
)

ROCWMMA_LOAD_16x16x16_F16_B_TRANS_DYN_INTRIN = "rocwmma_load_16x16x16_f16_b_trans_shared_dyn"
TensorIntrin.register(
    ROCWMMA_LOAD_16x16x16_F16_B_TRANS_DYN_INTRIN,
    *get_rocwmma_load_intrin(16, 16, 16, "float16", "shared.dyn", True, True),
)

ROCWMMA_LOAD_16x16x16_S8_A_INTRIN = "rocwmma_load_16x16x16_s8_a_shared"
TensorIntrin.register(
    ROCWMMA_LOAD_16x16x16_S8_A_INTRIN,
    *get_rocwmma_load_intrin(16, 16, 16, "int8", "shared", False, False),
)

ROCWMMA_LOAD_16x16x16_S8_A_DYN_INTRIN = "rocwmma_load_16x16x16_s8_a_shared_dyn"
TensorIntrin.register(
    ROCWMMA_LOAD_16x16x16_S8_A_DYN_INTRIN,
    *get_rocwmma_load_intrin(16, 16, 16, "int8", "shared.dyn", False, False),
)

ROCWMMA_LOAD_16x16x16_S8_B_INTRIN = "rocwmma_load_16x16x16_s8_b_shared"
TensorIntrin.register(
    ROCWMMA_LOAD_16x16x16_S8_B_INTRIN,
    *get_rocwmma_load_intrin(16, 16, 16, "int8", "shared", True, False),
)

ROCWMMA_LOAD_16x16x16_S8_B_DYN_INTRIN = "rocwmma_load_16x16x16_s8_b_shared_dyn"
TensorIntrin.register(
    ROCWMMA_LOAD_16x16x16_S8_B_DYN_INTRIN,
    *get_rocwmma_load_intrin(16, 16, 16, "int8", "shared.dyn", True, False),
)

ROCWMMA_LOAD_16x16x16_S8_A_TRANS_INTRIN = "rocwmma_load_16x16x16_s8_a_trans_shared"
TensorIntrin.register(
    ROCWMMA_LOAD_16x16x16_S8_A_TRANS_INTRIN,
    *get_rocwmma_load_intrin(16, 16, 16, "int8", "shared", False, True),
)

ROCWMMA_LOAD_16x16x16_S8_A_TRANS_DYN_INTRIN = "rocwmma_load_16x16x16_s8_a_trans_shared_dyn"
TensorIntrin.register(
    ROCWMMA_LOAD_16x16x16_S8_A_TRANS_DYN_INTRIN,
    *get_rocwmma_load_intrin(16, 16, 16, "int8", "shared.dyn", False, True),
)

ROCWMMA_LOAD_16x16x16_S8_B_TRANS_INTRIN = "rocwmma_load_16x16x16_s8_b_trans_shared"
TensorIntrin.register(
    ROCWMMA_LOAD_16x16x16_S8_B_TRANS_INTRIN,
    *get_rocwmma_load_intrin(16, 16, 16, "int8", "shared", True, True),
)

ROCWMMA_LOAD_16x16x16_S8_B_TRANS_DYN_INTRIN = "rocwmma_load_16x16x16_s8_b_trans_shared_dyn"
TensorIntrin.register(
    ROCWMMA_LOAD_16x16x16_S8_B_TRANS_DYN_INTRIN,
    *get_rocwmma_load_intrin(16, 16, 16, "int8", "shared.dyn", True, True),
)

ROCWMMA_FILL_16x16x16_F32_INTRIN = "rocwmma_fill_16x16x16_f32"
TensorIntrin.register(ROCWMMA_FILL_16x16x16_F32_INTRIN,
                      *get_rocwmma_fill_intrin(16, 16, 16, "float32"))

ROCWMMA_FILL_16x16x16_F16_INTRIN = "rocwmma_fill_16x16x16_f16"
TensorIntrin.register(ROCWMMA_FILL_16x16x16_F16_INTRIN,
                      *get_rocwmma_fill_intrin(16, 16, 16, "float16"))

ROCWMMA_FILL_16x16x16_S32_INTRIN = "rocwmma_fill_16x16x16_s32"
TensorIntrin.register(ROCWMMA_FILL_16x16x16_S32_INTRIN,
                      *get_rocwmma_fill_intrin(16, 16, 16, "int32"))

ROCWMMA_STORE_16x16x16_F32_SHARED_INTRIN = "rocwmma_store_16x16x16_f32_shared"
TensorIntrin.register(ROCWMMA_STORE_16x16x16_F32_SHARED_INTRIN,
                      *get_rocwmma_store_intrin(16, 16, 16, "float32", "shared"))

ROCWMMA_STORE_16x16x16_F32_SHARED_DYN_INTRIN = "rocwmma_store_16x16x16_f32_shared_dyn"
TensorIntrin.register(
    ROCWMMA_STORE_16x16x16_F32_SHARED_DYN_INTRIN,
    *get_rocwmma_store_intrin(16, 16, 16, "float32", "shared.dyn"),
)

ROCWMMA_STORE_16x16x16_F16_SHARED_INTRIN = "rocwmma_store_16x16x16_f16_shared"
TensorIntrin.register(ROCWMMA_STORE_16x16x16_F16_SHARED_INTRIN,
                      *get_rocwmma_store_intrin(16, 16, 16, "float16", "shared"))

ROCWMMA_STORE_16x16x16_F16_SHARED_DYN_INTRIN = "rocwmma_store_16x16x16_f16_shared_dyn"
TensorIntrin.register(
    ROCWMMA_STORE_16x16x16_F16_SHARED_DYN_INTRIN,
    *get_rocwmma_store_intrin(16, 16, 16, "float16", "shared.dyn"),
)

ROCWMMA_STORE_16x16x16_S32_SHARED_INTRIN = "rocwmma_store_16x16x16_s32_shared"
TensorIntrin.register(ROCWMMA_STORE_16x16x16_S32_SHARED_INTRIN,
                      *get_rocwmma_store_intrin(16, 16, 16, "int32", "shared"))

ROCWMMA_STORE_16x16x16_S32_SHARED_DYN_INTRIN = "rocwmma_store_16x16x16_s32_shared_dyn"
TensorIntrin.register(
    ROCWMMA_STORE_16x16x16_S32_SHARED_DYN_INTRIN,
    *get_rocwmma_store_intrin(16, 16, 16, "int32", "shared.dyn"),
)

ROCWMMA_STORE_16x16x16_F32_GLOBAL_INTRIN = "rocwmma_store_16x16x16_f32_global"
TensorIntrin.register(ROCWMMA_STORE_16x16x16_F32_GLOBAL_INTRIN,
                      *get_rocwmma_store_intrin(16, 16, 16, "float32", "global"))

ROCWMMA_STORE_16x16x16_F16_GLOBAL_INTRIN = "rocwmma_store_16x16x16_f16_global"
TensorIntrin.register(ROCWMMA_STORE_16x16x16_F16_GLOBAL_INTRIN,
                      *get_rocwmma_store_intrin(16, 16, 16, "float16", "global"))

ROCWMMA_STORE_16x16x16_S32_GLOBAL_INTRIN = "rocwmma_store_16x16x16_s32_global"
TensorIntrin.register(ROCWMMA_STORE_16x16x16_S32_GLOBAL_INTRIN,
                      *get_rocwmma_store_intrin(16, 16, 16, "int32", "global"))
