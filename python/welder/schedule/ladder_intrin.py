# import tvm convert
from tvm.runtime import convert
from tvm.tir.expr import Cast, IntImm
from tvm.tir.function import TensorIntrin
from tvm.script import tir as T
from tvm._ffi import register_func
lift = convert

M_DIM = 16
N_DIM = 16
WARP_SIZE = 32
HALF_WARP = WARP_SIZE // 2
HALF_WARP_expr = lift(HALF_WARP)


def C_shared_16x16_to_ldmatrix_32x8_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 8) // 2
    return thread_id, 4 * (j // 8) + (i // 8) * 2 + (j % 2)


def A_B_shared_16x16_to_ldmatrix_32x8_layout(i, j):
    return (i * 2 + j // 8, j % 8)


# @register_func("tir.index_map.shared_16x16_to_ldmatrix_32x8_layout")
# def index_map_shared_16x16_to_ldmatrix_32x8_layout(ind):
#     i, j = ind[0], ind[1]
#     thread_id, local_id = C_shared_16x16_to_ldmatrix_32x8_layout(i, j)
#     return convert([thread_id, local_id])


def shared_32x16_to_ldmatrix_32x16_layout(i, j):
    return (i * 2 + j // 16, j % 16)


'''
    32 x 16 means we have 32 threads in a warp, and each thread has 16 elements
'''


def shared_16x32_to_ldmatrix_32x16_layout(i, j):
    # convert (i // 8, j // 16, i % 8, j % 16) to a 2d array
    return (i * 2 + j // 16, j % 16)


def shared_16x16_to_ldmatrix_32x8_permutation(i, j):
    return (i // 8) * 16 + (j // 8) * 8 + i % 8, j % 8


def shared_load_16x16_to_A_global_16x16_layout(i, j):
    thread_id = i + (j // 8) * 16
    row = thread_id // 2
    col = (thread_id % 2) * 8 + (j % 8)
    return row, col

def A_global_16x16_to_shared_load_16x16_layout(i, j):
    thread_id = i * 2 + j // 8
    row = thread_id % 16
    col = (j % 8) + (thread_id // 16) * 8
    return row, col


# NN Layout Transform: Vector Wise Pattern
def shared_load_16x16_to_B_global_16x16_layout(i, j):
    row_group = i // 8
    col_group = j // 8

    thread_id = (row_group % 2) * 8 + (col_group % 2) * 8 * 2 + (i % 8)
    row = thread_id // 2
    col = (thread_id % 2) * 8 + (j % 8)
    return row, col



def B_global_16x16_to_shared_load_16x16_layout(i, j):
    thread_id = i * 2 + j // 8
    row = (i // 8) * 8 + (thread_id % 8)
    col = (j % 8) + 8 * ((thread_id // 8) % 2)
    return row, col


def global_16x32_to_shared_load_16x32_layout(i, j):
    # 0, 0-16 -> 0, 0-16
    # 1, 0-16 -> 1, 0-16
    # 2, 0-16 -> 2, 0-16
    # 3, 0-16 -> 3, 0-16
    """
        re-orgnize the global memory to shared memory access pattern
        key context : 
            j % 16 -> index
            j // 16 
            i % 16 -> index
    """
    thread_id = i * 2 + j // 16
    row = thread_id % 16
    col = (j % 16) + (thread_id // 16) * 16
    return row, col


def shared_16x32_to_ldmatrix_32x16_permutation(i, j):
    return (j // 16) * 16 + (i // 8) * 8 + i % 8, j % 16


# def shared_16x32_to_ldmatrix_32x16_layout(i, j):
#     return shared_16x32_to_ldmatrix_32x16_permutation(i, j)

def get_ldmatrix_intrin(k_dim, dtype, is_b, transposed, shared_scope="shared"):
    local_size = (M_DIM * k_dim) // WARP_SIZE
    shared_offset = None
    index_map = None

    if transposed:
        assert is_b, "Transposed A matrix not supported"

    ldmatrix_col_major = is_b and not transposed

    if k_dim == 16:
        assert dtype == "float16"

        index_map = A_B_shared_16x16_to_ldmatrix_32x8_layout

        if transposed:
            shared_offset = (
                # stride = 32 if int8 , = 16 if fp16
                lambda tx, stride: 8 * tx
            )
        else:
            # assert False, "Still not yet implemente none tranposed"
            def shared_offset(tx, stride):
                return 8 * tx
    else:
        assert (
            k_dim == 32 and dtype == "int8"
        ), "Only k_dim == 16 (float16) or k_dim == 32 (int8) supported for now"

        if ldmatrix_col_major:
            index_map = shared_32x16_to_ldmatrix_32x16_layout
            # A dummy offset, ldmatrix cannot be used for int8 + trans case.
            # We still use the ldmatrix intrinsic, but lower it to a manual loop in the codegen.
            # Only the stride information is required.
            def shared_offset(tx, stride): return 16 * tx
        elif is_b and transposed:
            index_map = shared_16x32_to_ldmatrix_32x16_layout
            # 32x16
            shared_offset = (
                lambda tx, stride: 16 * tx
            )
        else:
            index_map = shared_16x32_to_ldmatrix_32x16_layout
            def shared_offset(tx, stride): return 16 * tx

    assert index_map and shared_offset

    if is_b and not transposed:
        row_dim = k_dim
        col_dim = M_DIM
    else:
        row_dim = M_DIM
        col_dim = k_dim

    shmem_shape = (row_dim, col_dim)

    @T.prim_func
    def ldmatrix_desc(warp_handle: T.handle, shared_handle: T.handle) -> None:
        shared = T.match_buffer(
            shared_handle,
            shmem_shape,
            dtype,
            align=64,
            offset_factor=16,
            scope=shared_scope,
        )
        warp = T.match_buffer(
            warp_handle, (WARP_SIZE, local_size), dtype, align=64, offset_factor=16, scope="warp"
        )

        with T.block("root"):
            T.reads(shared[0:row_dim, 0:col_dim])
            T.writes(warp[0:WARP_SIZE, 0:local_size])

            for ax0, ax1 in T.grid(row_dim, col_dim):
                with T.block("shared_warp"):
                    v0, v1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(shared[v0, v1])
                    thread_id, local_id = T.meta_var(index_map(v0, v1))
                    T.writes(warp[thread_id, local_id])
                    warp[thread_id, local_id] = shared[v0, v1]

    @T.prim_func
    def ldmatrix_impl(warp_handle: T.handle, shared_handle: T.handle) -> None:
        s0 = T.var("int32")
        s1 = T.var("int32")
        shared = T.match_buffer(
            shared_handle,
            shmem_shape,
            dtype,
            align=64,
            offset_factor=16,
            scope=shared_scope,
            strides=[s0, s1],
        )
        warp = T.match_buffer(
            warp_handle, (WARP_SIZE, local_size), dtype, align=64, offset_factor=16, scope="warp"
        )

        with T.block("root"):
            T.reads(shared[0:row_dim, 0:col_dim])
            T.writes(warp[0:WARP_SIZE, 0:local_size])
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, WARP_SIZE)

            T.evaluate(
                T.ptx_ldmatrix(
                    ldmatrix_col_major,
                    4,  # Always load 4 matrices
                    ".b16",
                    warp.data,
                    warp.elem_offset + lift(local_size) * tx,
                    shared.access_ptr("r"),
                    shared_offset(tx, s0),
                    dtype=dtype,
                )
            )

    return ldmatrix_desc, ldmatrix_impl


def get_mma_intrin(k_dim, out_dtype, b_transposed):
    local_size = (M_DIM * k_dim) // WARP_SIZE
    local_size_out = (M_DIM * N_DIM) // 32

    index_map_C = C_shared_16x16_to_ldmatrix_32x8_layout

    if k_dim == 16:
        index_map_A = A_B_shared_16x16_to_ldmatrix_32x8_layout
        index_map_B = A_B_shared_16x16_to_ldmatrix_32x8_layout
        mma_prefix = "m16n8k16"
    elif k_dim == 32 and b_transposed:
        index_map_A = index_map_B = shared_16x32_to_ldmatrix_32x16_layout
        mma_prefix = "m16n8k32"
    elif k_dim == 32 and not b_transposed:
        index_map_A = shared_16x32_to_ldmatrix_32x16_layout
        index_map_B = shared_32x16_to_ldmatrix_32x16_layout
        mma_prefix = "m16n8k32"
    else:
        assert False

    out_dtype_abbrv = {"float16": "fp16",
                       "float32": "fp32", "int32": "int32"}[out_dtype]

    if out_dtype in ["float16", "float32"]:
        in_dtype = "float16"
        in_dtype_abbrv = "fp16"
    else:
        in_dtype = "int8"
        in_dtype_abbrv = "int8"

    def maybe_cast(v):
        if out_dtype in ["float32", "int32"]:
            return Cast(out_dtype, v)
        return v

    def maybe_swap(i, j):
        if b_transposed:
            return j, i
        return i, j

    @T.prim_func
    def mma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (WARP_SIZE, local_size), in_dtype, align=64, offset_factor=16, scope="warp"
        )
        B = T.match_buffer(
            b, (WARP_SIZE, local_size), in_dtype, align=64, offset_factor=16, scope="warp"
        )
        C = T.match_buffer(
            c, (WARP_SIZE, local_size_out), out_dtype, align=64, offset_factor=16, scope="warp"
        )

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
                    b_row_ind, b_col_ind = maybe_swap(k, j)

                    thread_id_C, local_id_C = T.meta_var(index_map_C(i, j))
                    thread_id_A, local_id_A = T.meta_var(index_map_A(i, k))
                    thread_id_B, local_id_B = T.meta_var(
                        index_map_B(b_row_ind, b_col_ind))

                    T.reads(
                        C[thread_id_C, local_id_C],
                        A[thread_id_A, local_id_A],
                        B[thread_id_B, local_id_B],
                    )
                    T.writes(C[thread_id_C, local_id_C])

                    C[thread_id_C, local_id_C] += maybe_cast(
                        A[thread_id_A, local_id_A]
                    ) * maybe_cast(B[thread_id_B, local_id_B])

    @T.prim_func
    def mma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a, (WARP_SIZE, local_size), in_dtype, align=64, offset_factor=16, scope="warp"
        )
        B = T.match_buffer(
            b, (WARP_SIZE, local_size), in_dtype, align=64, offset_factor=16, scope="warp"
        )
        C = T.match_buffer(
            c, (WARP_SIZE, local_size_out), out_dtype, align=64, offset_factor=16, scope="warp"
        )

        with T.block("root"):
            T.reads(
                C[0:WARP_SIZE, 0:local_size_out],
                A[0:WARP_SIZE, 0:local_size],
                B[0:WARP_SIZE, 0:local_size],
            )
            T.writes(C[0:WARP_SIZE, 0:local_size_out])
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, WARP_SIZE)

            T.evaluate(
                T.ptx_mma(
                    mma_prefix,
                    "row",
                    "col",
                    in_dtype_abbrv,
                    in_dtype_abbrv,
                    out_dtype_abbrv,
                    A.data,
                    A.elem_offset + tx * lift(local_size),
                    B.data,
                    B.elem_offset + tx * lift(local_size),
                    C.data,
                    C.elem_offset + tx * lift(local_size_out),
                    False,
                    dtype=out_dtype,
                )
            )

            T.evaluate(
                T.ptx_mma(
                    mma_prefix,
                    "row",
                    "col",
                    in_dtype_abbrv,
                    in_dtype_abbrv,
                    out_dtype_abbrv,
                    A.data,
                    A.elem_offset + tx * lift(local_size),
                    B.data,
                    B.elem_offset + tx * lift(local_size) + lift(local_size) // 2,
                    C.data,
                    C.elem_offset + tx * lift(local_size_out) + lift(local_size_out) // 2,
                    False,
                    dtype=out_dtype,
                )
            )

    return mma_sync_desc, mma_sync_impl


def get_mma_fill_intrin(dtype, local_size):
    zero = IntImm("int32", 0).astype(dtype)

    # Assume M = N = 16
    index_map = C_shared_16x16_to_ldmatrix_32x8_layout

    @T.prim_func
    def mma_fill_desc(a: T.handle) -> None:
        C_warp = T.match_buffer(
            a, [WARP_SIZE, local_size], dtype=dtype, scope="warp")

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
            a, [WARP_SIZE, local_size], dtype=dtype, scope="warp", offset_factor=1
        )

        with T.block("root"):
            T.reads()
            T.writes(C_warp[0:WARP_SIZE, 0:local_size])
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, WARP_SIZE)

            T.evaluate(T.mma_fill(local_size, C_warp.data,
                       C_warp.elem_offset, dtype=dtype))

    return mma_fill_desc, mma_fill_impl


def get_mma_store_intrin(dtype, local_size, scope="global"):
    # Assume M = N = 16
    index_map = C_shared_16x16_to_ldmatrix_32x8_layout

    @T.prim_func
    def mma_store_desc(a: T.handle, c: T.handle) -> None:
        C_warp = T.match_buffer(
            a, [WARP_SIZE, local_size], dtype=dtype, scope="warp")
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
    def mma_store_impl(a: T.handle, c: T.handle) -> None:
        s0 = T.var("int32")
        s1 = T.var("int32")

        C_warp = T.match_buffer(
            a, [WARP_SIZE, local_size], dtype=dtype, scope="warp", offset_factor=1
        )
        C = T.match_buffer(
            c, [M_DIM, N_DIM], dtype=dtype, scope=scope, offset_factor=1, strides=[s0, s1]
        )

        with T.block("root"):
            T.reads(C_warp[0:WARP_SIZE, 0:local_size])
            T.writes(C[0:M_DIM, 0:N_DIM])
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, WARP_SIZE)

            T.evaluate(
                T.mma_store(
                    M_DIM,
                    N_DIM,
                    C.access_ptr("w"),
                    C_warp.data,
                    C_warp.elem_offset,
                    s0,
                    dtype=dtype,
                )
            )

    return mma_store_desc, mma_store_impl


TRICKY_LDMATRIX_16x16_A_INTRIN = "TRICKY_mma.ldmatrix_16x16_a"
TensorIntrin.register(TRICKY_LDMATRIX_16x16_A_INTRIN, *
                      get_ldmatrix_intrin(16, "float16", False, False))

TRICKY_LDMATRIX_16x16_B_INTRIN = "TRICKY_mma.ldmatrix_16x16_b"
TensorIntrin.register(TRICKY_LDMATRIX_16x16_B_INTRIN, *
                      get_ldmatrix_intrin(16, "float16", True, False))

TRICKY_LDMATRIX_16x16_B_TRANS_INTRIN = "TRICKY_mma.ldmatrix_16x16_b_trans"
TensorIntrin.register(
    TRICKY_LDMATRIX_16x16_B_TRANS_INTRIN, *
    get_ldmatrix_intrin(16, "float16", True, True)
)


TRICKY_LDMATRIX_16x16_A_INTRIN_DYN = "TRICKY_mma.ldmatrix_16x16_a_DYN"
TensorIntrin.register(TRICKY_LDMATRIX_16x16_A_INTRIN_DYN, *
                      get_ldmatrix_intrin(16, "float16", False, False, "shared.dyn"))


TRICKY_LDMATRIX_16x16_B_INTRIN_DYN = "TRICKY_mma.ldmatrix_16x16_b_DYN"
TensorIntrin.register(TRICKY_LDMATRIX_16x16_B_INTRIN_DYN, *
                      get_ldmatrix_intrin(16, "float16", True, False, "shared.dyn"))

TRICKY_LDMATRIX_16x16_B_TRANS_INTRIN_DYN = "TRICKY_mma.ldmatrix_16x16_b_trans_DYN"
TensorIntrin.register(
    TRICKY_LDMATRIX_16x16_B_TRANS_INTRIN_DYN, *
    get_ldmatrix_intrin(16, "float16", True, True, "shared.dyn")
)


TRICKY_MMA_f16f16f16_INTRIN = "TRICKY_mma_f16f16f16"
TensorIntrin.register(TRICKY_MMA_f16f16f16_INTRIN, *
                      get_mma_intrin(16, "float16", False))

TRICKY_MMA_f16f16f16_TRANS_INTRIN = "TRICKY_mma_f16f16f16_trans"
TensorIntrin.register(TRICKY_MMA_f16f16f16_TRANS_INTRIN, *
                      get_mma_intrin(16, "float16", True))

TRICKY_MMA_fill_16x16_f16_INTRIN = "TRICKY_mma_fill_16x16_f16"
TensorIntrin.register(TRICKY_MMA_fill_16x16_f16_INTRIN, *
                      get_mma_fill_intrin("float16", 8))

TRICKY_MMA_store_16x16_f16_global_INTRIN = "TRICKY_mma_store_16x16_f16_global_"
TensorIntrin.register(
    TRICKY_MMA_store_16x16_f16_global_INTRIN, *
    get_mma_store_intrin("float16", 8, "global")
)

TRICKY_MMA_store_16x16_f16_shared_INTRIN = "TRICKY_mma_store_16x16_f16_shared"
TensorIntrin.register(
    TRICKY_MMA_store_16x16_f16_shared_INTRIN, *
    get_mma_store_intrin("float16", 8, "shared")
)


TRICKY_MMA_store_16x16_f16_shared_INTRIN_DYN = "TRICKY_mma_store_16x16_f16_shared_DYN"
TensorIntrin.register(
    TRICKY_MMA_store_16x16_f16_shared_INTRIN_DYN, *
    get_mma_store_intrin("float16", 8, "shared.dyn")
)


def get_aync_copy_intrin(dtype, scope="shared"):
    if dtype == "float32":
        elems = 4
    elif dtype == "float16":
        elems = 8
    elif dtype == "int8":
        elems = 16
    else:
        raise ValueError("Unsupported dtype: {}".format(dtype))

    @T.prim_func
    def async_copy_desc(global_handle: T.handle, shared_handle: T.handle) -> None:
        globalVar = T.match_buffer(
            global_handle,
            (elems),
            dtype,
            align=64,
            offset_factor=elems,
            scope="global",
        )
        sharedVar = T.match_buffer(
            shared_handle, (elems), dtype, align=64, offset_factor=16, scope=scope
        )

        with T.block("root"):
            T.reads(globalVar[0:elems])
            T.writes(sharedVar[0:elems])

            for ax0 in T.vectorized(elems):
                with T.block("shared_warp"):
                    v0 = T.axis.remap("S", [ax0])
                    T.reads(globalVar[v0])
                    T.writes(sharedVar[v0])
                    sharedVar[v0] = globalVar[v0]

    @T.prim_func
    def async_copy_imlp(global_handle: T.handle, shared_handle: T.handle) -> None:
        globalVar = T.match_buffer(
            global_handle,
            (elems),
            dtype,
            align=64,
            offset_factor=elems,
            scope="global",
        )
        sharedVar = T.match_buffer(
            shared_handle, (elems), dtype, align=64, offset_factor=elems, scope=scope
        )

        with T.block("root"):
            T.reads(globalVar[0:elems])
            T.writes(sharedVar[0:elems])
            T.attr(0, "async_scope", 1)
            for ax0 in T.vectorized(elems):
                with T.block("shared_warp"):
                    v0 = T.axis.remap("S", [ax0])
                    T.reads(globalVar[v0])
                    T.writes(sharedVar[v0])
                    sharedVar[v0] = globalVar[v0]

    return async_copy_desc, async_copy_imlp


ASYNC_COPY_F16_X8_INTRIN = "async_copy.f16._x8"
TensorIntrin.register(ASYNC_COPY_F16_X8_INTRIN, *
                      get_aync_copy_intrin("float16"))

ASYNC_COPY_S8_X16_INTRIN = "async_copy.s8._x16"
TensorIntrin.register(ASYNC_COPY_S8_X16_INTRIN, *
                      get_aync_copy_intrin("int8"))

ASYNC_COPY_F16_X8_INTRIN_DYN = "async_copy.f16._x8_DYN"
TensorIntrin.register(ASYNC_COPY_F16_X8_INTRIN_DYN, *
                      get_aync_copy_intrin("float16", scope="shared.dyn"))

ASYNC_COPY_S8_X16_INTRIN_DYN = "async_copy.s8._x16_DYN"
TensorIntrin.register(ASYNC_COPY_S8_X16_INTRIN_DYN, *
                      get_aync_copy_intrin("int8", scope="shared.dyn"))
