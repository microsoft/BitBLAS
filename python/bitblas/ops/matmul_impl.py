# pre-transformed tir expression of matmul
import tvm
from tvm.script import tir as T
from tvm import te


def matmul_nt_dyn_m(N, K, in_dtype="float16", out_dtype="float16"):
    @tvm.script.ir_module
    class MatmulNT:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(a, [m, K], dtype=in_dtype)
            B = T.match_buffer(b, [N, K], dtype=in_dtype)
            C = T.match_buffer(c, [m, N], dtype=out_dtype)

            for i, j, k in T.grid(m, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = tvm.tir.const(0, out_dtype)
                    C[vi, vj] = C[vi, vj] + A[vi, vk].astype(out_dtype) * B[
                        vj, vk
                    ].astype(out_dtype)

    return MatmulNT


def matmul_nn_dyn_m(N, K, in_dtype="float16", out_dtype="float16"):
    @tvm.script.ir_module
    class MatmulNN:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(a, [m, K], dtype=in_dtype)
            B = T.match_buffer(b, [K, N], dtype=in_dtype)
            C = T.match_buffer(c, [m, N], dtype=out_dtype)

            for i, j, k in T.grid(m, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = tvm.tir.const(0, out_dtype)
                    C[vi, vj] = C[vi, vj] + A[vi, vk].astype(out_dtype) * B[
                        vk, vj
                    ].astype(out_dtype)

    return MatmulNN


def matmul_nn(M, N, K, in_dtype="float16", out_dtype="float16"):
    @tvm.script.ir_module
    class MatmulNN:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, [M, K], dtype=in_dtype)
            B = T.match_buffer(b, [K, N], dtype=in_dtype)
            C = T.match_buffer(c, [M, N], dtype=out_dtype)

            for i, j, k in T.grid(M, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = tvm.tir.const(0, out_dtype)
                    C[vi, vj] = C[vi, vj] + A[vi, vk].astype(out_dtype) * B[
                        vk, vj
                    ].astype(out_dtype)

    return MatmulNN

def matmul_nt(M, N, K, in_dtype="float16", out_dtype="float16"):
    @tvm.script.ir_module
    class MatmulNT:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, [M, K], dtype=in_dtype)
            B = T.match_buffer(b, [N, K], dtype=in_dtype)
            C = T.match_buffer(c, [M, N], dtype=out_dtype)

            for i, j, k in T.grid(M, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = tvm.tir.const(0, out_dtype)
                    C[vi, vj] = C[vi, vj] + A[vi, vk].astype(out_dtype) * B[
                        vj, vk
                    ].astype(out_dtype)

    return MatmulNT

def matmul_nt_propagate_b_f16_f16_mma(M, N, K, in_dtype="float16", out_dtype="float16"):
    wm, wn, wk = 16, 16, 16
    if in_dtype == "int8":
        wm, wn, wk = 16, 16, 32

    @tvm.script.ir_module
    class MyModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {"global_symbol": "main", "tir.noalias": True, "smooth_b": True}
            )
            A = T.match_buffer(a, [M, K], dtype=in_dtype)
            B = T.match_buffer(b, [N // wn, K // wk, wn, wk], dtype=in_dtype)
            C = T.match_buffer(c, [M, N], dtype=out_dtype)
            B_reindex = T.alloc_buffer([N, K], dtype=in_dtype)

            for j, k in T.grid(N, K):
                with T.block("B_reindex"):
                    vj, vk = T.axis.remap("SS", [j, k])
                    B_reindex[vj, vk] = B[
                        vj // wn,
                        vk // wk,
                        vj % wn // 8 * 8 + vj % 4 * 2 + vk % wn // 8,
                        vj % 8 // 4 * 8 + vk % 8,
                    ]

            for i, j, k in T.grid(M, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = tvm.tir.const(0, out_dtype)
                    C[vi, vj] = C[vi, vj] + A[vi, vk].astype(out_dtype) * B_reindex[
                        vj, vk
                    ].astype(out_dtype)

    return MyModule

def matmul_nt_propagate_a_b(M, N, K, in_dtype="float16", out_dtype="float16"):
    wm, wn, wk = 16, 16, 16
    if in_dtype == "int8":
        wm, wn, wk = 16, 16, 32

    @tvm.script.ir_module
    class MyModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr(
                {
                    "global_symbol": "main",
                    "tir.noalias": True,
                    "smooth_a": True,
                    "smooth_b": True,
                }
            )
            A = T.match_buffer(a, [M // wm, K // wk, wm, wk], dtype=in_dtype)
            B = T.match_buffer(b, [N // wn, K // wk, wn, wk], dtype=in_dtype)
            C = T.match_buffer(c, [M, N], dtype=out_dtype)
            A_reindex = T.alloc_buffer([M, K], dtype=in_dtype)
            B_reindex = T.alloc_buffer([N, K], dtype=in_dtype)

            for i, k in T.grid(M, K):
                with T.block("A_reindex"):
                    vj, vk = T.axis.remap("SS", [i, k])
                    A_reindex[vj, vk] = A[vj // wm, vk // wk, vj % wm, vk % wk]

            for j, k in T.grid(N, K):
                with T.block("B_reindex"):
                    vj, vk = T.axis.remap("SS", [j, k])
                    B_reindex[vj, vk] = B[vj // wn, vk // wk, vj % wn, vk % wk]

            for i, j, k in T.grid(M, N, K):
                with T.block("C"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = tvm.tir.const(0, out_dtype)
                    C[vi, vj] = C[vi, vj] + A_reindex[vi, vk].astype(
                        out_dtype
                    ) * B_reindex[vj, vk].astype(out_dtype)

    return MyModule


def matmul_nt_i4(M, N, K, in_dtype="float16", out_dtype="float16"):
    bit = 4
    n_float_per_i8 = 8 // bit

    def _tir_u8_to_int_to_float(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return ((val >> (pos * nbit).astype("int8")) & mask).astype(dtype)
    
    A = te.placeholder((M, K), name='A', dtype=in_dtype)
    B = te.placeholder((N, K // 8 * bit), name='B', dtype='int8')
    
    def decode_func(n, k):
        w = _tir_u8_to_int_to_float(bit, B[n, k // n_float_per_i8], k % n_float_per_i8, dtype=in_dtype)
        return w

    B_decode = te.compute(
        (N, K),
        decode_func,
        name='B_decode'
    )

    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name='k')
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B_decode[j, k], axis=k),
        name='C'
    )
    func = te.create_prim_func([A, B, C]).with_attr("dequantize_info", {
        'B': {
            'decode_block': 'B_decode',
            'fast_decoding': True,
            'source_format':{
                'bits': 4,
                'format': 'int',
            },
            'target_format': "float16"
        }
    })
    return tvm.IRModule.from_expr(func)


def matmul_nt_i4_propagate_b(M, N, K, in_dtype="float16", out_dtype="float16"):
    bit = 4
    n_float_per_i8 = 8 // bit

    def _tir_u8_to_int_to_float(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return ((val >> (pos * nbit).astype("int8")) & mask).astype(dtype)
    
    A = te.placeholder((M, K), name='A', dtype=in_dtype)
    B = te.placeholder((N // 16, K // 16, 16, 16 // 8 * bit), name='B', dtype='int8')
    
    def decode_func(n, k, nn, kk):
        w = _tir_u8_to_int_to_float(bit, B[n, k, nn, kk // n_float_per_i8], kk % n_float_per_i8, dtype=in_dtype)
        return w

    B_decode = te.compute(
        (N // 16, K // 16, 16, 16),
        decode_func,
        name='B_decode'
    )
    
    B_reindex = te.compute(
        (N, K),
        lambda i, j: B_decode[i // 16, j // 16, i % 16, j % 16],
        name="B_reindex"
    )

    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name='k')
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B_reindex[j, k], axis=k),
        name='C'
    )
    func = te.create_prim_func([A, B, C]).with_attr("dequantize_info", {
        'B': {
            'decode_block': 'B_decode',
            'fast_decoding': True,
            'source_format':{
                'bits': 4,
                'format': 'int',
            },
            'target_format': "float16"
        }
    })
    func = func.with_attr("smooth_b", True)

    return tvm.IRModule.from_expr(func)


def matmul_nt_i4_propagate_a_b(M, N, K, in_dtype="float16", out_dtype="float16"):
    bit = 4
    n_float_per_i8 = 8 // bit

    def _tir_u8_to_int_to_float(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return ((val >> (pos * nbit).astype("int8")) & mask).astype(dtype)
    
    A = te.placeholder((M // 16, K // 16, 16, 16), name='A', dtype=in_dtype)
    B = te.placeholder((N // 16, K // 16, 16, 16 // 8 * bit), name='B', dtype='int8')
    
    def decode_func(n, k, nn, kk):
        w = _tir_u8_to_int_to_float(bit, B[n, k, nn, kk // n_float_per_i8], kk % n_float_per_i8, dtype=in_dtype)
        return w

    B_decode = te.compute(
        (N // 16, K // 16, 16, 16),
        decode_func,
        name='B_decode'
    )
    
    B_reindex = te.compute(
        (N, K),
        lambda i, j: B_decode[i // 16, j // 16, i % 16, j % 16],
        name="B_reindex"
    )
    
    A_reindex = te.compute(
        (M, K),
        lambda i, j: A[i // 16, j // 16, i % 16, j % 16],
        name="A_reindex"
    )
    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name='k')
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A_reindex[i, k] * B_reindex[j, k], axis=k),
        name='C'
    )
    func = te.create_prim_func([A, B, C]).with_attr("dequantize_info", {
        'B': {
            'decode_block': 'B_decode',
            'fast_decoding': True,
            'source_format':{
                'bits': 4,
                'format': 'int',
            },
            'target_format': "float16"
        }
    })
    func = func.with_attr("smooth_a", True)
    func = func.with_attr("smooth_b", True)

    return tvm.IRModule.from_expr(func)


def matmul_nt_af4(M, N, K, in_dtype="float16", out_dtype="float16"):
    bit = 4
    n_float_per_i8 = 8 // bit

    def _tir_u8_to_int(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return (val >> (pos * nbit).astype("int8")) & mask
        
    A = te.placeholder((M, K), name='A', dtype=in_dtype)
    B = te.placeholder((N, K // 8 * bit), name='B', dtype='int8')
    LUT = te.placeholder((1 << bit, ), name='LUT', dtype='float16')


    def decode_func(n, k):
        w = _tir_u8_to_int(bit, B[n, k // n_float_per_i8], k % n_float_per_i8)
        return LUT[w]

    B_decode = te.compute(
        (N, K),
        decode_func,
        name='B_decode'
    )

    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name='k')
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B_decode[j, k], axis=k),
        name='C'
    )
    func = te.create_prim_func([A, B, LUT, C]).with_attr("dequantize_info", {
        'B': {
            'decode_block': 'B_decode',
            'source_format':{
                'bits': 4,
                'format': 'af',
            },
            'target_format': "float16"
        }
    })
    return tvm.IRModule.from_expr(func)

def matmul_nt_af4_propagate_a_b(M, N, K, in_dtype="float16", out_dtype="float16"):
    bit = 4
    n_float_per_i8 = 8 // bit

    def _tir_u8_to_int(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return (val >> (pos * nbit).astype("int8")) & mask
    
    A = te.placeholder((M // 16, K // 16, 16, 16), name='A', dtype=in_dtype)
    B = te.placeholder((N // 16, K // 16, 16, 16 // 8 * bit), name='B', dtype='int8')
    LUT = te.placeholder((1 << bit, ), name='LUT', dtype='float16')

    def decode_func(n, k, nn, kk):
        w = _tir_u8_to_int(bit, B[n, k, nn, kk // n_float_per_i8], kk % n_float_per_i8)
        return LUT[w]

    B_decode = te.compute(
        (N // 16, K // 16, 16, 16),
        decode_func,
        name='B_decode'
    )
    
    B_reindex = te.compute(
        (N, K),
        lambda i, j: B_decode[i // 16, j // 16, i % 16, j % 16],
        name="B_reindex"
    )
    
    A_reindex = te.compute(
        (M, K),
        lambda i, j: A[i // 16, j // 16, i % 16, j % 16],
        name="A_reindex"
    )
    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name='k')
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A_reindex[i, k] * B_reindex[j, k], axis=k),
        name='C'
    )
    func = te.create_prim_func([A, B, LUT, C]).with_attr("dequantize_info", {
        'B': {
            'decode_block': 'B_decode',
            'source_format':{
                'bits': 4,
                'format': 'af',
            },
            'target_format': "float16"
        }
    })
    func = func.with_attr("smooth_a", True)
    func = func.with_attr("smooth_b", True)

    return tvm.IRModule.from_expr(func)


# register the func 
matmul_impl_factory = {
    'matmul_nt': matmul_nt,
    'matmul_nt_dyn_m': matmul_nt_dyn_m,
    'matmul_nn': matmul_nn,
    'matmul_nn_dyn_m': matmul_nn_dyn_m,
    'matmul_nt_propagate_b_f16_f16_mma': matmul_nt_propagate_b_f16_f16_mma,
    'matmul_nt_propagate_a_b': matmul_nt_propagate_a_b,
    'matmul_nt_propagate_a_b_f16_f16_mma': matmul_nt_propagate_a_b,
}