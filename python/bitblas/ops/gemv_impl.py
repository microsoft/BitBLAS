# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# pre-transformed tir expression of gemv
import tvm
from tvm.script import tir as T
from tvm import te


def gemv_i4(M, N, K, dtype="float16"):
    bit = 4
    n_float_per_i8 = 8 // bit

    def _tir_u8_to_int_to_float(
        nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str
    ):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return ((val >> (pos * nbit).astype("int8")) & mask).astype(dtype)

    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((N, K // 8 * bit), name="B", dtype="int8")

    def decode_func(n, k):
        w = _tir_u8_to_int_to_float(
            bit, B[n, k // n_float_per_i8], k % n_float_per_i8, dtype=dtype
        )
        return w

    B_decode = te.compute((N, K), decode_func, name="B_decode")

    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N), lambda i, j: te.sum(A[i, k] * B_decode[j, k], axis=k), name="C"
    )
    func = te.create_prim_func([A, B, C]).with_attr(
        "dequantize_info",
        {
            "B": {
                "decode_block": "B_decode",
                "fast_decoding": True,
                "source_format": {
                    "bits": 4,
                    "format": "int",
                },
                "target_format": {
                    "bits": 16,
                    "format": "float",
                },
            }
        },
    )
    return tvm.IRModule.from_expr(func)


def gemv(M, N, K, dtype="float16"):
    @tvm.script.ir_module
    class GEMV:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, [M, K], dtype=dtype)
            B = T.match_buffer(b, [N, K], dtype=dtype)
            C = T.match_buffer(c, [M, N], dtype=dtype)

            for i, j, k in T.grid(M, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = 0.0
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    return GEMV
