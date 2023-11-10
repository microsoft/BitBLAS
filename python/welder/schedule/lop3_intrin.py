# registeration
import tvm
from tvm.runtime import convert
from tvm.tir.function import TensorIntrin
from tvm.script import tir as T
lift = convert

def get_fast_decode_intrin(storage_nbit=4, storage_dtype="int8", target_dtype="float16", loops_extent=8):
    if target_dtype == "float16":
        d4f = "f16"
    elif target_dtype == "int8":
        d4f = "i8s"
    else:
        raise ValueError("Unsupported target dtype: {}".format(target_dtype))
    func_name = "decode_i{}s_to_{}".format(storage_nbit, d4f)
    def _tir_u8_to_int_to_float(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return ((val >> (pos * nbit).astype("int8")) & mask).astype(dtype)
    
    assert storage_dtype == "int8"
    elem_per_i8 = 8 // storage_nbit
    n_storage_elems = loops_extent // elem_per_i8
    @T.prim_func
    def fast_decode_desc(compressed: T.handle, decompressed: T.handle) -> None:
        Compressed = T.match_buffer(
            compressed, [n_storage_elems,], dtype=storage_dtype, scope="local"
        )
        Decompressed = T.match_buffer(
            decompressed, [loops_extent,], dtype=target_dtype, scope="local"
        )
    
        with T.block("root"):
            T.reads(Compressed[0:n_storage_elems])
            T.writes(Decompressed[0:loops_extent])
            for i in T.grid(loops_extent):
                with T.block("decode"):
                    vi = T.axis.remap("S", [i])
                    Decompressed[vi] = _tir_u8_to_int_to_float(storage_nbit, Compressed[vi // elem_per_i8], vi % elem_per_i8, dtype=target_dtype)

    @T.prim_func
    def fast_decode_impl(compressed: T.handle, decompressed: T.handle) -> None:
        Compressed = T.match_buffer(
            compressed, [n_storage_elems,], dtype=storage_dtype, scope="local"
        )
        Decompressed = T.match_buffer(
            decompressed, [loops_extent,], dtype=target_dtype, scope="local"
        )
    
        with T.block("root"):
            T.reads(Compressed[0:n_storage_elems])
            T.writes(Decompressed[0:loops_extent])
            T.call_extern("handle", func_name, Compressed.data, Decompressed.data, loops_extent)
    
    return fast_decode_desc, fast_decode_impl


LOP3_FAST_DECODE_INT4_TO_FP16_INTRIN = "lop3_fast_decode_int4_to_fp16"
TensorIntrin.register(
    LOP3_FAST_DECODE_INT4_TO_FP16_INTRIN, *
    get_fast_decode_intrin(storage_nbit=4, storage_dtype="int8", target_dtype="float16")
)

LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN = "lop3_fast_decode_int4_to_int8"
TensorIntrin.register(
    LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN, *
    get_fast_decode_intrin(storage_nbit=4, storage_dtype="int8", target_dtype="int8")
)

LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN_L16 = "lop3_fast_decode_int4_to_int8_l16"
TensorIntrin.register(
    LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN_L16, *
    get_fast_decode_intrin(storage_nbit=4, storage_dtype="int8", target_dtype="int8", loops_extent=16)
)

LOP3_FAST_DECODE_INT2_TO_INT8_INTRIN_L8 = "lop3_fast_decode_int2_to_int8_l8"
TensorIntrin.register(
    LOP3_FAST_DECODE_INT2_TO_INT8_INTRIN_L8, *
    get_fast_decode_intrin(storage_nbit=2, storage_dtype="int8", target_dtype="int8", loops_extent=8)
)

LOP3_FAST_DECODE_INT2_TO_INT8_INTRIN_L16 = "lop3_fast_decode_int2_to_int8_l16"
TensorIntrin.register(
    LOP3_FAST_DECODE_INT2_TO_INT8_INTRIN_L16, *
    get_fast_decode_intrin(storage_nbit=2, storage_dtype="int8", target_dtype="int8", loops_extent=16)
)

LOP3_FAST_DECODE_INT1_TO_INT8_INTRIN_L16 = "lop3_fast_decode_int1_to_int8_l16"
TensorIntrin.register(
    LOP3_FAST_DECODE_INT1_TO_INT8_INTRIN_L16, *
    get_fast_decode_intrin(storage_nbit=1, storage_dtype="int8", target_dtype="int8", loops_extent=16)
)


LOP3_FAST_DECODE_INT1_TO_INT8_INTRIN_L32 = "lop3_fast_decode_int1_to_int8_l32"
TensorIntrin.register(
    LOP3_FAST_DECODE_INT1_TO_INT8_INTRIN_L32, *
    get_fast_decode_intrin(storage_nbit=1, storage_dtype="int8", target_dtype="int8", loops_extent=32)
)
