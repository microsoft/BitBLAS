# registeration
import tvm
from tvm.runtime import convert
from tvm.tir.function import TensorIntrin
from tvm.script import tir as T
lift = convert

def get_fast_decode_intrin(storage_nbit=4, storage_dtype="int8", target_dtype="float16", with_scale=False, group_size=-1):
    
    def _tir_u8_to_int_to_float(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return ((val >> (pos * nbit).astype("int8")) & mask).astype(dtype)
    
    assert storage_dtype == "int8"
    elem_per_i8 = 8 // storage_nbit
    mask = (1 << storage_nbit) - 1
    @T.prim_func
    def fast_decode_desc(compressed: T.handle, decompressed: T.handle) -> None:
        Compressed = T.match_buffer(
            compressed, [4,], dtype=storage_dtype, scope="local"
        )
        Decompressed = T.match_buffer(
            decompressed, [8,], dtype=target_dtype, scope="local"
        )
    
        with T.block("root"):
            T.reads(Compressed[0:4])
            T.writes(Decompressed[0:8])
            for i in T.grid(8):
                with T.block("decode"):
                    vi = T.axis.remap("S", [i])
                    Decompressed[vi] = _tir_u8_to_int_to_float(storage_nbit, Compressed[vi // elem_per_i8], vi % elem_per_i8, dtype=target_dtype)

    @T.prim_func
    def fast_decode_impl(compressed: T.handle, decompressed: T.handle) -> None:
        Compressed = T.match_buffer(
            compressed, [4,], dtype=storage_dtype, scope="local"
        )
        Decompressed = T.match_buffer(
            decompressed, [8,], dtype=target_dtype, scope="local"
        )
    
        with T.block("root"):
            T.reads(Compressed[0:4])
            T.writes(Decompressed[0:8])
            T.call_extern("handle", "decode_i4s_to_f16", Compressed.data, Decompressed.data)
    
    return fast_decode_desc, fast_decode_impl


LOP3_FAST_DECODE_INT4_TO_FP16_INTRIN = "lop3_fast_decode_int4_to_fp16"
TensorIntrin.register(
    LOP3_FAST_DECODE_INT4_TO_FP16_INTRIN, *
    get_fast_decode_intrin(storage_nbit=4, storage_dtype="int8", target_dtype="float16")
)
