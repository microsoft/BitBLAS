# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tvm
from tvm.tir.function import TensorIntrin
from tvm.script import tir as T
from typing import Dict, Literal
from bitblas.quantization import (
    _tir_packed_int_to_int_to_float,
    _tir_packed_uint_to_uint_to_float,
)


decode_i4_to_f16 = """
template <typename T1, typename T2, bool isSigned = false>
__device__ void decode_i4b_to_f16(T1 *_i4s, T2 *B_local_decode, const int N = 8)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x000f000f;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint MEDIAN_NUM = isSigned ? 0x64076407 : 0x64006400;
    uint const i4s = *reinterpret_cast<uint *>(_i4s);
#pragma unroll
    for (int i = 0; i < (N / 2); i++)
    {

        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(h[i])
                     : "r"(i4s >> (4 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
    }
}

template <typename T1, typename T2>
__device__ void decode_i4s_to_f16(T1 *_i4s, T2 *B_local_decode, const int N = 8)
{
    decode_i4b_to_f16<T1, T2, true>(_i4s, B_local_decode, N);
}

template <typename T1, typename T2>
__device__ void decode_i4u_to_f16(T1 *_i4u, T2 *B_local_decode, const int N = 8)
{
    decode_i4b_to_f16<T1, T2, false>(_i4u, B_local_decode, N);
}
"""

decode_i4_to_f16_scale = """
template <typename T1, typename T2, typename T3, bool isSigned = false>
__device__ void decode_i4b_to_f16_scale(T1 *_i4s, T2 *B_local_decode, T3 *scale, const int N = 8)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x000f000f;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint MEDIAN_NUM = isSigned ? 0x64076407 : 0x64006400;
    uint const i4s = *reinterpret_cast<uint *>(_i4s);
#pragma unroll
    for (int i = 0; i < (N / 2); i++)
    {

        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(h[i])
                     : "r"(i4s >> (4 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
        unsigned v0 = *((unsigned short *)scale);
        unsigned v1 = *((unsigned short *)scale);
        unsigned __packed_scale = (v1 << 16) | v0;
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(__packed_scale), "r"(0));
    }
    
}

template <typename T1, typename T2, typename T3>
__device__ void decode_i4s_to_f16_scale(T1 *_i4s, T2 *B_local_decode, T3 *scale, const int N = 8)
{
    decode_i4b_to_f16_scale<T1, T2, T3, true>(_i4s, B_local_decode, scale, N);
}

template <typename T1, typename T2, typename T3>
__device__ void decode_i4u_to_f16_scale(T1 *_i4u, T2 *B_local_decode,  T3 *scale, const int N = 8)
{
    decode_i4b_to_f16_scale<T1, T2, T3, false>(_i4u, B_local_decode, scale, N);
}
"""

decode_i1s_to_i8s_l16 = """template <typename T1, typename T2>
__device__ void decode_i1s_to_i8s_l16(T1 *_i1s, T2 *_i8s, const int N = 16)
{
  int *i8s = reinterpret_cast<int *>(_i8s);
  int16_t i1s_i16 = *reinterpret_cast<int16_t *>(_i1s);
  // permutate: {e0,e4,e8,e12,e2,e6,e10,e14,e1,e5,e9,e13,e3,e7,e11,e15}
  // into: {e0,e4,e8,e12,x,x,x,x,e1,e5,e9,x,x,x,x,e13,e2,e6,e10,e14,e1,e5,e9,e13,e3,e7,e11,e15,x,x,x,x}
  int i1s = (i1s_i16 & 0x0f0f);
  i1s |= ((i1s_i16 & 0xf0f0) << 12); 
  // i1s        {0..,e15,e14,e13,e12,e11,e10,e9,e8,e7,e6,e5,e4,e3,e2,e1,e0}
  // interleave {0..,e15,e13,e11,e9,e7,e5,e3,e1,e14,e12,e10,e8,e6,e4,e2,e0}
  // First, we extract the i1s and construct an intermediate fp16 number.
  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa; // 0b11101010
  static constexpr uint BOTTOM_MASK = 0x01010101;      // 0x1 -> 0b01 select 0,1
  static constexpr uint I8s_MAGIC_NUM = 0x00000000;

  for (int i = 0; i < N / 4; i++)
  {
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                 : "=r"(i8s[i])
                 : "r"(i1s >> i), "n"(BOTTOM_MASK), "n"(I8s_MAGIC_NUM), "n"(immLut));
  }
}
"""

decode_i2s_to_i8s = """template <typename T1, typename T2>
__device__ void decode_i2s_to_i8s(T1 *_i2s, T2 *_i8s, const int N = 16)
{
  // convert 8 int2b_t to 8 int8b_t -> 2 int32
  uint *i8s = reinterpret_cast<uint *>(_i8s);

  // i2s = {e7,e6,e5,e4,e3,e2,e1,e0}
  // also require interleave {e7,e3,e6,e2,e5,e1,e4,e0}
  uint const i2s = *_i2s;

  // First, we extract the i4s and construct an intermediate fp16 number.
  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;     // 0b11101010
  static constexpr uint BOTTOM_MASK = 0x03030303;          // 0xf -> 0b11 select 0,3
  static constexpr uint I4s_TO_INT8_TO_I8s_MAGIC_NUM = 0x00000000; // 1024

#pragma unroll
  for (int i = 0; i < (N / 2); i++)
  {
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                 : "=r"(i8s[i])
                 : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(I4s_TO_INT8_TO_I8s_MAGIC_NUM), "n"(immLut));
  }
}
"""

decode_i4s_to_i8s = """template <typename T1, typename T2>
__device__ void decode_i4s_to_i8s(T1 *_i4s, T2 *_i8s, const int N = 16)
{
  uint *i8s = reinterpret_cast<uint *>(_i8s);
  uint *i4s = reinterpret_cast<uint *>(_i4s);
  // First, we extract the i4s and construct an intermediate fp16 number.
  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;     // 0b11101010
  static constexpr uint BOTTOM_MASK = 0x0f0f0f0f;          // 0xf -> 0b1111 select 0,4
  static constexpr uint I4s_TO_INT8_TO_I8s_MAGIC_NUM = 0x00000000; // 1024
#pragma unroll
  for (int i = 0; i < (N / 8); i++)
  {
    // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                 : "=r"(i8s[i])
                 : "r"(i4s[0] >> (4 * i)), "n"(BOTTOM_MASK), "n"(I4s_TO_INT8_TO_I8s_MAGIC_NUM), "n"(immLut));

    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
              : "=r"(i8s[i + 2])
              : "r"(i4s[1] >> (4 * i)), "n"(BOTTOM_MASK), "n"(I4s_TO_INT8_TO_I8s_MAGIC_NUM), "n"(immLut));
  }
}
"""


def get_fast_decode_intrin(
    source_bit=4,
    storage_dtype="int8",
    source_format="int",
    target_dtype="float16",
    loops_extent=8,
    with_scale=False,
):
    """
    loops extent is the number of elements to be decoded in one stage
    for memory friendly process, the loops_extent should be a multiple of (sizeof(int) // 8).
    However, for the case of int1b, it is not possible to decode 8 elements in one stage, so we have to use 16.
    """
    if target_dtype == "float16":
        d4f = "f16"
    elif target_dtype == "int8":
        d4f = "i8s"
    else:
        raise ValueError("Unsupported target dtype: {}".format(target_dtype))
    source_symbol = "u" if source_format == "uint" else "i"
    func_name = "decode_{}{}s_to_{}".format(source_symbol, source_bit, d4f)
    if with_scale:
        func_name += "_scale"

    assert storage_dtype in ["int8", "int32", "uint32"]
    storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
    elem_per_unit = storage_nbit // source_bit
    n_storage_elems = loops_extent // elem_per_unit

    if storage_dtype[:3] == "int":
        decode_func = _tir_packed_int_to_int_to_float(storage_nbit)
    elif storage_dtype[:4] == "uint":
        decode_func = _tir_packed_uint_to_uint_to_float(storage_nbit)
    else:
        raise ValueError("Unsupported storage dtype: {}".format(storage_dtype))

    if with_scale is False:

        @T.prim_func
        def fast_decode_desc(compressed: T.handle, decompressed: T.handle) -> None:
            Compressed = T.match_buffer(
                compressed,
                [
                    n_storage_elems,
                ],
                dtype=storage_dtype,
                scope="local",
            )
            Decompressed = T.match_buffer(
                decompressed,
                [
                    loops_extent,
                ],
                dtype=target_dtype,
                scope="local",
            )

            with T.block("root"):
                T.reads(Compressed[0:n_storage_elems])
                T.writes(Decompressed[0:loops_extent])
                for i in T.grid(loops_extent):
                    with T.block("decode"):
                        vi = T.axis.remap("S", [i])
                        Decompressed[vi] = decode_func(
                            source_bit,
                            Compressed[vi // elem_per_unit],
                            vi % elem_per_unit,
                            dtype=target_dtype,
                        )

        @T.prim_func
        def fast_decode_impl(compressed: T.handle, decompressed: T.handle) -> None:
            Compressed = T.match_buffer(
                compressed,
                [
                    n_storage_elems,
                ],
                dtype=storage_dtype,
                scope="local",
            )
            Decompressed = T.match_buffer(
                decompressed,
                [
                    loops_extent,
                ],
                dtype=target_dtype,
                scope="local",
            )

            with T.block("root"):
                T.reads(Compressed[0:n_storage_elems])
                T.writes(Decompressed[0:loops_extent])
                T.call_extern(
                    "handle",
                    func_name,
                    Compressed.data,
                    Decompressed.data,
                    loops_extent,
                )

    else:

        @T.prim_func
        def fast_decode_desc(
            compressed: T.handle, decompressed: T.handle, scale: T.handle
        ) -> None:
            Compressed = T.match_buffer(
                compressed,
                [
                    n_storage_elems,
                ],
                dtype=storage_dtype,
                scope="local",
            )
            Decompressed = T.match_buffer(
                decompressed,
                [
                    loops_extent,
                ],
                dtype=target_dtype,
                scope="local",
            )
            Scale = T.match_buffer(
                scale,
                [
                    1,
                ],
                dtype=target_dtype,
            )
            with T.block("root"):
                T.reads(Compressed[0:n_storage_elems], Scale[0:1])
                T.writes(Decompressed[0:loops_extent])
                for i in T.grid(loops_extent):
                    with T.block("decode"):
                        vi = T.axis.remap("S", [i])
                        Decompressed[vi] = (
                            decode_func(
                                source_bit,
                                Compressed[vi // elem_per_unit],
                                vi % elem_per_unit,
                                dtype=target_dtype,
                            )
                            * Scale[0]
                        )

        @T.prim_func
        def fast_decode_impl(
            compressed: T.handle, decompressed: T.handle, scale: T.handle
        ) -> None:
            s0 = T.int32()

            Compressed = T.match_buffer(
                compressed,
                [
                    n_storage_elems,
                ],
                dtype=storage_dtype,
                scope="local",
            )
            Decompressed = T.match_buffer(
                decompressed,
                [
                    loops_extent,
                ],
                dtype=target_dtype,
                scope="local",
            )
            Scale = T.match_buffer(
                scale,
                [
                    1,
                ],
                dtype=target_dtype,
                offset_factor=1,
                strides=[s0],
            )
            with T.block("root"):
                T.reads(Compressed[0:n_storage_elems], Scale[0:1])
                T.writes(Decompressed[0:loops_extent])
                T.call_extern(
                    "handle",
                    func_name,
                    Compressed.data,
                    Decompressed.data,
                    Scale.access_ptr("r"),
                    loops_extent,
                )

    return fast_decode_desc, fast_decode_impl


LOP3_FAST_DECODE_UINT4_TO_INT8_TO_FP16_L8_INTRIN = (
    "lop3_fast_decode_u4_to_int8_to_f16_l8_"
)
TensorIntrin.register(
    LOP3_FAST_DECODE_UINT4_TO_INT8_TO_FP16_L8_INTRIN,
    *get_fast_decode_intrin(
        source_bit=4, storage_dtype="int8", target_dtype="float16", loops_extent=8
    ),
)


LOP3_FAST_DECODE_UINT4_TO_INT32_TO_FP16_L8_INTRIN = (
    "lop3_fast_decode_u4_to_int32_to_f16_l8_"
)
TensorIntrin.register(
    LOP3_FAST_DECODE_UINT4_TO_INT32_TO_FP16_L8_INTRIN,
    *get_fast_decode_intrin(
        source_bit=4, storage_dtype="int32", target_dtype="float16", loops_extent=8
    ),
)


LOP3_FAST_DECODE_UINT4_TO_INT32_TO_FP16_L8_SCALE_INTRIN = (
    "lop3_fast_decode_u4_to_int32_to_f16_l8_scale_"
)
TensorIntrin.register(
    LOP3_FAST_DECODE_UINT4_TO_INT32_TO_FP16_L8_SCALE_INTRIN,
    *get_fast_decode_intrin(
        source_bit=4,
        storage_dtype="int32",
        target_dtype="float16",
        loops_extent=8,
        with_scale=True,
    ),
)

LOP3_FAST_DECODE_UINT4_TO_UINT32_TO_FP16_L8_INTRIN = (
    "lop3_fast_decode_u4_to_uint32_to_f16_l8_"
)
TensorIntrin.register(
    LOP3_FAST_DECODE_UINT4_TO_UINT32_TO_FP16_L8_INTRIN,
    *get_fast_decode_intrin(
        source_bit=4, storage_dtype="uint32", target_dtype="float16", loops_extent=8
    ),
)


LOP3_FAST_DECODE_UINT4_TO_UINT32_TO_FP16_L8_SCALE_INTRIN = (
    "lop3_fast_decode_u4_to_uint32_to_f16_l8_scale_"
)
TensorIntrin.register(
    LOP3_FAST_DECODE_UINT4_TO_UINT32_TO_FP16_L8_SCALE_INTRIN,
    *get_fast_decode_intrin(
        source_bit=4,
        storage_dtype="uint32",
        target_dtype="float16",
        loops_extent=8,
        with_scale=True,
    ),
)


LOP3_FAST_DECODE_UINT4_TO_INT8_TO_FP16_L8_SCALE_INTRIN = (
    "lop3_fast_decode_u4_to_int8_to_f16_l8_scale_"
)
TensorIntrin.register(
    LOP3_FAST_DECODE_UINT4_TO_INT8_TO_FP16_L8_SCALE_INTRIN,
    *get_fast_decode_intrin(
        source_bit=4,
        storage_dtype="int8",
        target_dtype="float16",
        loops_extent=8,
        with_scale=True,
    ),
)

LOP3_FAST_DECODE_UINT4_TO_INT8_TO_INT8_L8_INTRIN = (
    "lop3_fast_decode_u4_to_int8_to_i8_l8_"
)
TensorIntrin.register(
    LOP3_FAST_DECODE_UINT4_TO_INT8_TO_INT8_L8_INTRIN,
    *get_fast_decode_intrin(
        source_bit=4, storage_dtype="int8", target_dtype="int8", loops_extent=8
    ),
)

LOP3_FAST_DECODE_UINT4_TO_INT8_TO_INT8_L16_INTRIN = (
    "lop3_fast_decode_u4_to_int8_to_i8_l16_"
)
TensorIntrin.register(
    LOP3_FAST_DECODE_UINT4_TO_INT8_TO_INT8_L16_INTRIN,
    *get_fast_decode_intrin(
        source_bit=4, storage_dtype="int8", target_dtype="int8", loops_extent=16
    ),
)

LOP3_FAST_DECODE_UINT2_TO_INT8_TO_INT8_L16_INTRIN = (
    "lop3_fast_decode_u2_to_int8_to_i8_l16_"
)
TensorIntrin.register(
    LOP3_FAST_DECODE_UINT2_TO_INT8_TO_INT8_L16_INTRIN,
    *get_fast_decode_intrin(
        source_bit=2, storage_dtype="int8", target_dtype="int8", loops_extent=16
    ),
)

LOP3_FAST_DECODE_INT2_TO_INT8_TO_INT8_L16_INTRIN = (
    "lop3_fast_decode_i2_to_int8_to_i8_l16_"
)
TensorIntrin.register(
    LOP3_FAST_DECODE_INT2_TO_INT8_TO_INT8_L16_INTRIN,
    *get_fast_decode_intrin(
        source_bit=2, storage_dtype="int8", target_dtype="int8", loops_extent=16
    ),
)

LOP3_FAST_DECODE_UINT1_TO_INT8_TO_INT8_L16_INTRIN = (
    "LOP3_FAST_DECODE_UINT1_to_int8_to_i8_l16_"
)
TensorIntrin.register(
    LOP3_FAST_DECODE_UINT1_TO_INT8_TO_INT8_L16_INTRIN,
    *get_fast_decode_intrin(
        source_bit=1, storage_dtype="int8", target_dtype="int8", loops_extent=16
    ),
)

LOP3_FAST_DECODE_INT4_TO_INT8_TO_FP16_L8_INTRIN = (
    "lop3_fast_decode_i4_to_int8_to_f16_l8_"
)
TensorIntrin.register(
    LOP3_FAST_DECODE_INT4_TO_INT8_TO_FP16_L8_INTRIN,
    *get_fast_decode_intrin(
        source_bit=4,
        storage_dtype="int8",
        source_format="int",
        target_dtype="float16",
        loops_extent=8,
    ),
)

LOP3_FAST_DECODE_INT4_TO_INT8_TO_FP16_L8_SCALE_INTRIN = (
    "lop3_fast_decode_i4_to_int8_to_f16_l8_scale_"
)
TensorIntrin.register(
    LOP3_FAST_DECODE_INT4_TO_INT8_TO_FP16_L8_SCALE_INTRIN,
    *get_fast_decode_intrin(
        source_bit=4,
        storage_dtype="int8",
        source_format="int",
        target_dtype="float16",
        loops_extent=8,
        with_scale=True,
    ),
)


def get_lop3_intrin_group(
    out_dtype: Literal["float16", "int8"],
    source_format: Literal["int", "uint"] = "uint",
    source_bit: int = 4,
    storage_dtype: Literal["int32", "int8"] = "int8",
    with_scaling: bool = False,
) -> Dict[str, str]:
    """
    This function is used to get the intrinsic group of the LOP3 operation to avoid the overhead of fast decoding.
    LOP3 is a type of logic operation that takes three inputs. The intrinsic group refers to the set of
    intrinsic operations that can be performed on these inputs. This function retrieves and returns this group.

    Parameters
    ----------
    in_dtype : Literal["int8"]
        The data type of the input. It should be "int8".

    out_dtype : Literal["float16", "int8"]
        The data type of the output. It can be either "float16" or "int8".

    storage_nbit : int, optional
        The number of bits used for storage. By default, it is 4.

    with_scale : bool, optional
        A boolean parameter that indicates whether scaling should be applied. By default, it is False.

    Returns
    -------
    Dict[str, str]
        A dictionary mapping the names of the intrinsics to their corresponding implementations.
    """
    assert out_dtype in ["float16", "int8"]

    dtype_mapping = {"float16": "f16", "int8": "i8", "int32": "i32"}
    target_dtype = dtype_mapping[out_dtype]
    target_bits = tvm.DataType(out_dtype).bits
    loop_extent = 128 // target_bits
    if source_format not in ["int", "uint"]:
        raise ValueError("Invalid source_format. Expected 'int' or 'uint'.")
    source_symbol = "i" if source_format == "int" else "u"

    _intrin = f"lop3_fast_decode_{source_symbol}{source_bit}_to_{storage_dtype}_to_{target_dtype}_l{loop_extent}_"
    if with_scaling:
        _intrin += "scale_"

    import_c_map = {
        "i4_to_f16": decode_i4_to_f16,
        "i4_to_f16_scale": decode_i4_to_f16_scale,
        "i1_to_i8": decode_i1s_to_i8s_l16,
        "i2_to_i8": decode_i2s_to_i8s,
        "i4_to_i8": decode_i4s_to_i8s,
    }
    key = f"i{source_bit}_to_{target_dtype}"
    if with_scaling:
        key += "_scale"

    return {
        "c_source": import_c_map[key],
        "compute": _intrin,
    }
