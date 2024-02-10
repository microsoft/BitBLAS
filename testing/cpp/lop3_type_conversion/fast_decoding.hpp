#include <cuda_runtime.h>

void general_compress(const int8_t *lowbit, int8_t *compressed, const int nbit, const int N, bool isSigned = false)
{
    int zero_point = isSigned ? ((1 << (nbit - 1)) - 1) : 0;
    const int nbit_per_byte = 8 / nbit;

    for (int i = 0; i < N / nbit_per_byte; i++)
    {
        compressed[i] = 0;
        for (int j = 0; j < nbit_per_byte; j++)
        {
            compressed[i] |= ((lowbit[nbit_per_byte * i + j] + zero_point) << (nbit * j));
        }
    }
}

void general_interleave_fp16(int8_t *origin_arr, int8_t *interleaved, const int nbit, size_t size_in_bytes, bool verbose = false)
{
    // For fp16 example
    // i4s        {e7,e6,e5,e4,e3,e2,e1,e0}
    //            |-8b-||-8b-||-8b-||-8b-|
    // interleave {e7,e5,e3,e1,e6,e4,e2,e0}
    /*
      BOTTOM_MASK        0    0    0    f    0    0    0    f
      i4s                e7   e5   e3   e1   e6   e4   e2   e0
      selectedVal       0000 0000 0000  e1  0000 0000 0000  e0  // selectedVal = i4s & BOTTOM_MASK
      h[0]              0110 0100  0    e1  0110 0100  0    e0  //  selectVal | 0x6400
    */
    // i2s        {e15,e14,e13,e12,e11,e10,e9,e8,e7,e6,e5,e4,e3,e2,e1,e0}
    // interleave {e15,e13,e11,e9,e7,e5,e3,e1,e14,e12,e10,e8,e6,e4,e2,e0}
    // i1s        {e31,e30,e29,e28,e27,e26,e25,e24,e23,e22,e21,e20,e19,e18,e17,e16,e15,e14,e13,e12,e11,e10,e9,e8,e7,e6,e5,e4,e3,e2,e1,e0}
    // interleave {e31,e29,e27,e25,e23,e21,e19,e17,e15,e13,e11,e9,e7,e5,e3,e1,e30,e28,e26,e24,e22,e20,e18,e16,e14,e12,e10,e8,e6,e4,e2,e0}
    // Assuming size is the number of int32 elements in origin_arr
    size_t size = size_in_bytes / sizeof(int32_t);
    int32_t *int32_origin = (int32_t *)origin_arr;
    int32_t *int32_interleaved = (int32_t *)interleaved;

    int mask = (1 << nbit) - 1;
    int num_groups = (32 / nbit) / 2;

    for (int idx = 0; idx < size; ++idx)
    {
        int32_t current_value = int32_origin[idx];
        int32_t new_value = 0;

        for (int i = 0; i < num_groups; ++i)
        {
            int left_shift = nbit * i;
            int right_shift = nbit * (num_groups - i - 1);
            new_value |= (current_value & (mask << nbit * (2 * i))) >> left_shift;
            new_value |= (current_value & (mask << nbit * (2 * i + 1))) << right_shift;
            if (verbose)
            {
                printf("put %d to %d\n", (2 * i), (nbit * (2 * i) - left_shift) / nbit);
                printf("put %d to %d\n", (2 * i + 1), (nbit * (2 * i + 1) + right_shift) / nbit);
            }
        }
        if (nbit == 2)
        {
            int32_t _new_value_n16 = (new_value & 0xff0000ff);
            _new_value_n16 |= ((new_value & 0x0000ff00) >> 8) << 16;
            _new_value_n16 |= ((new_value & 0x00ff0000) >> 16) << 8;
            int32_interleaved[idx] = _new_value_n16;
        }
        else if (nbit == 1)
        {
            int32_t _new_value_n16 = (new_value & 0xf000000f);
            _new_value_n16 |= ((new_value & 0x000000f0) >> 4) << 8;
            _new_value_n16 |= ((new_value & 0x00000f00) >> 8) << 16;
            _new_value_n16 |= ((new_value & 0x0000f000) >> 12) << 24;
            _new_value_n16 |= ((new_value & 0x000f0000) >> 16) << 4;
            _new_value_n16 |= ((new_value & 0x00f00000) >> 20) << 12;
            _new_value_n16 |= ((new_value & 0x0f000000) >> 24) << 20;
            int32_interleaved[idx] = _new_value_n16;
        }
        else
            int32_interleaved[idx] = new_value;
    }

    // Convert back to int8_t if needed
    memcpy(interleaved, int32_interleaved, size * sizeof(int32_t));
}

template <typename T1, typename T2, bool isSigned>
__device__ void decode_i4b_to_f16(T1 *_i4s, T2 *B_local_decode, const int N = 8)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x000f000f;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    // Minus 7 to scale the value to signed
    static constexpr uint MEDIAN_NUM = isSigned ? 0x64076407 : 0x64006400;
    uint const i4s = *reinterpret_cast<uint *>(_i4s);
#pragma unroll
    // decode 2 elems at one time.
    for (int i = 0; i < (N / 2); i++)
    {

        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[i])
                     : "r"(i4s >> (4 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
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

template <typename T1, typename T2, bool isSigned>
__device__ void decode_i2b_to_f16(T1 *_i2s, T2 *B_local_decode, const int N = 8)
{
    uint *h = reinterpret_cast<uint *>(B_local_decode);

    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint BOTTOM_MASK = 0x00030003;
    static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint MEDIAN_NUM = isSigned ? 0x64016401 : 0x64006400;
    int16_t const i2s_i16 = *reinterpret_cast<int16_t *>(_i2s);
    // decode 2 elems at one time.
    // interleave {e15,e13,e11,e9,e7,e5,e3,e1,e14,e12,e10,e8,e6,e4,e2,e0}
    // only decode for {x,x,x,x,e7,e5,e3,e1,x,x,x,x,e6,e4,e2,e0}
    // otherwise the pointer of _i2s should be moved to
    int i2s = (i2s_i16 & 0x00ff);
    i2s |= ((i2s_i16 & 0xff00) << 8);

#pragma unroll
    for (int i = 0; i < (N / 2); i++)
    {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[i])
                     : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
    }
}

template <typename T1, typename T2>
__device__ void decode_i2s_to_f16(T1 *_i2s, T2 *B_local_decode, const int N = 8)
{
    decode_i2b_to_f16<T1, T2, true>(_i2s, B_local_decode, N);
}

template <typename T1, typename T2>
__device__ void decode_i2u_to_f16(T1 *_i2u, T2 *B_local_decode, const int N = 8)
{
    decode_i2b_to_f16<T1, T2, false>(_i2u, B_local_decode, N);
}
