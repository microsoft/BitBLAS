# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas import tvm as tvm
from tvm import tl

@tvm.register_func("tvm_callback_hip_postproc", override=True)
def tvm_callback_hip_postproc(code, _):
    print(code)
#     code = '''
#     #include <hip/hip_runtime.h>
# #include <tl_templates/hip/gemm.h>
# #include <tl_templates/hip/copy.h>
# #include <tl_templates/hip/reduce.h>
# #include <tl_templates/hip/ldsm.h>
# #include <tl_templates/hip/threadblock_swizzle.h>

# extern "C" __global__ void __launch_bounds__(128) main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {
#   float C_local[128];
#   __shared__ half_t A_shared[4096];
#   __shared__ half_t B_shared[4096];
#   #pragma unroll
#   for (int i = 0; i < 64; ++i) {
#     *(float2*)(C_local + (i * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
#   }
#   #pragma unroll
#   for (int i_1 = 0; i_1 < 4; ++i_1) {
#     *(uint4*)(A_shared + ((((i_1 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(A + ((i_1 * 1024) + (((int)threadIdx.x) * 8)));
#   }
#   #pragma unroll
#   for (int i_2 = 0; i_2 < 4; ++i_2) {
#     *(uint4*)(B_shared + ((((i_2 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(B + ((i_2 * 1024) + (((int)threadIdx.x) * 8)));
#   }
#   __syncthreads();
#   tl::gemm_ss<128, 128, 32, 2, 2, 0, 1>((&(A_shared[0])), (&(B_shared[0])), (&(C_local[0])));
#   if(threadIdx.x == 0){
#     for (size_t i = 0; i < 128; i++) {
#       printf("%f ", C_local[i]);
#     }
#   }
#   #pragma unroll
#   for (int i_3 = 0; i_3 < 64; ++i_3) {
#     uint1 __1;
#     float2 v_ = *(float2*)(C_local + (i_3 * 2));
#     ((half2*)(&(__1.x)))->x = (half_t)(v_.x);
#     ((half2*)(&(__1.x)))->y = (half_t)(v_.y);
#     *(uint1*)(C + (((((((((i_3 & 7) >> 1) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + ((i_3 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_3 >> 3) * 16)) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __1;
#   }
# }
#     '''
    return code

def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    dtypeAB,
    dtypeC,
    accum_dtype,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    import tvm.tl.language as T

    @T.prim_func
    def main(
        A: T.Buffer(A_shape, dtypeAB),
        B: T.Buffer(B_shape, dtypeAB),
        C: T.Buffer((M, N), dtypeC),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, dtypeAB, scope="shared")
            B_shared = T.alloc_shared(B_shared_shape, dtypeAB, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.serial(T.ceildiv(K, block_K)):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, False, True)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm(
    M,
    N,
    K,
    trans_A,
    trans_B,
    dtypeAB,
    dtypeC,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_threads=128,
):
    program = matmul(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        dtypeAB,
        dtypeC,
        dtypeAccum,
        num_threads,
    )
    mod, params = tl.lower(program, target="hip")
    # print(mod.imported_modules[0].get_source())
    mod = tl.Profiler(mod, params, [], tl.TensorSupplyType.Integer)
    import torch
    torch.random.manual_seed(0)
    a = torch.randn((M, K), dtype=torch.__getattribute__(dtypeAB)).to("cuda")
    # b = torch.randn((N, K), dtype=torch.__getattribute__(dtypeAB)).to("cuda")
    # a = torch.ones((M, K), dtype=torch.__getattribute__(dtypeAB)).to("cuda")
    b = torch.ones((N, K), dtype=torch.__getattribute__(dtypeAB)).to("cuda")
    c = torch.zeros((M, N), dtype=torch.__getattribute__(dtypeC)).to("cuda")
    print(f"{a=}")
    print(f"{b=}")
    mod(a, b, c)

    print(c)

    ref_c = torch.matmul(a, b.T).to(torch.__getattribute__(dtypeC))
    print(ref_c)

    latency = mod.do_bench(mod.func, profiler="tvm")
    print(f"Latency: {latency}")

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

if __name__ == "__main__":
    # run_gemm(
    #     64,
    #     16,
    #     16,
    #     False,
    #     True,
    #     "float16",
    #     "float32",
    #     "float32",
    #     64,
    #     16,
    #     16,
    #     128,
    # )

    run_gemm(
        256,
        256,
        256,
        False,
        True,
        "float16",
        "float32",
        "float32",
        128,
        128,
        32,
        256,
    )
