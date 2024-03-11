# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from string import Template
import os
import tvm
from tvm import IRModule
from tvm.target import Target
from bitblas.utils import match_global_kernel, get_target_from_env
from bitblas.base.analysis import get_reduction_blocks
from bitblas.ops import Operator
from bitblas.ops.matmul_dequantize import (
    MatmulWeightOnlyDequantize,
    MatmulWeightOnlyDequantizeConfig,
)
from bitblas.gpu.intrin.lop3 import (
    decode_i2_to_f16,
    decode_i2_to_f16_scale,
    decode_i4_to_f16,
    decode_i4_to_f16_scale,
)
bit = 2
mask = (1 << bit) - 1
group_size = 128


ft_shapes = [
    [1, 15360, 5120],
    [128, 15360, 5120],
]


target = tvm.target.Target(get_target_from_env())


def get_template_path():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        cur_dir, f"template/kernel_template.int{bit}.bitblas.cu.template"
    )


template_path = get_template_path()


def get_codegen_result(ops: Operator, target: Target):
    code = ops.codegen(target=target)
    return code


def get_thread_block_infomation(mod: IRModule):
    sch = tvm.tir.Schedule(mod)
    root_block = sch.get_block("root")
    child_blocks = sch.get_child_blocks(root_block)
    reduction_blocks = get_reduction_blocks(sch, child_blocks)
    assert len(reduction_blocks) == 1
    (main_block,) = reduction_blocks
    loops = sch.get_loops(main_block)
    block_info = [1, 1, 1]
    grid_info = [1, 1, 1]
    for loop in loops:
        stmt = sch.get(loop)
        thread_binding = stmt.thread_binding
        extent = int(stmt.extent)
        if thread_binding is None:
            continue
        if thread_binding.thread_tag == "threadIdx.x":
            block_info[0] = extent
        elif thread_binding.thread_tag == "threadIdx.y":
            block_info[1] = extent
        elif thread_binding.thread_tag == "threadIdx.z":
            block_info[2] = extent
        elif thread_binding.thread_tag == "blockIdx.x":
            grid_info[0] = extent
        elif thread_binding.thread_tag == "blockIdx.y":
            grid_info[1] = extent
        elif thread_binding.thread_tag == "blockIdx.z":
            grid_info[2] = extent
    return block_info, grid_info

kernel_body = ""
kernel_call = ""
for M, N, K in ft_shapes:
    matmul_config = MatmulWeightOnlyDequantizeConfig(
        M=M,
        N=N,
        K=K,
        in_dtype="float16",
        out_dtype="float16",
        accum_dtype="float16",
        bit=4,
        storage_dtype="int8",
        source_format="int",
        with_zeros=True,
        with_scaling=True,
        group_size=group_size,
        fast_decoding=False,
        with_bias=False,
        propagate_a=False,
        propagate_b=False,
        layout="nt",
    )
    matmul = MatmulWeightOnlyDequantize(
        config=matmul_config,
        target=target,
    )
    matmul.hardware_aware_finetune(topk=10)
    code = get_codegen_result(matmul, target)
    index = match_global_kernel(code)
    headers = code[:index]
    headers.replace('extern "C" ', "")
    declarations = code[index:].split(";")[0]
    index = code.index("{", index)

    function_body = declarations + code[index:]
    # get block infomation from mod
    block_size, grid_size = get_thread_block_infomation(matmul.optimized_func)
    if M != 1 and block_size[0] == 1:
        block_size[0] = 32

    new_kernel_name = (
        f"bitblas_kernel_fp16_int{bit}_fp16_m{M}n{N}k{K}_nt"
    )
    Qweight_bytes = N * K // 8 * bit
    Scale_bytes = N * K // group_size * 2
    function_body = function_body.replace("main_kernel", new_kernel_name)
    call = f"""
            // const dim3 GridDim({grid_size[0]}, {grid_size[1]}, {grid_size[2]});
            // const dim3 BlockDim({block_size[0]}, {block_size[1]}, {block_size[2]});
            // {new_kernel_name}<<<GridDim, BlockDim>>>(input_0, input_1, output);
        """
    function_body = function_body.replace(
        "(half* __restrict__ A, signed char* __restrict__ B, half* __restrict__ D, half* __restrict__ Scale, half* __restrict__ Zeros){",
        f"(half* __restrict__ A, half* __restrict__ QB, half* __restrict__ D) {{\n\
            signed char* B = ((int8_t *)QB);\n\t half* Scale = (half *)((int8_t *)QB + {Qweight_bytes}); \n\t half* Zeros = (half *)((int8_t *)QB + {Qweight_bytes + Scale_bytes}); \
                {call}",
    )
    kernel_body += function_body
    kernel_body += "\n\n"
    real_call = call.replace("//", "")
    real_call = f"""
    if (M == {M} && N == {N} && K == {K}){{
        {real_call}
        return 0;
    }}

    """
    kernel_call += real_call


# make output
cur_dir = os.path.dirname(os.path.abspath(__file__))
ladder_path = os.path.join(cur_dir, f"kenrel_output")
if not os.path.exists(ladder_path):
    os.makedirs(ladder_path)
ladder_kernel_path = os.path.join(ladder_path, f"ladder_kernel.cu")
ladder_header_path = os.path.join(ladder_path, f"ladder_kernel.h")

with open(template_path, mode="r", encoding="utf-8") as r_f, open(
    ladder_kernel_path, mode="w", encoding="utf8"
) as w_f:
    template_content = r_f.read()
    template = Template(template_content)
    data = template.substitute(kernel_body=kernel_body, kernel_call=kernel_call)
    w_f.write(data)

pack_half2 = """
// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}
"""
with open(
    ladder_header_path, mode="w", encoding="utf8"
) as w_f:
    headers = f"""// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef __LADDER_KERNEL_H__
#define __LADDER_KERNEL_H__
#include <cuda_fp16.h>
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800) 
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800) 
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 1
#else
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 0
#endif

{decode_i4_to_f16},

{decode_i4_to_f16_scale},

{decode_i2_to_f16},

{decode_i2_to_f16_scale},

{pack_half2}


int ladder_gemm_fp16xint{bit}_fp16(half *input_0, half *input_1, half *output, const int M, const int N, const int K, const int trans_a, const int trans_b, half *workspace_ptr);

#endif

    """
    w_f.write(headers)
