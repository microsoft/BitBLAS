# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tqdm
import bitblas
from bitblas.base.arch import CUDA
from bitblas.ops.general_matmul.tirscript.matmul_dequantize_impl import (
    matmul_nt_dequantize_b_propagate_b,
)
import tvm
import itertools

search_space = {
    "block_row_warps": [1],
    "block_col_warps": [1, 2, 4, 8, 16],
    "warp_row_tiles": [1],
    "warp_col_tiles": [1, 2, 4, 8, 16],
    "chunk": [1, 2, 4, 8, 16, 32],
    "stage": [2, 3, 4],
    "block_reduce": [2, 4],
}

keys = search_space.keys()
values = search_space.values()
combinations = list(itertools.product(*values))

combinations_dicts = [dict(zip(keys, combination)) for combination in combinations]

# for combination in combinations_dicts:
#     print(combination)
print(len(combinations_dicts))
group_size = -1
# fmt:off
llm_shape_fp16xint4 = [
    # square test
    (matmul_nt_dequantize_b_propagate_b, (16, 16384, 16384, "float16", "float16", "float16", 4, "int8", "uint",
                            False, False, group_size, True, False)),
]

# fmt:on

target = tvm.target.Target(bitblas.auto_detect_nvidia_target())
benchmark_sets = llm_shape_fp16xint4
tuning_results = {}

min_time = 1e9
min_combination = None
sucess_combinations = []
for get_prim_func, input_args in benchmark_sets:
    ir_module = get_prim_func(*input_args, transform_kind=3)
    func = ir_module["main"]
    arch = CUDA(target)

    M, N, K = input_args[0], input_args[1], input_args[2]
    import numpy as np

    np.random.seed(0)
    # a = np.random.randn(M // 16, K // 16, 16, 16).astype(np.float16)
    a = np.random.randn(M, K).astype(np.float16)
    b = np.random.randn(N // 16, K // 16, 16, 8).astype(np.int8)
    c = np.random.randn(M, N).astype(np.float16)

    tvm_a = tvm.nd.array(a, device=tvm.cuda(0))
    tvm_b = tvm.nd.array(b, device=tvm.cuda(0))
    tvm_c = tvm.nd.array(c, device=tvm.cuda(0))

    intrin_info = bitblas.base.hint.IntrinInfo(
        in_dtype="float16",
        out_dtype="float16",
        trans_b=True,
        input_transform_kind=0,
        weight_transform_kind=3,
    )

    # set up tqdm
    pbar = tqdm.tqdm(combinations_dicts)
    for combination in pbar:
        pbar.set_description(
            f"sucess_combinations: {len(sucess_combinations)} min_time: {min_time}"
        )
        block_row_warps = combination["block_row_warps"]
        block_col_warps = combination["block_col_warps"]
        warp_row_tiles = combination["warp_row_tiles"]
        warp_col_tiles = combination["warp_col_tiles"]
        chunk = combination["chunk"]
        stage = combination["stage"]
        block_reduce = combination["block_reduce"]

        mma_row = mma_col = 16
        mma_k = 16

        block = [
            block_row_warps * warp_row_tiles * mma_row,
            block_col_warps * warp_col_tiles * mma_col,
        ]
        warp = [mma_row * warp_row_tiles, mma_col * warp_col_tiles]
        rstep = [mma_k * chunk * block_reduce]
        pipeline_stage = stage
        block_reduction_depth = block_reduce
        hint = bitblas.base.Hint.from_dict(
            {
                "use_tc": True,
                "arch": arch,
                "block": block,
                "warp": warp,
                "rstep": rstep,
                "pipeline_stage": pipeline_stage,
                "use_async": True,
                "intrin_info": intrin_info,
                "shared_scope": "shared.dyn",
                "vectorize": {"b": 8, "a": 8},
                "block_reduction_depth": block_reduction_depth,
                "rasterization_plan": bitblas.base.rasterization.Rasterization2DColumn(
                    10
                ),
            }
        )
        print("Tuning Hint is", hint)
        try:
            sch = bitblas.gpu.MatmulTensorizationMMAWithDequantizeInfo().sch_warp_memory_prefetch_with_config(
                func, config=hint
            )

            with tvm.transform.PassContext(
                config={
                    "tir.use_async_copy": True,
                    "tir.merge_static_smem": False,
                    "tir.disable_cse_tir": True,
                }
            ):
                rt_mod = tvm.build(sch.mod, target=target)

            time_evaluator = rt_mod.time_evaluator(
                rt_mod.entry_name, tvm.cuda(0), number=10
            )

            t = time_evaluator(tvm_a, tvm_b, tvm_c).mean * 1e3

            print(f"For combination {combination}, time is {t} ms")
            tuning_results["-".join([str(v) for v in combination.values()])] = t
            if t < min_time:
                min_time = t
                min_combination = combination
            sucess_combinations.append(combination)
        except Exception as e:
            del e
            print(f"Failed for combination {combination}")
            continue

print(f"Minimum time is {min_time} for combination {min_combination}")
