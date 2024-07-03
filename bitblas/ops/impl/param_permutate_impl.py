# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas.gpu.matmul_analysis import get_propagate_map
from ..operator import TransformKind
from typing import Literal
from tvm import te, IRModule


def select_implementation(
    M: int,
    N: int,
    datatype: Literal["float16"] = "float16",
    transpose_matrix: bool = True,
    group_size: int = -1,
    propagate_kind: TransformKind = TransformKind.NonTransform,
    target_instruction: Literal["nvidia-mma"] = "nvidia-mma",
):
    if target_instruction != "nvidia-mma":
        raise ValueError("Currently only support nvidia-mma instruction")
    if propagate_kind < TransformKind.IntraWarpTransform:
        raise ValueError("Currently only support propagate_kind >= IntraWarpTransform")
    if transpose_matrix is not True:
        raise ValueError("Currently only support transpose_matrix == True")
    # This is trick to get the basic tile size for the current datatype
    # as for nvidia tensorcore instruction, the basic tile size is 16x16/16x32 for float16/int8
    l = r = 16  # noqa: E741
    if datatype in ["int8", "e4m3_float8", "e5m2_float8"]:
        l, r = 16, 32  # noqa: E741
    if group_size == -1:
        group_size = N

    intra_index_map, inverse_indexmap = get_propagate_map(
        transpose_matrix, dtype=datatype, matrix_name=propagate_kind)

    inp = te.placeholder((M, N // group_size), name="inp", dtype=datatype)

    def fcompute(n, k):
        rl, rr = n, k
        warp_i, warp_j = rl % l, rr % r
        spatial_i, spatial_j = rl // l, rr // r
        if propagate_kind >= TransformKind.IntraWarpTransform:
            warp_i, warp_j = intra_index_map.map_indices([warp_i, warp_j])
        new_index = (spatial_i * l + warp_i, (spatial_j * r + warp_j) // group_size)
        return inp[new_index]

    inp_prmt = te.compute(
        (M, N // group_size),
        fcompute,
        name="intra_warp_permutate",
    )

    args = [inp, inp_prmt]

    func = te.create_prim_func(args)

    return IRModule.from_expr(func)
