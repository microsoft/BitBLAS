# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm
from tvm import te
from tvm.tir import IndexMap
from tvm.contrib.dlpack import to_pytorch_func
import torch


def apply_transform_on_input(input: torch.Tensor, index_map: IndexMap) -> torch.Tensor:
    dtype = str(input.dtype).split(".")[1]
    inp = te.placeholder(input.shape, name="inp", dtype=dtype)
    args = [inp]
    arg = args[-1]

    def fcompute(*args):
        warp_i, warp_j = args[-2:]
        spatial_args = args[:-2]
        permutate_i, permutate_j = index_map.map_indices([warp_i, warp_j])
        new_index = (*spatial_args, permutate_i, permutate_j)
        return arg[new_index]

    out = te.compute(
        input.shape,
        fcompute,
        name="permutate",
    )
    args.append(out)
    func = te.create_prim_func(args)
    rt_mod = tvm.build(func, target="llvm", name="permutate")
    output = torch.zeros_like(input)
    torch_func = to_pytorch_func(rt_mod)
    torch_func(input, output)

    return output
