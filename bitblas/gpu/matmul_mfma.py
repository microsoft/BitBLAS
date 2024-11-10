# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: disable=missing-docstring, invalid-name
"""A GEMM schedule rule for GPU operators."""
from typing import Optional

from tvm import tir

from ..base.roller import Hint
from ..base import analysis
from .base import GPUScheduleRule
from .matmul_analysis import get_reduction_blocks


def get_index_map_3d(index_map, l=16, r=16):  # noqa: E741

    def index_map_3d(i, j):
        return (
            i // l,
            j // r,
            *index_map(i % l, j % r),
        )

    return index_map_3d


def get_index_map_5d(index_map):
    """
    for layout transformed gemm, the index map should be 5d
    """

    def index_map_5d(i, j, ii, jj):
        return (
            i,
            j,
            *index_map(ii, jj),
        )

    return index_map_5d


def get_warp_index_map(index_map, l=16, r=16, is_5d=False):  # noqa: E741
    if is_5d:
        return get_index_map_5d(index_map)
    return get_index_map_3d(index_map, l, r)


class MatmulTensorizationMFMA(GPUScheduleRule):
    """
    The schedule rule for float16 tensor core matmul computation.
    func with attr 'dlight.do_not_tensorize' will not be tensorized.
    """

    def apply_config(
        self,
        func: tir.PrimFunc,
        config: Hint,
    ) -> Optional[tir.Schedule]:

        from bitblas.gpu.intrin.hip import (
            get_mfma_intrin_group,)

        is_cross_thread_reduce = (
            hasattr(config, "block_reduction_depth") and config.block_reduction_depth is not None)
        block_reduction_depth = config.block_reduction_depth if is_cross_thread_reduce else 1

        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        main_block = reduction_blocks[0]

        #cache_write_required = True

        shared_scope = config.shared_scope

        intrin_info = config.intrin_info
        intrin_group = get_mfma_intrin_group(
            load_scope=shared_scope,
            store_scope="global",
            a_dtype=intrin_info.in_dtype,
            b_dtype=intrin_info.in_dtype,
            out_dtype=intrin_info.out_dtype,
            trans_a=intrin_info.trans_a,
            trans_b=intrin_info.trans_b,
            not_use_mfma_store_intrinic=False,
        )

        # Start schedule
        warp_row_tiles = config.warp[0]
        warp_col_tiles = config.warp[1]
        block_row_warps = config.block[0] // warp_row_tiles
        block_col_warps = config.block[1] // warp_col_tiles
        reduce_k = block_reduction_depth
        chunk = int(config.rstep[0] / reduce_k)

        #tensor core intrinsic size
        micro_size_x, micro_size_y, micro_size_k = intrin_group["micro_kernel"]

        # get the axis for layout transform
        def get_axis(l, r, trans):
            return (r, l) if trans else (l, r)

        a_lr = get_axis(micro_size_x, micro_size_k, intrin_info.trans_a)
        b_lr = get_axis(micro_size_k, micro_size_y, intrin_info.trans_b)

        # matrix core not support swizzle

        warp_size = 64

        block = main_block

        (i, j, k) = sch.get_loops(block)
        by, i = sch.split(i, factors=[None, config.block[0]])
        bx, j = sch.split(j, factors=[None, config.block[1]])
        bk, k = sch.split(k, factors=[None, (chunk * micro_size_k)])

        sch.reorder(by, bx, bk, i, j, k)

        sch.bind(bx, "blockIdx.x")
        sch.bind(by, "blockIdx.y")

        block_tz, block_inner_i = sch.split(i, factors=[block_row_warps, None])

        block_ty, block_inner_j = sch.split(j, factors=[block_col_warps, None])

        sch.reorder(block_tz, block_ty, bk, block_inner_i, block_inner_j, k)

        sch.bind(block_tz, "threadIdx.z")
        sch.bind(block_ty, "threadIdx.y")

        #schedule the shared memory
        def fetch_to_shared(block, idx, vec_len=8, can_swizzle=False, is_smooth=False, reduce_k=1):
            block_read = sch.cache_read(block, idx, shared_scope)
            sch.compute_at(block_read, bk, preserve_unit_loops=True)
            fused = sch.fuse(*sch.get_loops(block_read)[-2:])

            _, f_0, f_1, f_2, f_3 = sch.split(
                fused, factors=[None, block_row_warps, block_col_warps, warp_size, vec_len])
            sch.bind(f_2, "threadIdx.x")
            sch.bind(f_1, "threadIdx.y")
            sch.bind(f_0, "threadIdx.z")
            sch.vectorize(f_3)

        # fetch A,B to shared
        # 0->A, 1->B
        fetch_to_shared(main_block, 0)
        fetch_to_shared(main_block, 1)

        # blockize for mma tensorize
        block_inner_i, block_inner_i_tc = sch.split(block_inner_i, factors=[None, micro_size_x])
        block_inner_j, block_inner_j_tc = sch.split(block_inner_j, factors=[None, micro_size_y])
        k, k_tc = sch.split(k, factors=[None, micro_size_k])

        if intrin_info.trans_b:
            sch.reorder(k, block_inner_i, block_inner_j, block_inner_i_tc, block_inner_j_tc, k_tc)
        else:
            sch.reorder(block_inner_i, block_inner_j, k, block_inner_i_tc, block_inner_j_tc, k_tc)

        A_mat = sch.cache_read(main_block, 0, "warp")
        B_mat = sch.cache_read(main_block, 1, "warp")
        sch.compute_at(A_mat, k)
        sch.compute_at(B_mat, k)

        C_store = sch.cache_write(main_block, 0, "warp")

        sch.reverse_compute_at(C_store, block_ty)

        i, j = sch.get_loops(C_store)[-2:]
        i0, i1 = sch.split(i, factors=[None, micro_size_x])
        j0, j1 = sch.split(j, factors=[None, micro_size_y])
        sch.reorder(i0, j0, i1, j1)

        def tile_wmma_fragment(block_read, height, width):
            i, j = sch.get_loops(block_read)[-2:]
            i0, i1 = sch.split(i, factors=[None, height])
            j0, j1 = sch.split(j, factors=[None, width])
            sch.reorder(i0, j0, i1, j1)
            return i1

        if intrin_info.trans_b:
            a_loop_warp = tile_wmma_fragment(A_mat, micro_size_x, micro_size_k)
            b_loop_warp = tile_wmma_fragment(B_mat, micro_size_k, micro_size_y)
        else:
            a_loop_warp, _ = sch.get_loops(A_mat)[-2:]
            b_loop_warp, _ = sch.get_loops(B_mat)[-2:]

        block_init_c = sch.decompose_reduction(main_block, bk)

        # Tensorization by hardware intrinsics
        index_map_a, index_map_b, index_map_c = intrin_group["index_map"]

        sch.transform_layout(A_mat, ("write", 0),
                             get_warp_index_map(index_map_a, *b_lr, intrin_info.inter_transform_a))

        sch.transform_layout(
            B_mat,
            ("write", 0),
            get_warp_index_map(index_map_b, *a_lr, intrin_info.inter_transform_b),
        )

        sch.transform_layout(
            C_store,
            ("read", 0),
            get_warp_index_map(index_map_c, is_5d=False),
        )

        sch.tensorize(a_loop_warp, intrin_group["load_a"])
        sch.tensorize(b_loop_warp, intrin_group["load_b"])

        sch.tensorize(block_inner_i_tc, intrin_group["compute"])

        sch.tensorize(sch.get_loops(block_init_c)[-2], intrin_group["init"])

        sch.tensorize(sch.get_loops(C_store)[-2], intrin_group["store"])

        return sch
