import numpy as np
from tvm import tir
import os
from ..layout import *
from .tir_base import TIRSchedulerBase
from .utils import write_sch


class TIRLadderMMAScheduler4D(TIRSchedulerBase):
    
    # schedule for general cuda device
    def schedule_consistent(self):
        from .ladder_intrin import (
            TRICKY_MMA_fill_16x16_f16_INTRIN,
            TRICKY_MMA_fill_16x16_f32_INTRIN,
            TRICKY_LDMATRIX_16x16_A_INTRIN,
            TRICKY_LDMATRIX_16x16_B_INTRIN,
            TRICKY_LDMATRIX_16x16_B_TRANS_INTRIN,
            TRICKY_MMA_f16f16f16_INTRIN,
            TRICKY_MMA_f16f16f16_TRANS_INTRIN,
            TRICKY_MMA_f16f16f32_TRANS_INTRIN,
            TRICKY_MMA_bf16bf16f32_TRANS_INTRIN,
            TRICKY_MMA_store_16x16_f16_shared_INTRIN,
            TRICKY_MMA_store_16x16_f16_global_INTRIN,
            TRICKY_MMA_store_16x16_f32_shared_INTRIN,
            A_global_16x16_to_shared_load_16x16_layout,
            B_global_16x16_to_shared_load_16x16_layout,
            C_shared_16x16_to_ldmatrix_32x8_layout,
            A_B_shared_16x16_to_ldmatrix_32x8_layout,
            ASYNC_COPY_F16_X8_INTRIN,
            ASYNC_COPY_S8_X16_INTRIN,
            TRICKY_MMA_fill_16x16_i32_INTRIN,
            TRICKY_LDMATRIX_16x32_A_INTRIN,
            TRICKY_LDMATRIX_32x16_B_INTRIN,
            TRICKY_LDMATRIX_16x32_B_TRANS_INTRIN,
            TRICKY_MMA_i8i8i32_INTRIN,
            TRICKY_MMA_i8i8i32_TRANS_INTRIN,
            TRICKY_MMA_i4i4i32_TRANS_INTRIN,
            TRICKY_MMA_store_16x16_i32_shared_INTRIN,
            TRICKY_MMA_store_16x16_i32_global_INTRIN,
            TRICKY_MMA_store_16x16_f32_global_INTRIN,
            shared_16x16_to_ldmatrix_32x8_layout,
            shared_32x16_to_ldmatrix_32x16_layout,
            shared_16x32_to_ldmatrix_32x16_layout,
            shared_16x32_to_ldmatrix_32x16_permutation,
            A_global_16x32_to_shared_load_16x32_layout,
            B_global_16x32_to_shared_load_16x32_layout,
        )
        # const val for testing
        warp_size = self.config.arch.warp_size
        compute_dtype = self.reduce_op.output(0).dtype
        wmma_k = 32 if compute_dtype == "int32" else 16
        shared_cache_c = self.reduce_op != self.output_op
        sch, config = self.sche, self.config
        write_sch(sch, "original")
        C = sch.get_block(self.reduce_op.name)
        try:
            i, j, kernel_i, kernel_j, k, kernel_k = sch.get_loops(C)
        except ValueError:
            b, i, j, kernel_i, kernel_j, k, kernel_k = sch.get_loops(C)
            sch.bind(b, "blockIdx.z")
        A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n = config.tc_extra_conf.tc_axis
        propagate_inter_a= config.ladder_config.propagate_inter_a
        propagate_inter_b = config.ladder_config.propagate_inter_b 
        transpose_A = A_ax_k < A_ax_m
        transpose_B = B_ax_k > B_ax_n
        block_tile_M, block_tile_N = self.config.block[0], self.config.block[1]
        warp_tile_M, warp_tile_N = self.config.warp[0], self.config.warp[1]
        out_dtype = self.reduce_op.output(0).dtype
        def get_vec(in_dtype):
            if in_dtype == "float32" or in_dtype == "int32":
                vec = 4
            elif in_dtype == "float16":
                vec = 8
            elif in_dtype == "int8":
                vec = 16
            else:
                raise NotImplementedError("dtype {} not supported".format(in_dtype))
            return vec
        vecA = get_vec(self.args[0].dtype)
        vecB = get_vec(self.args[1].dtype)
        vecC = get_vec(self.args[-1].dtype)
        raster = self.config.raster_factor
        # ------------------------ Block and Warp level job partition ------------------------
        chunk_size = config.rstep[0] // wmma_k

        warp_row_tiles = warp_tile_M
        warp_col_tiles = warp_tile_N
        block_row_warps = block_tile_M // warp_tile_M
        block_col_warps = block_tile_N // warp_tile_N
        chunk = chunk_size
        stage = config.pipeline_stage
        use_async = (propagate_inter_a and propagate_inter_b) and stage > 1
       
        ## params for debugging

        # block_row_warps = 2
        # block_col_warps = 2
        # warp_row_tiles = 4
        # warp_col_tiles = 4
        # chunk = 2
        # stage = 2
        # use_async = 1
        # raster = 10
       
        block_i, i, ii = sch.split(i, factors=[None, block_row_warps, warp_row_tiles])
        block_j, j, jj = sch.split(j, factors=[None, block_col_warps, warp_col_tiles])
        ko, ki = sch.split(k, factors=[None, chunk])
        sch.reorder(block_i, block_j, i, j, ko, ki, ii, jj, kernel_i, kernel_j, kernel_k)

        if self.sche.get_sref(ko).stmt.extent <= 32:
            self.sche.unroll(ko)
            sch.annotate(ko, "pragma_unroll_explicit", False) 

        write_sch(sch, "BlockTile")

        sch.bind(block_i, "blockIdx.y")
        sch.bind(block_j, "blockIdx.x")
        sch.bind(i, "threadIdx.y")
        sch.bind(j, "threadIdx.z")
        self.block_size = (self.config.arch.warp_size, sch.get_sref(i).stmt.extent, sch.get_sref(j).stmt.extent)
        self.grid_size = (sch.get_sref(block_j).stmt.extent, sch.get_sref(block_i).stmt.extent, 1)
        write_sch(sch, "thread_bind")

        # ------------------------ Shared memory layout for multiplicand A and B ------------------------

        AS = sch.cache_read(C, 0, "shared")
        BS = sch.cache_read(C, 1, "shared")
        sch.compute_at(AS, ko, preserve_unit_loops=True)
        sch.compute_at(BS, ko, preserve_unit_loops=True)
        write_sch(sch, "cached_shared")
        
        # ------------------------ Schedule output fragment layout ------------------------
        if shared_cache_c:
            C_shared = sch.cache_write(C, 0, "shared")
        C_warp = sch.cache_write(C, 0, "warp")
        
        sch.reverse_compute_at(C_warp, j, preserve_unit_loops=True)
        if shared_cache_c:
            sch.reverse_compute_at(
                C_shared,
                sch.get_loops(C_warp)[-3],
                preserve_unit_loops=True,
            )
        
                
        self.schedule_compute_inline()

        def schedule_shared_output(block):
            o_shared_fused = sch.fuse(*sch.get_loops(block)[-4:])
            if sch.get_sref(o_shared_fused).stmt.extent // warp_size < vecC:
                vec = sch.get_sref(o_shared_fused).stmt.extent // warp_size
            else:
                vec = vecC
            oo, o_shared_tx, o_shared_vi = sch.split(
                o_shared_fused, factors=[None, warp_size, vec]
            )     
            sch.vectorize(o_shared_vi)
            sch.bind(o_shared_tx, "threadIdx.x")
            sch.unroll(oo)
            sch.annotate(oo, "pragma_unroll_explicit", False)
        if shared_cache_c:
            schedule_shared_output(C_shared)
        
        write_sch(sch, "schedule_warp")

        write_sch(sch, "schedule_compute_inline")

        a_prmt_func = A_global_16x32_to_shared_load_16x32_layout if compute_dtype == "int32" else A_global_16x16_to_shared_load_16x16_layout
        b_prmt_func = B_global_16x32_to_shared_load_16x32_layout if compute_dtype == "int32" else B_global_16x16_to_shared_load_16x16_layout

        def A_permutation(*args):
            kernel_i, kernel_j = args[-2], args[-1]
            other_args = args[:-2]
            result = (*other_args, *a_prmt_func(kernel_i, kernel_j))
            return result

        def B_permutation(*args):
            kernel_i, kernel_j = args[-2], args[-1]
            other_args = args[:-2]
            if transpose_B:
                return (*other_args, *b_prmt_func(kernel_i, kernel_j))
            else:
                return (*other_args, *a_prmt_func(kernel_i, kernel_j))

        if not propagate_inter_a:
            sch.transform_layout(AS, ("read", 0),
                                    A_permutation)
        if not propagate_inter_b:
            sch.transform_layout(BS, ("read", 0),
                                    B_permutation)
        
        def cooperative_fetch(block, dims=4, vec=1, use_pragma_unroll=False, force_async_copy=False):
            read_dtype = sch.get_sref(block).stmt.reads[-1].buffer.dtype
            out_dtype = sch.get_sref(block).stmt.writes[-1].buffer.dtype
            if read_dtype == "int8" and out_dtype == "float16":
                vec = 4
            
            loops = sch.get_loops(block)[-dims:]
            other_loop = loops[:-1]
            shared_j = loops[-1]
            shared_j, shared_vi = sch.split(shared_j, factors=[None, vec])
            sch.vectorize(shared_vi)
            if force_async_copy and read_dtype == out_dtype:
                sch.tensorize(shared_vi, ASYNC_COPY_F16_X8_INTRIN if read_dtype == "float16" else ASYNC_COPY_S8_X16_INTRIN)
                sch.annotate(ki, "pragma_commit_wait", "")
            shared_fused = sch.fuse(*other_loop, shared_j)
            shared_inner, shared_ty, shared_tz, shared_tx = sch.split(
                shared_fused, factors=[None, block_row_warps, block_col_warps, warp_size])
            sch.bind(shared_tx, "threadIdx.x")
            sch.bind(shared_ty, "threadIdx.y")
            sch.bind(shared_tz, "threadIdx.z")
            self.sche.unroll(shared_inner)
            if use_pragma_unroll:
                self.sche.annotate(shared_inner, "pragma_unroll_explicit", False)

        cooperative_fetch(AS, dims=4, vec=vecA, use_pragma_unroll=True, force_async_copy=(propagate_inter_a and not propagate_inter_b))
        cooperative_fetch(BS, dims=4, vec=vecB, use_pragma_unroll=True, force_async_copy=(propagate_inter_b and not propagate_inter_a))
        
        write_sch(sch, "schedule_shared")
        # ------------------------ Warp memory layout for multiplicand A and B ------------------------
        AW = sch.cache_read(C, 0, "warp")
        BW = sch.cache_read(C, 1, "warp")
        sch.compute_at(AW, ki, preserve_unit_loops=True)
        sch.compute_at(BW, ki, preserve_unit_loops=True)
        
        a_index_map = shared_16x32_to_ldmatrix_32x16_layout if compute_dtype == "int32" else A_B_shared_16x16_to_ldmatrix_32x8_layout
        b_index_map = shared_16x32_to_ldmatrix_32x16_layout if compute_dtype == "int32" else A_B_shared_16x16_to_ldmatrix_32x8_layout
        c_index_map = C_shared_16x16_to_ldmatrix_32x8_layout if compute_dtype == "int32" else C_shared_16x16_to_ldmatrix_32x8_layout
        
        def index_map_A(i, k, wmma_m, wmma_k):
            return (i, k, *a_index_map(wmma_m, wmma_k))


        def index_map_B(*args):
            kernel_i, kernel_j = args[-2], args[-1]  
            other_args = args[:-2]    
            result = (*other_args, *b_index_map(kernel_i, kernel_j))
            return result


        def index_map_C(m, n, wmma_m, wmma_n):
            return (m, n, *c_index_map(wmma_m, wmma_n),)


        sch.transform_layout(AW, ("write", 0), index_map_A)
        sch.transform_layout(BW, ("write", 0), index_map_A if not transpose_B else index_map_B)
        sch.transform_layout(C_warp, ("read", 0), index_map_C)
        
        # ------------------------ Tensorize and Pipelining -------------------------
        init_intrin = TRICKY_MMA_fill_16x16_i32_INTRIN if compute_dtype == "int32" else TRICKY_MMA_fill_16x16_f16_INTRIN if compute_dtype == "float16" else TRICKY_MMA_fill_16x16_f32_INTRIN
        load_a_intrin = TRICKY_LDMATRIX_16x32_A_INTRIN if compute_dtype == "int32" else TRICKY_LDMATRIX_16x16_A_INTRIN
        load_b_intrin = TRICKY_LDMATRIX_32x16_B_INTRIN if compute_dtype == "int32" else TRICKY_LDMATRIX_16x16_B_INTRIN
        load_b_intrin_trans = TRICKY_LDMATRIX_16x32_B_TRANS_INTRIN if compute_dtype == "int32" else TRICKY_LDMATRIX_16x16_B_TRANS_INTRIN
        compute_intrin = TRICKY_MMA_i8i8i32_INTRIN if compute_dtype == "int32" else TRICKY_MMA_f16f16f16_INTRIN
        compute_trans_intrin = TRICKY_MMA_i8i8i32_TRANS_INTRIN if compute_dtype == "int32" else TRICKY_MMA_f16f16f16_TRANS_INTRIN if compute_dtype == "float16" else TRICKY_MMA_f16f16f32_TRANS_INTRIN
        if shared_cache_c:
            store_intrin = TRICKY_MMA_store_16x16_i32_shared_INTRIN if compute_dtype == "int32" else  TRICKY_MMA_store_16x16_f16_shared_INTRIN if compute_dtype == "float16" else TRICKY_MMA_store_16x16_f32_shared_INTRIN
        else:
            store_intrin = TRICKY_MMA_store_16x16_i32_global_INTRIN if compute_dtype == "int32" else  TRICKY_MMA_store_16x16_f16_global_INTRIN if compute_dtype == "float16" else TRICKY_MMA_store_16x16_f32_global_INTRIN
    
        if config.ladder_compute_type == "int4":
            compute_trans_intrin = TRICKY_MMA_i4i4i32_TRANS_INTRIN
        init_block_b = sch.decompose_reduction(C, ko)
        write_sch(sch, "decompose_reduction")
        init_block_b_loops = sch.get_loops(init_block_b)
        init_block_b_i, init_block_b_j = sch.get_loops(init_block_b)[-4:-2]
        sch.annotate(init_block_b_i, "pragma_unroll_explicit", False)
        sch.annotate(init_block_b_j, "pragma_unroll_explicit", False)
        sch.tensorize(sch.get_loops(init_block_b)[-2], init_intrin)
        sch.tensorize(
            sch.get_loops(AW)[-2], load_a_intrin
        )
        sch.tensorize(
            sch.get_loops(
                BW)[-2], load_b_intrin if not transpose_B else load_b_intrin_trans
        )
        sch.tensorize(
            kernel_i, compute_intrin if not transpose_B else compute_trans_intrin)
        sch.tensorize(
            sch.get_loops(C_warp)[-2],
            store_intrin,
        )
        if raster > 0:
            sch.annotate(init_block_b_loops[-4], ann_key="thread_rasterization", ann_val=raster)
        if stage > 1:
            if stage > 1:
                sch.annotate(ko, ann_key="software_pipeline_stage", ann_val=[0, 0, stage - 1])
                sch.annotate(ko, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
            if use_async:
                sch.annotate(ko, "software_pipeline_async_stages", [0])

        write_sch(sch, "cache_small_tensor")

        return sch.mod["main"]
    
    # schedule for general game card
    def schedule_inconsistent(self, is_a_consistent=False, is_b_consistent=False):
        from .ladder_intrin import (
            TRICKY_MMA_fill_16x16_f16_INTRIN,
            TRICKY_MMA_fill_16x16_f32_INTRIN,
            TRICKY_LDMATRIX_16x16_A_INTRIN,
            TRICKY_LDMATRIX_16x16_B_INTRIN,
            TRICKY_LDMATRIX_16x16_B_TRANS_INTRIN,
            TRICKY_MMA_f16f16f16_INTRIN,
            TRICKY_MMA_f16f16f16_TRANS_INTRIN,
            TRICKY_MMA_f16f16f32_TRANS_INTRIN,
            TRICKY_MMA_bf16bf16f32_TRANS_INTRIN,
            TRICKY_MMA_store_16x16_f16_shared_INTRIN,
            TRICKY_MMA_store_16x16_f16_global_INTRIN,
            TRICKY_MMA_store_16x16_f32_shared_INTRIN,
            TRICKY_MMA_store_16x16_f32_global_INTRIN,
            A_global_16x16_to_shared_load_16x16_layout,
            B_global_16x16_to_shared_load_16x16_layout,
            C_shared_16x16_to_ldmatrix_32x8_layout,
            A_B_shared_16x16_to_ldmatrix_32x8_layout,
            ASYNC_COPY_F16_X8_INTRIN,
            ASYNC_COPY_S8_X16_INTRIN,
            TRICKY_MMA_fill_16x16_i32_INTRIN,
            TRICKY_LDMATRIX_16x32_A_INTRIN,
            TRICKY_LDMATRIX_32x16_B_INTRIN,
            TRICKY_LDMATRIX_16x32_B_TRANS_INTRIN,
            TRICKY_MMA_i8i8i32_INTRIN,
            TRICKY_MMA_i8i8i32_TRANS_INTRIN,
            TRICKY_MMA_store_16x16_i32_shared_INTRIN,
            TRICKY_MMA_store_16x16_i32_global_INTRIN,
            shared_16x16_to_ldmatrix_32x8_layout,
            shared_16x32_to_ldmatrix_32x16_layout,
            A_global_16x32_to_shared_load_16x32_layout,
            B_global_16x32_to_shared_load_16x32_layout,
        )
        # const val for testing
        # assert is_a_consistent, "currently A should be consistent"
        num_args = len(self.args)
        is_lut = False
        if num_args >= 4:
            lut_arg = self.args[2] # assume the 3rd arg is the lut
            lut_shape = np.prod(lut_arg.shape)
            if lut_shape == 16:
                is_lut = True
        warp_size = self.config.arch.warp_size
        compute_dtype = self.reduce_op.output(0).dtype
        wmma_k = 32 if compute_dtype == "int32" else 16
        shared_cache_c = (compute_dtype != "float32")
        sch, config = self.sche, self.config
        write_sch(sch, "original")
        C = sch.get_block(self.reduce_op.name)
        i, j, kernel_i, kernel_j, k, kernel_k = sch.get_loops(C)
        A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n = config.tc_extra_conf.tc_axis
        propagate_inter_a= config.ladder_config.propagate_inter_a
        propagate_inter_b = config.ladder_config.propagate_inter_b 
        transpose_A = A_ax_k < A_ax_m
        transpose_B = B_ax_k > B_ax_n
        block_tile_M, block_tile_N = self.config.block[0], self.config.block[1]
        warp_tile_M, warp_tile_N = self.config.warp[0], self.config.warp[1]
        out_dtype = self.reduce_op.output(0).dtype
        def get_vec(in_dtype):
            if in_dtype == "float32" or in_dtype == "int32":
                vec = 4
            elif in_dtype == "float16":
                vec = 8
            elif in_dtype == "int8":
                vec = 16
            else:
                raise NotImplementedError("dtype {} not supported".format(in_dtype))
            return vec
        vecA = get_vec(self.args[0].dtype)
        vecB = get_vec(self.args[1].dtype)
        vecC = get_vec(self.args[-1].dtype)
        raster = self.config.raster_factor
        # ------------------------ Block and Warp level job partition ------------------------
        chunk_size = config.rstep[0] // wmma_k
        warp_row_tiles = warp_tile_M
        warp_col_tiles = warp_tile_N
        block_row_warps = block_tile_M // warp_tile_M
        block_col_warps = block_tile_N // warp_tile_N
        chunk = chunk_size
        stage = config.pipeline_stage
        use_async = (propagate_inter_a and propagate_inter_b) and stage > 1

        
        block_i, i, ii = sch.split(i, factors=[None, block_row_warps, warp_row_tiles])
        block_j, j, jj = sch.split(j, factors=[None, block_col_warps, warp_col_tiles])
        ko, ki = sch.split(k, factors=[None, chunk])
        sch.reorder(block_i, block_j, i, j, ko, ki, ii, jj, kernel_i, kernel_j, kernel_k)

        write_sch(sch, "BlockTile")

        sch.bind(block_i, "blockIdx.y")
        sch.bind(block_j, "blockIdx.x")
        sch.bind(i, "threadIdx.y")
        sch.bind(j, "threadIdx.z")
        self.block_size = (self.config.arch.warp_size, sch.get_sref(i).stmt.extent, sch.get_sref(j).stmt.extent)
        self.grid_size = (sch.get_sref(block_j).stmt.extent, sch.get_sref(block_i).stmt.extent, 1)
        write_sch(sch, "thread_bind")

        # ------------------------ Shared memory layout for multiplicand A and B ------------------------

        AS = sch.cache_read(C, 0, "shared")
        BS = sch.cache_read(C, 1, "shared")
        sch.compute_at(AS, ko, preserve_unit_loops=True)
        sch.compute_at(BS, ko, preserve_unit_loops=True)
        write_sch(sch, "cached_shared")
        
        # ------------------------ Schedule output fragment layout ------------------------
        if shared_cache_c:
            C_shared = sch.cache_write(C, 0, "shared")
        C_warp = sch.cache_write(C, 0, "warp")
        
        sch.reverse_compute_at(C_warp, j, preserve_unit_loops=True)
        if shared_cache_c:
            sch.reverse_compute_at(
                C_shared,
                sch.get_loops(C_warp)[-3],
                preserve_unit_loops=True,
            )
        
        write_sch(sch, "schedule_warp")
        
        a_prmt_func = A_global_16x32_to_shared_load_16x32_layout if compute_dtype == "int32" else A_global_16x16_to_shared_load_16x16_layout
        b_prmt_func = B_global_16x32_to_shared_load_16x32_layout if compute_dtype == "int32" else B_global_16x16_to_shared_load_16x16_layout

        def A_permutation(*args):
            kernel_i, kernel_j = args[-2], args[-1]
            other_args = args[:-2]
            result = (*other_args, *a_prmt_func(kernel_i, kernel_j))
            return result

        def B_permutation(*args):
            kernel_i, kernel_j = args[-2], args[-1]
            other_args = args[:-2]
            if transpose_B:
                return (*other_args, *b_prmt_func(kernel_i, kernel_j))
            else:
                return (*other_args, *a_prmt_func(kernel_i, kernel_j))

        if not propagate_inter_a:
            sch.transform_layout(AS, ("read", 0),
                                    A_permutation)
        if not propagate_inter_b:
            sch.transform_layout(BS, ("read", 0),
                                    B_permutation)
        
        def cooperative_fetch(block, dims=4, vec=1, use_pragma_unroll=False, force_async_copy=False):
            read_dtype = sch.get_sref(block).stmt.reads[-1].buffer.dtype
            out_dtype = sch.get_sref(block).stmt.writes[-1].buffer.dtype
            if read_dtype == "int8" and out_dtype == "float16":
                vec = 4
            
            loops = sch.get_loops(block)[-dims:]
            other_loop = loops[:-1]
            shared_j = loops[-1]
            shared_j, shared_vi = sch.split(shared_j, factors=[None, vec])
            if sch.get_sref(shared_vi).stmt.extent in [1, 2, 4, 8, 16]:
                sch.vectorize(shared_vi)
            if force_async_copy and read_dtype == out_dtype:
                sch.tensorize(shared_vi, ASYNC_COPY_F16_X8_INTRIN if read_dtype == "float16" else ASYNC_COPY_S8_X16_INTRIN)
                sch.annotate(ki, "pragma_commit_wait", "")
            shared_fused = sch.fuse(*other_loop, shared_j)
            shared_inner, shared_ty, shared_tz, shared_tx = sch.split(
                shared_fused, factors=[None, block_row_warps, block_col_warps, warp_size])
            sch.bind(shared_tx, "threadIdx.x")
            sch.bind(shared_ty, "threadIdx.y")
            sch.bind(shared_tz, "threadIdx.z")
            self.sche.unroll(shared_inner)
            if use_pragma_unroll:
                self.sche.annotate(shared_inner, "pragma_unroll_explicit", False)

        if is_a_consistent:
            cooperative_fetch(AS, dims=4, vec=vecA, use_pragma_unroll=True, force_async_copy=(propagate_inter_a and not propagate_inter_b))
        if is_b_consistent:
            cooperative_fetch(BS, dims=4, vec=vecB, use_pragma_unroll=True, force_async_copy=(propagate_inter_b and not propagate_inter_a))
        A_decode_block = None
        B_decode_block = None
        other_blocks = []
        for op in reversed(self.ops):
            if op not in (self.reduce_op, *[arg.op for arg in self.output_args]):
                if op.name == 'A_decode':
                    A_decode_block = self.sche.get_block(op.name)
                elif op.name == 'B_decode':
                    B_decode_block = self.sche.get_block(op.name)
                elif op.name == 'mediate0' and is_a_consistent and not is_b_consistent:
                    B_decode_block = self.sche.get_block(op.name)
                else:
                    block = self.sche.get_block(op.name)
                    other_blocks.append(block)
            
        if not is_a_consistent and A_decode_block != None:
            # cache_decompress
            A_shared_jj = sch.get_loops(AS)[-1]
            A_shared_jj, A_shared_vi, A_shared_vj = sch.split(A_shared_jj, factors=[None, 1, 8])
            block_local_A_decompress = sch.cache_read(AS, 0, "local")
            sch.compute_inline(A_decode_block)
            block_local_A = sch.cache_read(block_local_A_decompress, 0, "local")
            sch.compute_at(block_local_A_decompress, A_shared_vi)
            sch.compute_at(block_local_A, A_shared_vi)
            A_shared_fused = sch.fuse(*sch.get_loops(AS)[-6:-2])
            A_shared_inner, A_shared_ty, A_shared_tz, A_shared_tx = sch.split(
                A_shared_fused, factors=[None, block_row_warps, block_col_warps, warp_size])
            sch.vectorize(sch.get_loops(AS)[-1])
            sch.vectorize(sch.get_loops(block_local_A)[-1]) 
            sch.bind(A_shared_tx, "threadIdx.x")
            sch.bind(A_shared_ty, "threadIdx.y")
            sch.bind(A_shared_tz, "threadIdx.z")
        if not is_b_consistent and B_decode_block != None:
            # cache_decompress
            B_shared_jj = sch.get_loops(BS)[-1]
            B_shared_jj, B_shared_vi, B_shared_vj = sch.split(B_shared_jj, factors=[None, 1, 8])
            block_local_B_decompress = sch.cache_read(BS, 0, "local")
            sch.compute_inline(B_decode_block)
            if is_lut:
                block_local_B = sch.cache_read(block_local_B_decompress, 1, "local")
            else:
                block_local_B = sch.cache_read(block_local_B_decompress, 0, "local")
            sch.compute_at(block_local_B_decompress, B_shared_vi)
            sch.compute_at(block_local_B, B_shared_vi)
            B_shared_fused = sch.fuse(*sch.get_loops(BS)[-6:-2])
            B_shared_inner, B_shared_ty, B_shared_tz, B_shared_tx = sch.split(
                B_shared_fused, factors=[None, block_row_warps, block_col_warps, warp_size])
            sch.vectorize(sch.get_loops(BS)[-1])
            sch.vectorize(sch.get_loops(block_local_B)[-1]) 
            sch.bind(B_shared_tx, "threadIdx.x")
            sch.bind(B_shared_ty, "threadIdx.y")
            sch.bind(B_shared_tz, "threadIdx.z")
            if is_lut:
                block_shared_lut = sch.cache_read(block_local_B_decompress, 0, "shared")
                sch.reverse_compute_at(block_shared_lut, j)
                _, B_shared_tx = sch.split(
                    sch.get_loops(block_shared_lut)[-1], factors=[None, warp_size])
                sch.bind(B_shared_tx, "threadIdx.x")
        
        for block in other_blocks:
            self.sche.compute_inline(block)
        if self.reduce_op != None and self.reduce_op != self.output_op:
            block = self.sche.get_block(self.output_op.name)
            self.sche.reverse_compute_inline(block)

        def schedule_shared_output(block):
            o_shared_fused = sch.fuse(*sch.get_loops(block)[-4:])
            oo, o_shared_tx, o_shared_vi = sch.split(
                o_shared_fused, factors=[None, warp_size, vecC]
            )     
            sch.vectorize(o_shared_vi)
            sch.bind(o_shared_tx, "threadIdx.x")
            sch.unroll(oo)
            sch.annotate(oo, "pragma_unroll_explicit", False)
        
        if shared_cache_c:
            schedule_shared_output(C_shared)
        
        write_sch(sch, "schedule_shared")
        # ------------------------ Warp memory layout for multiplicand A and B ------------------------
        AW = sch.cache_read(C, 0, "warp")
        BW = sch.cache_read(C, 1, "warp")
        sch.compute_at(AW, ki, preserve_unit_loops=True)
        sch.compute_at(BW, ki, preserve_unit_loops=True)

        a_index_map = shared_16x32_to_ldmatrix_32x16_layout if compute_dtype == "int32" else A_B_shared_16x16_to_ldmatrix_32x8_layout
        b_index_map = shared_16x32_to_ldmatrix_32x16_layout if compute_dtype == "int32" else A_B_shared_16x16_to_ldmatrix_32x8_layout
        c_index_map = C_shared_16x16_to_ldmatrix_32x8_layout if compute_dtype == "int32" else C_shared_16x16_to_ldmatrix_32x8_layout

        def index_map_A(i, k, wmma_m, wmma_k):
            return (i, k, *a_index_map(wmma_m, wmma_k))

        def index_map_B(*args):
            kernel_i, kernel_j = args[-2], args[-1]
            other_args = args[:-2]
            result = (*other_args, *b_index_map(kernel_i, kernel_j))
            return result

        def index_map_C(m, n, wmma_m, wmma_n):
            return (m, n, *c_index_map(wmma_m, wmma_n),)


        sch.transform_layout(AW, ("write", 0), index_map_A)
        sch.transform_layout(BW, ("write", 0), index_map_A if not transpose_B else index_map_B)
        sch.transform_layout(C_warp, ("read", 0), index_map_C)
        
        # ------------------------ Tensorize and Pipelining -------------------------
        init_intrin = TRICKY_MMA_fill_16x16_i32_INTRIN if compute_dtype == "int32" else TRICKY_MMA_fill_16x16_f16_INTRIN if compute_dtype == "float16" else TRICKY_MMA_fill_16x16_f32_INTRIN
        load_a_intrin = TRICKY_LDMATRIX_16x32_A_INTRIN if compute_dtype == "int32" else TRICKY_LDMATRIX_16x16_A_INTRIN
        load_b_intrin = TRICKY_LDMATRIX_32x16_B_INTRIN if compute_dtype == "int32" else TRICKY_LDMATRIX_16x16_B_INTRIN
        load_b_intrin_trans = TRICKY_LDMATRIX_16x32_B_TRANS_INTRIN if compute_dtype == "int32" else TRICKY_LDMATRIX_16x16_B_TRANS_INTRIN
        compute_intrin = TRICKY_MMA_i8i8i32_INTRIN if compute_dtype == "int32" else TRICKY_MMA_f16f16f16_INTRIN
        compute_trans_intrin = TRICKY_MMA_i8i8i32_TRANS_INTRIN if compute_dtype == "int32" else TRICKY_MMA_f16f16f16_TRANS_INTRIN if compute_dtype == "float16" else TRICKY_MMA_f16f16f32_TRANS_INTRIN
        if shared_cache_c:
            store_intrin = TRICKY_MMA_store_16x16_i32_shared_INTRIN if compute_dtype == "int32" else  TRICKY_MMA_store_16x16_f16_shared_INTRIN if compute_dtype == "float16" else TRICKY_MMA_store_16x16_f32_shared_INTRIN
        else:
            store_intrin = TRICKY_MMA_store_16x16_i32_global_INTRIN if compute_dtype == "int32" else  TRICKY_MMA_store_16x16_f16_global_INTRIN if compute_dtype == "float16" else TRICKY_MMA_store_16x16_f32_global_INTRIN
    
        init_block_b = sch.decompose_reduction(C, ko)
        write_sch(sch, "decompose_reduction")
        init_block_b_loops = sch.get_loops(init_block_b)
        init_block_b_i, init_block_b_j = sch.get_loops(init_block_b)[-4:-2]
        sch.annotate(init_block_b_i, "pragma_unroll_explicit", False)
        sch.annotate(init_block_b_j, "pragma_unroll_explicit", False)
        sch.tensorize(sch.get_loops(init_block_b)[-2], init_intrin)
        sch.tensorize(
            sch.get_loops(AW)[-2], load_a_intrin
        )
        sch.tensorize(
            sch.get_loops(
                BW)[-2], load_b_intrin if not transpose_B else load_b_intrin_trans
        )
        sch.tensorize(
            kernel_i, compute_intrin if not transpose_B else compute_trans_intrin)
        sch.tensorize(
            sch.get_loops(C_warp)[-2],
            store_intrin,
        )
        write_sch(sch, "tensorize_store")
        if raster > 0:
            sch.annotate(init_block_b_loops[-4], ann_key="thread_rasterization", ann_val=raster)
        if stage > 1:
            sch.annotate(ko, ann_key="software_pipeline_stage", ann_val=[0, 0, stage - 1])
            sch.annotate(ko, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
            if use_async:
                sch.annotate(ko, "software_pipeline_async_stages", [0])

        write_sch(sch, "cache_small_tensor")

        return sch.mod["main"]

    # schedule for sm80 device.
    def schedule_inconsistent_shared_decode(self, is_a_consistent=False, is_b_consistent=False):
        from .ladder_intrin import (
            TRICKY_MMA_fill_16x16_f16_INTRIN,
            TRICKY_MMA_fill_16x16_f32_INTRIN,
            TRICKY_LDMATRIX_16x16_A_INTRIN,
            TRICKY_LDMATRIX_16x16_B_INTRIN,
            TRICKY_LDMATRIX_16x16_B_TRANS_INTRIN,
            TRICKY_MMA_f16f16f16_INTRIN,
            TRICKY_MMA_f16f16f16_TRANS_INTRIN,
            TRICKY_MMA_f16f16f32_TRANS_INTRIN,
            TRICKY_MMA_bf16bf16f32_TRANS_INTRIN,
            TRICKY_MMA_store_16x16_f16_shared_INTRIN,
            TRICKY_MMA_store_16x16_f32_shared_INTRIN,
            A_global_16x16_to_shared_load_16x16_layout,
            B_global_16x16_to_shared_load_16x16_layout,
            C_shared_16x16_to_ldmatrix_32x8_layout,
            A_B_shared_16x16_to_ldmatrix_32x8_layout,
            ASYNC_COPY_F16_X8_INTRIN,
            ASYNC_COPY_S8_X16_INTRIN,
            TRICKY_MMA_fill_16x16_i32_INTRIN,
            TRICKY_LDMATRIX_16x32_A_INTRIN,
            TRICKY_LDMATRIX_32x16_B_INTRIN,
            TRICKY_LDMATRIX_16x32_B_TRANS_INTRIN,
            TRICKY_MMA_i8i8i32_INTRIN,
            TRICKY_MMA_i8i8i32_TRANS_INTRIN,
            TRICKY_MMA_store_16x16_i32_shared_INTRIN,
            TRICKY_MMA_store_16x16_i32_global_INTRIN,
            TRICKY_MMA_store_16x16_f32_global_INTRIN,
            TRICKY_MMA_store_16x16_f16_global_INTRIN,
            shared_16x16_to_ldmatrix_32x8_layout,
            shared_32x16_to_ldmatrix_32x16_layout,
            shared_16x32_to_ldmatrix_32x16_layout,
            shared_16x32_to_ldmatrix_32x16_permutation,
            A_global_16x32_to_shared_load_16x32_layout,
            B_global_16x32_to_shared_load_16x32_layout,
        )
        # const val for testing
        # assert is_a_consistent, "currently A should be consistent"
        num_args = len(self.args)
        is_lut = False
        if num_args >= 4:
            lut_arg = self.args[2] # assume the 3rd arg is the lut
            lut_shape = np.prod(lut_arg.shape)
            if lut_shape == 16:
                is_lut = True
        warp_size = self.config.arch.warp_size
        compute_dtype = self.reduce_op.output(0).dtype
        wmma_k = 32 if compute_dtype == "int32" else 16
        shared_cache_c = self.reduce_op != self.output_op
        shared_cache_scale = self.config.ladder_compute_type == "mxfp"
        sch, config = self.sche, self.config
        write_sch(sch, "original")
        C = sch.get_block(self.reduce_op.name)
        i, j, kernel_i, kernel_j, k, kernel_k = sch.get_loops(C)
        A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n = config.tc_extra_conf.tc_axis
        propagate_inter_a= config.ladder_config.propagate_inter_a
        propagate_inter_b = config.ladder_config.propagate_inter_b 
        transpose_A = A_ax_k < A_ax_m
        transpose_B = B_ax_k > B_ax_n
        block_tile_M, block_tile_N = self.config.block[0], self.config.block[1]
        warp_tile_M, warp_tile_N = self.config.warp[0], self.config.warp[1]
        out_dtype = self.reduce_op.output(0).dtype
        def get_vec(in_dtype):
            if in_dtype == "float32" or in_dtype == "int32":
                vec = 4
            elif in_dtype == "float16":
                vec = 8
            elif in_dtype == "int8":
                vec = 16
            else:
                raise NotImplementedError("dtype {} not supported".format(in_dtype))
            return vec
        vecA = get_vec(self.args[0].dtype)
        vecB = get_vec(self.args[1].dtype)
        vecC = get_vec(self.args[-1].dtype)
        raster = self.config.raster_factor
        # ------------------------ Block and Warp level job partition ------------------------
        chunk_size = config.rstep[0] // wmma_k
        warp_row_tiles = warp_tile_M
        warp_col_tiles = warp_tile_N
        block_row_warps = block_tile_M // warp_tile_M
        block_col_warps = block_tile_N // warp_tile_N
        chunk = chunk_size
        stage = config.pipeline_stage
        use_async = (propagate_inter_a and propagate_inter_b) and stage > 1

        # block_row_warps = 2
        # block_col_warps = 2
        # warp_row_tiles = 4
        # warp_col_tiles = 4
        # chunk = 2
        # stage = 2
        # use_async = 1
        # raster = 0

        block_i, i, ii = sch.split(i, factors=[None, block_row_warps, warp_row_tiles])
        block_j, j, jj = sch.split(j, factors=[None, block_col_warps, warp_col_tiles])
        ko, ki = sch.split(k, factors=[None, chunk])
        sch.reorder(block_i, block_j, i, j, ko, ki, ii, jj, kernel_i, kernel_j, kernel_k)


        write_sch(sch, "BlockTile")

        sch.bind(block_i, "blockIdx.y")
        sch.bind(block_j, "blockIdx.x")
        sch.bind(i, "threadIdx.y")
        sch.bind(j, "threadIdx.z")
        self.block_size = (self.config.arch.warp_size, sch.get_sref(i).stmt.extent, sch.get_sref(j).stmt.extent)
        self.grid_size = (sch.get_sref(block_j).stmt.extent, sch.get_sref(block_i).stmt.extent, 1)
        write_sch(sch, "thread_bind")

        # ------------------------ Shared memory layout for multiplicand A and B ------------------------

        AS = sch.cache_read(C, 0, "shared")
        BS = sch.cache_read(C, 1, "shared")

        sch.compute_at(AS, ko, preserve_unit_loops=True)
        sch.compute_at(BS, ko, preserve_unit_loops=True)
        write_sch(sch, "cached_shared")
        
        # ------------------------ Schedule output fragment layout ------------------------
        if shared_cache_c:
            C_shared = sch.cache_write(C, 0, "shared")
        C_warp = sch.cache_write(C, 0, "warp")
        
        sch.reverse_compute_at(C_warp, j, preserve_unit_loops=True)

        a_prmt_func = A_global_16x32_to_shared_load_16x32_layout if compute_dtype == "int32" else A_global_16x16_to_shared_load_16x16_layout
        b_prmt_func = B_global_16x32_to_shared_load_16x32_layout if compute_dtype == "int32" else B_global_16x16_to_shared_load_16x16_layout

        def A_permutation(*args):
            kernel_i, kernel_j = args[-2], args[-1]
            other_args = args[:-2]
            result = (*other_args, *a_prmt_func(kernel_i, kernel_j))
            return result

        def B_permutation(*args):
            kernel_i, kernel_j = args[-2], args[-1]
            other_args = args[:-2]
            if transpose_B:
                return (*other_args, *b_prmt_func(kernel_i, kernel_j))
            else:
                return (*other_args, *a_prmt_func(kernel_i, kernel_j))

        if not propagate_inter_a:
            sch.transform_layout(AS, ("read", 0),
                                    A_permutation)
        if not propagate_inter_b:
            sch.transform_layout(BS, ("read", 0),
                                    B_permutation)
        
        def cooperative_fetch(block, dims=4, vec=1, use_pragma_unroll=False, force_async_copy=False):
            read_dtype = sch.get_sref(block).stmt.reads[-1].buffer.dtype
            out_dtype = sch.get_sref(block).stmt.writes[-1].buffer.dtype
            if read_dtype == "int8" and out_dtype == "float16":
                vec = 4
            
            loops = sch.get_loops(block)[-dims:]
            other_loop = loops[:-1]
            shared_j = loops[-1]
            shared_j, shared_vi = sch.split(shared_j, factors=[None, vec])
            sch.vectorize(shared_vi)
            if force_async_copy and read_dtype == out_dtype:
                sch.tensorize(shared_vi, ASYNC_COPY_F16_X8_INTRIN if read_dtype == "float16" else ASYNC_COPY_S8_X16_INTRIN)
                sch.annotate(ki, "pragma_commit_wait", "")
            shared_fused = sch.fuse(*other_loop, shared_j)
            shared_inner, shared_ty, shared_tz, shared_tx = sch.split(
                shared_fused, factors=[None, block_row_warps, block_col_warps, warp_size])
            sch.bind(shared_tx, "threadIdx.x")
            sch.bind(shared_ty, "threadIdx.y")
            sch.bind(shared_tz, "threadIdx.z")
            self.sche.unroll(shared_inner)
            if use_pragma_unroll:
                self.sche.annotate(shared_inner, "pragma_unroll_explicit", False)
        if is_a_consistent:
            cooperative_fetch(AS, dims=4, vec=vecA, use_pragma_unroll=True, force_async_copy=(propagate_inter_a and not propagate_inter_b))
        if is_b_consistent:
            cooperative_fetch(BS, dims=4, vec=vecB, use_pragma_unroll=True, force_async_copy=(propagate_inter_b and not propagate_inter_a))
        
        # handle shared scope
        if not is_a_consistent and not is_b_consistent:
            if compute_dtype == "int32":
                a_smem_store_vec = 16
            elif compute_dtype == "float16" or compute_dtype == "float32":
                a_smem_store_vec = 8
            if compute_dtype == "int32":
                b_smem_store_vec = 16
            elif compute_dtype == "float16" or compute_dtype == "float32":
                b_smem_store_vec = 8
            # cache_decompress
            A_shared_jj = sch.get_loops(AS)[-1]
            A_shared_jj, A_shared_vi, A_shared_vj = sch.split(A_shared_jj, factors=[None, 1, a_smem_store_vec])
            B_shared_jj = sch.get_loops(BS)[-1]
            B_shared_jj, B_shared_vi, B_shared_vj = sch.split(B_shared_jj, factors=[None, 1, b_smem_store_vec])
            block_local_A_decompress = sch.cache_read(AS, 0, "local")
            block_local_B_decompress = sch.cache_read(BS, 0, "local")
            
            A_decode_block = None
            B_decode_block = None
            other_blocks = []
            for op in reversed(self.ops):
                if op not in (self.reduce_op, *[arg.op for arg in self.output_args]):
                    if 'B_decode' in op.name or 'decompress' in op.name or 'mediate0' in op.name:
                        B_decode_block = self.sche.get_block(op.name)
                    elif 'A_decode' in op.name:
                        A_decode_block = self.sche.get_block(op.name)
                    else:
                        other_blocks.append(self.sche.get_block(op.name))
            for block in other_blocks:
                self.sche.compute_inline(block)
            if self.reduce_op != None and self.reduce_op != self.output_op:
                block = self.sche.get_block(self.output_op.name)
                self.sche.reverse_compute_inline(block)

            write_sch(sch, "schedule_compute_inline")
            if A_decode_block != None:
                if shared_cache_scale:
                    ASS = sch.cache_read(A_decode_block, 1, "shared")
                read_shape = sch.get_sref(A_decode_block).stmt.reads[0].buffer.shape
                write_shape = sch.get_sref(A_decode_block).stmt.writes[0].buffer.shape
                compress_rate = np.prod(write_shape) // np.prod(read_shape) 
                if self.args[0].dtype == 'float16':
                    bits = 16 // compress_rate
                elif self.args[0].dtype == 'int8':
                    bits = 8 // compress_rate
                sch.compute_inline(A_decode_block)   
            if B_decode_block != None:
                if shared_cache_scale:
                    BSS = sch.cache_read(B_decode_block, 1, "shared")
                read_shape = sch.get_sref(B_decode_block).stmt.reads[0].buffer.shape
                write_shape = sch.get_sref(B_decode_block).stmt.writes[0].buffer.shape
                compress_rate = np.prod(write_shape) // np.prod(read_shape) 
                if self.args[0].dtype == 'float16':
                    bits = 16 // compress_rate
                elif self.args[0].dtype == 'int8':
                    bits = 8 // compress_rate
                sch.compute_inline(B_decode_block)
            write_sch(sch, "decode_inline")  
            if is_lut:            
                block_local_A_shared_cache = sch.cache_read(block_local_A_decompress, 1, "shared")
                block_local_B_shared_cache = sch.cache_read(block_local_B_decompress, 1, "shared")
                block_local_A_shared_cache_local = sch.cache_read(block_local_A_decompress, 1, "local")
                block_local_B_shared_cache_local = sch.cache_read(block_local_B_decompress, 1, "local")
            else:
                block_local_A_shared_cache = sch.cache_read(block_local_A_decompress, 0, "shared")
                block_local_B_shared_cache = sch.cache_read(block_local_B_decompress, 0, "shared")
                block_local_A_shared_cache_local = sch.cache_read(block_local_A_decompress, 0, "local")
                block_local_B_shared_cache_local = sch.cache_read(block_local_B_decompress, 0, "local")
            
            write_sch(sch, "decode_compute_inline")  
            sch.compute_at(block_local_A_decompress, A_shared_vi)
            write_sch(sch, "decode_compute_at_block_local_A_decompress")  
            sch.compute_at(block_local_A_shared_cache_local, A_shared_vi)
            write_sch(sch, "decode_compute_at_block_local_A_shared_cache_local")  

            sch.compute_at(block_local_B_decompress, B_shared_vi)
            sch.compute_at(block_local_B_shared_cache_local, B_shared_vi)
            write_sch(sch, "decode_compute_at")  
            # fast decoding
            if self.config.fast_decoding:
                from ladder.schedule.lop3_intrin import (
                    LOP3_FAST_DECODE_INT4_TO_FP16_INTRIN,
                    LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN,
                    LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN_L16,
                    LOP3_FAST_DECODE_INT2_TO_INT8_INTRIN_L8,
                    LOP3_FAST_DECODE_INT2_TO_INT8_INTRIN_L16,
                    LOP3_FAST_DECODE_INT1_TO_INT8_INTRIN_L16,
                )
                if self.args[0].dtype == 'float16':
                    sch.tensorize(sch.get_loops(block_local_A_decompress)[-1], LOP3_FAST_DECODE_INT4_TO_FP16_INTRIN)
                elif self.args[0].dtype == 'int8':
                    loop = sch.get_loops(block_local_A_decompress)[-1]
                    loop_extent = sch.get_sref(loop).stmt.extent
                    # compute the decode bits.
                    if bits == 4:
                        if loop_extent == 16:
                            sch.tensorize(sch.get_loops(block_local_A_decompress)[-1], LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN_L16)
                        elif loop_extent == 8:
                            sch.tensorize(sch.get_loops(block_local_A_decompress)[-1], LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN)
                    elif bits == 2:
                        if loop_extent == 16:
                            sch.tensorize(loop, LOP3_FAST_DECODE_INT2_TO_INT8_INTRIN_L16)
                    elif bits == 1:
                        sch.tensorize(sch.get_loops(block_local_A_decompress)[-1], LOP3_FAST_DECODE_INT1_TO_INT8_INTRIN_L16)
                if self.args[0].dtype == 'float16':
                        sch.tensorize(sch.get_loops(block_local_B_decompress)[-1], LOP3_FAST_DECODE_INT4_TO_FP16_INTRIN)
                elif self.args[0].dtype == 'int8':
                    loop = sch.get_loops(block_local_B_decompress)[-1]
                    loop_extent = sch.get_sref(loop).stmt.extent
                    # compute the decode bits.
                    if bits == 4:
                        if loop_extent == 16:
                            sch.tensorize(sch.get_loops(block_local_B_decompress)[-1], LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN_L16)
                        elif loop_extent == 8:
                            sch.tensorize(sch.get_loops(block_local_B_decompress)[-1], LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN)
                    elif bits == 2:
                        if loop_extent == 16:
                            sch.tensorize(loop, LOP3_FAST_DECODE_INT2_TO_INT8_INTRIN_L16)
                    elif bits == 1:
                        sch.tensorize(sch.get_loops(block_local_B_decompress)[-1], LOP3_FAST_DECODE_INT1_TO_INT8_INTRIN_L16)

            A_shared_fused = sch.fuse(*sch.get_loops(AS)[-6:-2])
            A_shared_inner, A_shared_ty, A_shared_tz, A_shared_tx = sch.split(
                A_shared_fused, factors=[None, block_row_warps, block_col_warps, warp_size])
            sch.compute_at(block_local_A_shared_cache, ko, preserve_unit_loops=True)
            sch.bind(A_shared_tx, "threadIdx.x")
            sch.bind(A_shared_ty, "threadIdx.y")
            sch.bind(A_shared_tz, "threadIdx.z")
            sch.vectorize(sch.get_loops(AS)[-1])
            sch.vectorize(sch.get_loops(block_local_A_shared_cache_local)[-1])
            block_local_A_shared_cache_fused = sch.fuse(*sch.get_loops(block_local_A_shared_cache)[-4:])
            
            assert (self.args[1].dtype == "int8"), "currently only support b stored in int8 format"
            
            _extent_for_bcache = sch.get_sref(block_local_A_shared_cache_fused).stmt.extent
            if _extent_for_bcache // (vecA * warp_size) == 0:
                block_local_A_shared_cache_fused, A_shared_tx, A_shared_vi = sch.split(
                    block_local_A_shared_cache_fused, factors=[1, warp_size, None])
                sch.bind(A_shared_tx, "threadIdx.x")
                sch.vectorize(A_shared_vi)
                _extent_for_bcache = sch.get_sref(block_local_A_shared_cache_fused).stmt.extent

            # warp_size - 1 handling for 32x7 alike case, which may cause unaligned threadIdx.x mapping.
            if _extent_for_bcache // vecA >= 1 and _extent_for_bcache & (vecA - 1) == 0 and (_extent_for_bcache // vecA) & (warp_size -1) == 0:
                block_local_A_shared_cache_fused, A_shared_vi = sch.split(
                block_local_A_shared_cache_fused, factors=[None, vecA])
                sch.vectorize(A_shared_vi)
                _extent_for_bcache = sch.get_sref(block_local_A_shared_cache_fused).stmt.extent

            if _extent_for_bcache // warp_size >= 1 and _extent_for_bcache % warp_size == 0:
                block_local_A_shared_cache_fused, A_shared_tx = sch.split(
                    block_local_A_shared_cache_fused, factors=[None, warp_size])
                sch.bind(A_shared_tx, "threadIdx.x")
                _extent_for_bcache = sch.get_sref(block_local_A_shared_cache_fused).stmt.extent

            if _extent_for_bcache // block_row_warps >= 1 and _extent_for_bcache % block_row_warps == 0:
                block_local_A_shared_cache_fused, A_shared_ty = sch.split(
                    block_local_A_shared_cache_fused, factors=[None, block_row_warps])
                sch.bind(A_shared_ty, "threadIdx.y")
                _extent_for_bcache = sch.get_sref(block_local_A_shared_cache_fused).stmt.extent
            
            if _extent_for_bcache // block_col_warps >= 1 and _extent_for_bcache % block_col_warps == 0:
                block_local_A_shared_cache_fused, A_shared_tz = sch.split(
                    block_local_A_shared_cache_fused, factors=[None, block_col_warps])
                sch.bind(A_shared_tz, "threadIdx.z")
                _extent_for_bcache = sch.get_sref(block_local_A_shared_cache_fused).stmt.extent

            if is_lut:
                block_shared_lut = sch.cache_read(block_local_A_decompress, 0, "shared")
                sch.reverse_compute_at(block_shared_lut, j)
                _, A_shared_tx = sch.split(
                    sch.get_loops(block_shared_lut)[-1], factors=[None, warp_size])
                sch.bind(A_shared_tx, "threadIdx.x")
            
            B_shared_fused = sch.fuse(*sch.get_loops(BS)[-6:-2])
            B_shared_inner, B_shared_ty, B_shared_tz, B_shared_tx = sch.split(
                B_shared_fused, factors=[None, block_row_warps, block_col_warps, warp_size])
            sch.compute_at(block_local_B_shared_cache, ko, preserve_unit_loops=True)
            sch.bind(B_shared_tx, "threadIdx.x")
            sch.bind(B_shared_ty, "threadIdx.y")
            sch.bind(B_shared_tz, "threadIdx.z")
            sch.vectorize(sch.get_loops(BS)[-1])
            sch.vectorize(sch.get_loops(block_local_B_shared_cache_local)[-1])

            if shared_cache_scale:
                sch.compute_at(ASS, ko, preserve_unit_loops=True)
                sch.compute_at(BSS, ko, preserve_unit_loops=True)
                SS_tx, SS_v = sch.split(sch.get_loops(ASS)[-1], factors=[warp_size, None])
                sch.bind(SS_tx, "threadIdx.x")
                extent = sch.get_sref(SS_v).stmt.extent
                if extent in [1, 2, 4, 8, 16]:
                    sch.vectorize(SS_v)
            
                SS_tx, SS_v = sch.split(sch.get_loops(BSS)[-1], factors=[warp_size, None])
                sch.bind(SS_tx, "threadIdx.x")
                extent = sch.get_sref(SS_v).stmt.extent
                if extent in [1, 2, 4, 8, 16]:
                    sch.vectorize(SS_v)
            
            block_local_B_shared_cache_fused = sch.fuse(*sch.get_loops(block_local_B_shared_cache)[-4:])
            assert (self.args[1].dtype == "int8"), "currently only support b stored in int8 format"
            
            _extent_for_bcache = sch.get_sref(block_local_B_shared_cache_fused).stmt.extent
            if _extent_for_bcache // (vecB * warp_size) == 0:
                block_local_B_shared_cache_fused, B_shared_tx, B_shared_vi = sch.split(
                    block_local_B_shared_cache_fused, factors=[1, warp_size, None])
                sch.bind(B_shared_tx, "threadIdx.x")
                sch.vectorize(B_shared_vi)
                _extent_for_bcache = sch.get_sref(block_local_B_shared_cache_fused).stmt.extent

            # warp_size - 1 handling for 32x7 alike case, which may cause unaligned threadIdx.x mapping.
            if _extent_for_bcache // vecB >= 1 and _extent_for_bcache & (vecB - 1) == 0 and (_extent_for_bcache // vecB) & (warp_size -1) == 0:
                block_local_B_shared_cache_fused, B_shared_vi = sch.split(
                block_local_B_shared_cache_fused, factors=[None, vecB])
                sch.vectorize(B_shared_vi)
                _extent_for_bcache = sch.get_sref(block_local_B_shared_cache_fused).stmt.extent

            if _extent_for_bcache // warp_size >= 1 and _extent_for_bcache % warp_size == 0:
                block_local_B_shared_cache_fused, B_shared_tx = sch.split(
                    block_local_B_shared_cache_fused, factors=[None, warp_size])
                sch.bind(B_shared_tx, "threadIdx.x")
                _extent_for_bcache = sch.get_sref(block_local_B_shared_cache_fused).stmt.extent

            if _extent_for_bcache // block_row_warps >= 1 and _extent_for_bcache % block_row_warps == 0:
                block_local_B_shared_cache_fused, B_shared_ty = sch.split(
                    block_local_B_shared_cache_fused, factors=[None, block_row_warps])
                sch.bind(B_shared_ty, "threadIdx.y")
                _extent_for_bcache = sch.get_sref(block_local_B_shared_cache_fused).stmt.extent
            
            if _extent_for_bcache // block_col_warps >= 1 and _extent_for_bcache % block_col_warps == 0:
                block_local_B_shared_cache_fused, B_shared_tz = sch.split(
                    block_local_B_shared_cache_fused, factors=[None, block_col_warps])
                sch.bind(B_shared_tz, "threadIdx.z")
                _extent_for_bcache = sch.get_sref(block_local_B_shared_cache_fused).stmt.extent

            if is_lut:
                block_shared_lut = sch.cache_read(block_local_B_decompress, 0, "shared")
                sch.reverse_compute_at(block_shared_lut, j)
                _, B_shared_tx = sch.split(
                    sch.get_loops(block_shared_lut)[-1], factors=[None, warp_size])
                sch.bind(B_shared_tx, "threadIdx.x")
        else:
            # handle local decode
            if not is_a_consistent:
                if compute_dtype == "int32":
                    a_smem_store_vec = 16
                elif compute_dtype == "float16" or compute_dtype == "float32":
                    a_smem_store_vec = 8
                # cache_decompress
                A_shared_jj = sch.get_loops(AS)[-1]
                A_shared_jj, A_shared_vi, A_shared_vj = sch.split(A_shared_jj, factors=[None, 1, a_smem_store_vec])
                block_local_A_decompress = sch.cache_read(AS, 0, "local")
                write_sch(sch, "schedule_compute_inline")
                
                decode_block = None
                other_blocks = []
                for op in reversed(self.ops):
                    if op not in (self.reduce_op, *[arg.op for arg in self.output_args]):
                        if 'A_decode' in op.name:
                            decode_block = self.sche.get_block(op.name)

                if decode_block != None:
                    read_shape = sch.get_sref(decode_block).stmt.reads[0].buffer.shape
                    write_shape = sch.get_sref(decode_block).stmt.writes[0].buffer.shape
                    compress_rate = np.prod(write_shape) // np.prod(read_shape) 
                    if self.args[0].dtype == 'float16':
                        bits = 16 // compress_rate
                    elif self.args[0].dtype == 'int8':
                        bits = 8 // compress_rate
                    sch.compute_inline(decode_block)   

                if is_lut:            
                    block_local_A_shared_cache = sch.cache_read(block_local_A_decompress, 1, "shared")
                    block_local_A_shared_cache_local = sch.cache_read(block_local_A_decompress, 1, "local")
                else:
                    block_local_A_shared_cache = sch.cache_read(block_local_A_decompress, 0, "shared")
                    block_local_A_shared_cache_local = sch.cache_read(block_local_A_decompress, 0, "local")

                sch.compute_at(block_local_A_decompress, A_shared_vi)
                sch.compute_at(block_local_A_shared_cache_local, A_shared_vi)

                # fast decoding
                if self.config.fast_decoding:
                    from ladder.schedule.lop3_intrin import (
                        LOP3_FAST_DECODE_INT4_TO_FP16_INTRIN,
                        LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN,
                        LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN_L16,
                        LOP3_FAST_DECODE_INT2_TO_INT8_INTRIN_L8,
                        LOP3_FAST_DECODE_INT2_TO_INT8_INTRIN_L16,
                        LOP3_FAST_DECODE_INT1_TO_INT8_INTRIN_L16,
                    )
                    if self.args[0].dtype == 'float16':
                        sch.tensorize(sch.get_loops(block_local_A_decompress)[-1], LOP3_FAST_DECODE_INT4_TO_FP16_INTRIN)
                    elif self.args[0].dtype == 'int8':
                        loop = sch.get_loops(block_local_A_decompress)[-1]
                        loop_extent = sch.get_sref(loop).stmt.extent
                        # compute the decode bits.
                        if bits == 4:
                            if loop_extent == 16:
                                sch.tensorize(sch.get_loops(block_local_A_decompress)[-1], LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN_L16)
                            elif loop_extent == 8:
                                sch.tensorize(sch.get_loops(block_local_A_decompress)[-1], LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN)
                        elif bits == 2:
                            if loop_extent == 16:
                                sch.tensorize(loop, LOP3_FAST_DECODE_INT2_TO_INT8_INTRIN_L16)
                        elif bits == 1:
                            sch.tensorize(sch.get_loops(block_local_A_decompress)[-1], LOP3_FAST_DECODE_INT1_TO_INT8_INTRIN_L16)

                A_shared_fused = sch.fuse(*sch.get_loops(AS)[-6:-2])
                A_shared_inner, A_shared_ty, A_shared_tz, A_shared_tx = sch.split(
                    A_shared_fused, factors=[None, block_row_warps, block_col_warps, warp_size])
                sch.compute_at(block_local_A_shared_cache, ko, preserve_unit_loops=True)

                sch.bind(A_shared_tx, "threadIdx.x")
                sch.bind(A_shared_ty, "threadIdx.y")
                sch.bind(A_shared_tz, "threadIdx.z")
                sch.vectorize(sch.get_loops(AS)[-1])
                sch.vectorize(sch.get_loops(block_local_A_shared_cache_local)[-1])

                block_local_A_shared_cache_fused = sch.fuse(*sch.get_loops(block_local_A_shared_cache)[-4:])
                assert (self.args[1].dtype == "int8"), "currently only support b stored in int8 format"
                
                _extent_for_bcache = sch.get_sref(block_local_A_shared_cache_fused).stmt.extent
                if _extent_for_bcache // (vecA * warp_size) == 0:
                    block_local_A_shared_cache_fused, A_shared_tx, A_shared_vi = sch.split(
                        block_local_A_shared_cache_fused, factors=[1, warp_size, None])
                    sch.bind(A_shared_tx, "threadIdx.x")
                    sch.vectorize(A_shared_vi)
                    _extent_for_bcache = sch.get_sref(block_local_A_shared_cache_fused).stmt.extent

                # warp_size - 1 handling for 32x7 alike case, which may cause unaligned threadIdx.x mapping.
                if _extent_for_bcache // vecA >= 1 and _extent_for_bcache & (vecA - 1) == 0 and (_extent_for_bcache // vecA) & (warp_size -1) == 0:
                    block_local_A_shared_cache_fused, A_shared_vi = sch.split(
                    block_local_A_shared_cache_fused, factors=[None, vecA])
                    sch.vectorize(A_shared_vi)
                    _extent_for_bcache = sch.get_sref(block_local_A_shared_cache_fused).stmt.extent

                if _extent_for_bcache // warp_size >= 1 and _extent_for_bcache % warp_size == 0:
                    block_local_A_shared_cache_fused, A_shared_tx = sch.split(
                        block_local_A_shared_cache_fused, factors=[None, warp_size])
                    sch.bind(A_shared_tx, "threadIdx.x")
                    _extent_for_bcache = sch.get_sref(block_local_A_shared_cache_fused).stmt.extent

                if _extent_for_bcache // block_row_warps >= 1 and _extent_for_bcache % block_row_warps == 0:
                    block_local_A_shared_cache_fused, A_shared_ty = sch.split(
                        block_local_A_shared_cache_fused, factors=[None, block_row_warps])
                    sch.bind(A_shared_ty, "threadIdx.y")
                    _extent_for_bcache = sch.get_sref(block_local_A_shared_cache_fused).stmt.extent
                
                if _extent_for_bcache // block_col_warps >= 1 and _extent_for_bcache % block_col_warps == 0:
                    block_local_A_shared_cache_fused, A_shared_tz = sch.split(
                        block_local_A_shared_cache_fused, factors=[None, block_col_warps])
                    sch.bind(A_shared_tz, "threadIdx.z")
                    _extent_for_bcache = sch.get_sref(block_local_A_shared_cache_fused).stmt.extent

                if is_lut:
                    block_shared_lut = sch.cache_read(block_local_A_decompress, 0, "shared")
                    sch.reverse_compute_at(block_shared_lut, j)
                    _, A_shared_tx = sch.split(
                        sch.get_loops(block_shared_lut)[-1], factors=[None, warp_size])
                    sch.bind(A_shared_tx, "threadIdx.x")

            if not is_b_consistent:
                # todo(leiwang): bugs of tvm that consider int8 as int32 in some cases should be investigated.
                if self.config.ladder_compute_type == "int4":
                    b_smem_store_vec = 16
                elif compute_dtype == "int32":
                    b_smem_store_vec = 16
                elif compute_dtype == "float16" or compute_dtype == "float32":
                    b_smem_store_vec = 8
                # cache_decompress
                B_shared_jj = sch.get_loops(BS)[-1]
                B_shared_jj, B_shared_vi, B_shared_vj = sch.split(B_shared_jj, factors=[None, 1, b_smem_store_vec])
                block_local_B_decompress = sch.cache_read(BS, 0, "local")
                write_sch(sch, "schedule_compute_inline")
                decode_block = None
                other_blocks = []
                for op in reversed(self.ops):
                    if op not in (self.reduce_op, *[arg.op for arg in self.output_args]):
                        if 'B_decode' in op.name or 'decompress' in op.name or 'mediate0' in op.name:
                            decode_block = self.sche.get_block(op.name)
                        else:
                            other_blocks.append(self.sche.get_block(op.name))
                for block in other_blocks:
                    self.sche.compute_inline(block)
                if self.reduce_op != None and self.reduce_op != self.output_op:
                    block = self.sche.get_block(self.output_op.name)
                    self.sche.reverse_compute_inline(block)
                if decode_block != None:
                    if shared_cache_scale:
                        SS = sch.cache_read(decode_block, 1, "shared")
                    read_shape = sch.get_sref(decode_block).stmt.reads[0].buffer.shape
                    write_shape = sch.get_sref(decode_block).stmt.writes[0].buffer.shape
                    compress_rate = np.prod(write_shape) // np.prod(read_shape) 
                    if self.args[0].dtype == 'float16':
                        bits = 16 // compress_rate
                    elif self.args[0].dtype == 'int8':
                        bits = 8 // compress_rate
                    sch.compute_inline(decode_block)   
                if is_lut:            
                    block_local_B_shared_cache = sch.cache_read(block_local_B_decompress, 1, "shared")
                    block_local_B_shared_cache_local = sch.cache_read(block_local_B_decompress, 1, "local")
                else:
                    block_local_B_shared_cache = sch.cache_read(block_local_B_decompress, 0, "shared")
                    block_local_B_shared_cache_local = sch.cache_read(block_local_B_decompress, 0, "local")

                sch.compute_at(block_local_B_decompress, B_shared_vi)
                sch.compute_at(block_local_B_shared_cache_local, B_shared_vi)

                # fast decoding
                if self.config.fast_decoding:
                    from ladder.schedule.lop3_intrin import (
                        LOP3_FAST_DECODE_INT4_TO_FP16_INTRIN,
                        LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN,
                        LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN_L16,
                        LOP3_FAST_DECODE_INT2_TO_INT8_INTRIN_L8,
                        LOP3_FAST_DECODE_INT2_TO_INT8_INTRIN_L16,
                        LOP3_FAST_DECODE_INT1_TO_INT8_INTRIN_L16,
                    )
                    if self.args[0].dtype == 'float16':
                        sch.tensorize(sch.get_loops(block_local_B_decompress)[-1], LOP3_FAST_DECODE_INT4_TO_FP16_INTRIN)
                    elif self.args[0].dtype == 'int8':
                        loop = sch.get_loops(block_local_B_decompress)[-1]
                        loop_extent = sch.get_sref(loop).stmt.extent
                        # compute the decode bits.
                        if bits == 4:
                            if loop_extent == 16:
                                sch.tensorize(sch.get_loops(block_local_B_decompress)[-1], LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN_L16)
                            elif loop_extent == 8:
                                sch.tensorize(sch.get_loops(block_local_B_decompress)[-1], LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN)
                        elif bits == 2:
                            if loop_extent == 16:
                                sch.tensorize(loop, LOP3_FAST_DECODE_INT2_TO_INT8_INTRIN_L16)
                        elif bits == 1:
                            sch.tensorize(sch.get_loops(block_local_B_decompress)[-1], LOP3_FAST_DECODE_INT1_TO_INT8_INTRIN_L16)

                B_shared_fused = sch.fuse(*sch.get_loops(BS)[-6:-2])
                B_shared_inner, B_shared_ty, B_shared_tz, B_shared_tx = sch.split(
                    B_shared_fused, factors=[None, block_row_warps, block_col_warps, warp_size])
                sch.compute_at(block_local_B_shared_cache, ko, preserve_unit_loops=True)
                if shared_cache_scale:
                    sch.compute_at(SS, ko, preserve_unit_loops=True)
                sch.bind(B_shared_tx, "threadIdx.x")
                sch.bind(B_shared_ty, "threadIdx.y")
                sch.bind(B_shared_tz, "threadIdx.z")
                sch.vectorize(sch.get_loops(BS)[-1])
                sch.vectorize(sch.get_loops(block_local_B_shared_cache_local)[-1])
                if shared_cache_scale:
                    SS_tx, SS_v = sch.split(sch.get_loops(SS)[-1], factors=[warp_size, None])
                    sch.bind(SS_tx, "threadIdx.x")
                    extent = sch.get_sref(SS_v).stmt.extent
                    if extent in [1, 2, 4, 8, 16]:
                        sch.vectorize(SS_v)
                
                block_local_B_shared_cache_fused = sch.fuse(*sch.get_loops(block_local_B_shared_cache)[-4:])
                assert (self.args[1].dtype == "int8"), "currently only support b stored in int8 format"
                
                _extent_for_bcache = sch.get_sref(block_local_B_shared_cache_fused).stmt.extent
                if _extent_for_bcache // (vecB * warp_size) == 0:
                    # block_local_B_shared_cache_fused, B_shared_vi = sch.split(
                    #   block_local_B_shared_cache_fused, factors=[None, vecB],
                    # )
                    # sch.vectorize(B_shared_vi)
                    block_local_B_shared_cache_fused, B_shared_tx, B_shared_vi = sch.split(
                        block_local_B_shared_cache_fused, factors=[1, warp_size, None])
                    sch.bind(B_shared_tx, "threadIdx.x")
                    if sch.get_sref(B_shared_vi).stmt.extent in [1, 2, 4, 8, 16] and self.config.ladder_compute_type != "int4":
                        sch.vectorize(B_shared_vi)
                    _extent_for_bcache = sch.get_sref(block_local_B_shared_cache_fused).stmt.extent

                # warp_size - 1 handling for 32x7 alike case, which may cause unaligned threadIdx.x mapping.
                if _extent_for_bcache // vecB >= 1 and _extent_for_bcache & (vecB - 1) == 0 and (_extent_for_bcache // vecB) & (warp_size -1) == 0:
                    block_local_B_shared_cache_fused, B_shared_vi = sch.split(
                    block_local_B_shared_cache_fused, factors=[None, vecB])
                    sch.vectorize(B_shared_vi)
                    _extent_for_bcache = sch.get_sref(block_local_B_shared_cache_fused).stmt.extent

                if _extent_for_bcache // warp_size >= 1 and _extent_for_bcache % warp_size == 0:
                    block_local_B_shared_cache_fused, B_shared_tx = sch.split(
                        block_local_B_shared_cache_fused, factors=[None, warp_size])
                    sch.bind(B_shared_tx, "threadIdx.x")
                    _extent_for_bcache = sch.get_sref(block_local_B_shared_cache_fused).stmt.extent

                if _extent_for_bcache // block_row_warps >= 1 and _extent_for_bcache % block_row_warps == 0:
                    block_local_B_shared_cache_fused, B_shared_ty = sch.split(
                        block_local_B_shared_cache_fused, factors=[None, block_row_warps])
                    sch.bind(B_shared_ty, "threadIdx.y")
                    _extent_for_bcache = sch.get_sref(block_local_B_shared_cache_fused).stmt.extent
                
                if _extent_for_bcache // block_col_warps >= 1 and _extent_for_bcache % block_col_warps == 0:
                    block_local_B_shared_cache_fused, B_shared_tz = sch.split(
                        block_local_B_shared_cache_fused, factors=[None, block_col_warps])
                    sch.bind(B_shared_tz, "threadIdx.z")
                    _extent_for_bcache = sch.get_sref(block_local_B_shared_cache_fused).stmt.extent

                if is_lut:
                    block_shared_lut = sch.cache_read(block_local_B_decompress, 0, "shared")
                    sch.reverse_compute_at(block_shared_lut, j)
                    _, B_shared_tx = sch.split(
                        sch.get_loops(block_shared_lut)[-1], factors=[None, warp_size])
                    sch.bind(B_shared_tx, "threadIdx.x")

        if shared_cache_c:
            sch.reverse_compute_at(
                C_shared,
                sch.get_loops(C_warp)[-3],
                preserve_unit_loops=True,
            )
        
        def schedule_shared_output(block):
            o_shared_fused = sch.fuse(*sch.get_loops(block)[-4:])
            if sch.get_sref(o_shared_fused).stmt.extent // warp_size < vecC:
                vec = sch.get_sref(o_shared_fused).stmt.extent // warp_size
            else:
                vec = vecC
            oo, o_shared_tx, o_shared_vi = sch.split(
                o_shared_fused, factors=[None, warp_size, vec]
            )     
            sch.vectorize(o_shared_vi)
            sch.bind(o_shared_tx, "threadIdx.x")
            sch.unroll(oo)
            sch.annotate(oo, "pragma_unroll_explicit", False)

        if shared_cache_c:
            schedule_shared_output(C_shared)
        
        write_sch(sch, "schedule_warp")
        
        
        write_sch(sch, "schedule_shared")
        # ------------------------ Warp memory layout for multiplicand A and B ------------------------
        AW = sch.cache_read(C, 0, "warp")
        BW = sch.cache_read(C, 1, "warp")
        sch.compute_at(AW, ki, preserve_unit_loops=True)
        sch.compute_at(BW, ki, preserve_unit_loops=True)

        a_index_map = shared_16x32_to_ldmatrix_32x16_layout if compute_dtype == "int32" else A_B_shared_16x16_to_ldmatrix_32x8_layout
        b_index_map = shared_16x32_to_ldmatrix_32x16_layout if compute_dtype == "int32" else A_B_shared_16x16_to_ldmatrix_32x8_layout
        c_index_map = C_shared_16x16_to_ldmatrix_32x8_layout if compute_dtype == "int32" else C_shared_16x16_to_ldmatrix_32x8_layout

        def index_map_A(i, k, wmma_m, wmma_k):
            return (i, k, *a_index_map(wmma_m, wmma_k))

        def index_map_B(*args):
            kernel_i, kernel_j = args[-2], args[-1]
            other_args = args[:-2]
            result = (*other_args, *b_index_map(kernel_i, kernel_j))
            return result

        def index_map_C(m, n, wmma_m, wmma_n):
            return (m, n, *c_index_map(wmma_m, wmma_n),)

        sch.transform_layout(AW, ("write", 0), index_map_A)
        sch.transform_layout(BW, ("write", 0), index_map_A if not transpose_B else index_map_B)
        sch.transform_layout(C_warp, ("read", 0), index_map_C)
        
        # ------------------------ Tensorize and Pipelining -------------------------
        init_intrin = TRICKY_MMA_fill_16x16_i32_INTRIN if compute_dtype == "int32" else TRICKY_MMA_fill_16x16_f16_INTRIN if compute_dtype == "float16" else TRICKY_MMA_fill_16x16_f32_INTRIN
        load_a_intrin = TRICKY_LDMATRIX_16x32_A_INTRIN if compute_dtype == "int32" else TRICKY_LDMATRIX_16x16_A_INTRIN
        load_b_intrin = TRICKY_LDMATRIX_32x16_B_INTRIN if compute_dtype == "int32" else TRICKY_LDMATRIX_16x16_B_INTRIN
        load_b_intrin_trans = TRICKY_LDMATRIX_16x32_B_TRANS_INTRIN if compute_dtype == "int32" else TRICKY_LDMATRIX_16x16_B_TRANS_INTRIN
        compute_intrin = TRICKY_MMA_i8i8i32_INTRIN if compute_dtype == "int32" else TRICKY_MMA_f16f16f16_INTRIN
        compute_trans_intrin = TRICKY_MMA_i8i8i32_TRANS_INTRIN if compute_dtype == "int32" else TRICKY_MMA_f16f16f16_TRANS_INTRIN if compute_dtype == "float16" else TRICKY_MMA_f16f16f32_TRANS_INTRIN
        if shared_cache_c:
            store_intrin = TRICKY_MMA_store_16x16_i32_shared_INTRIN if compute_dtype == "int32" else  TRICKY_MMA_store_16x16_f16_shared_INTRIN if compute_dtype == "float16" else TRICKY_MMA_store_16x16_f32_shared_INTRIN
        else:
            store_intrin = TRICKY_MMA_store_16x16_i32_global_INTRIN if compute_dtype == "int32" else  TRICKY_MMA_store_16x16_f16_global_INTRIN if compute_dtype == "float16" else TRICKY_MMA_store_16x16_f32_global_INTRIN
        init_block_b = sch.decompose_reduction(C, ko)
        write_sch(sch, "decompose_reduction")
        init_block_b_loops = sch.get_loops(init_block_b)
        init_block_b_i, init_block_b_j = sch.get_loops(init_block_b)[-4:-2]
        sch.annotate(init_block_b_i, "pragma_unroll_explicit", False)
        sch.annotate(init_block_b_j, "pragma_unroll_explicit", False)
        sch.tensorize(sch.get_loops(init_block_b)[-2], init_intrin)
        sch.tensorize(
            sch.get_loops(AW)[-2], load_a_intrin
        )
        sch.tensorize(
            sch.get_loops(
                BW)[-2], load_b_intrin if not transpose_B else load_b_intrin_trans
        )
        sch.tensorize(
            kernel_i, compute_intrin if not transpose_B else compute_trans_intrin)
        sch.tensorize(
            sch.get_loops(C_warp)[-2],
            store_intrin,
        )
        write_sch(sch, "tensorize_store")
        if raster > 0:
            sch.annotate(init_block_b_loops[-4], ann_key="thread_rasterization", ann_val=raster)
        if stage > 1:
            if shared_cache_scale:
                if is_a_consistent:
                    sch.annotate(ko, ann_key="software_pipeline_stage", ann_val=[0, 0, 0, stage - 1, stage - 1])
                    sch.annotate(ko, ann_key="software_pipeline_order", ann_val=[0, 1, 2, 3, 4])
                else:
                    sch.annotate(ko, ann_key="software_pipeline_stage", ann_val=[0, 0, stage - 1, 0, 0, stage - 1, stage - 1])
                    sch.annotate(ko, ann_key="software_pipeline_order", ann_val=[0, 1, 2, 3, 4, 5, 6])
            else:
                if is_a_consistent:
                    sch.annotate(ko, ann_key="software_pipeline_stage", ann_val=[0, 0, stage - 1, stage - 1])
                    sch.annotate(ko, ann_key="software_pipeline_order", ann_val=[0, 1, 2, 3])
                else:
                    sch.annotate(ko, ann_key="software_pipeline_stage", ann_val=[0, stage - 1, 0, stage - 1, stage - 1])
                    sch.annotate(ko, ann_key="software_pipeline_order", ann_val=[0, 1, 2, 3, 4])
            if use_async:
                sch.annotate(ko, "software_pipeline_async_stages", [0])

        write_sch(sch, "cache_small_tensor")

        return sch.mod["main"]

    
    def schedule_consistent_rocm_cdna(self):
        from .ladder_intrin import (
            thread_id_shared_access_64x1_to_16x4_layout_A,
            thread_id_shared_access_64x4_to_16x16_layout_A,
            thread_id_shared_access_64x1_to_4x16_layout_B,
            thread_id_shared_access_64x4_to_16x16_layout_B,
            thread_id_shared_access_64x4_to_16x16_layout_C,
            shared_16x16_to_local_64x4_layout_A,
            shared_16x16_to_local_64x4_layout_B,
            shared_16x16_to_local_64x4_layout_C,
            HIP_MFMA_fill_16x16_f32_INTRIN,
            HIP_MFMA_LOAD_16x16_A_SHARED_f16_INTRIN,
            HIP_MFMA_LOAD_16x16_B_SHARED_f16_INTRIN,
            HIP_MFMA_LOAD_16x16_B_TRANS_SHARED_f16_INTRIN,
            HIP_MFMA_f16f16f32_INTRIN,
            HIP_MFMA_f16f16f32_TRANS_INTRIN,
            HIP_MFMA_STORE_SHARED_16x16_f32_INTRIN,
            HIP_MFMA_STORE_GLOBAL_16x16_f32_INTRIN,
        )
        # const val for testing
        warp_size = self.config.arch.warp_size
        compute_dtype = self.reduce_op.output(0).dtype
        wmma_k = 16 if compute_dtype == "int32" else 16
        sch, config = self.sche, self.config
        write_sch(sch, "original")
        C = sch.get_block(self.reduce_op.name)
        try:
            i, j, kernel_i, kernel_j, k, kernel_k = sch.get_loops(C)
        except ValueError:
            b, i, j, kernel_i, kernel_j, k, kernel_k = sch.get_loops(C)
            sch.bind(b, "blockIdx.z")
        A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n = config.tc_extra_conf.tc_axis
        propagate_inter_a= config.ladder_config.propagate_inter_a
        propagate_inter_b = config.ladder_config.propagate_inter_b 
        transpose_A = A_ax_k < A_ax_m
        transpose_B = B_ax_k > B_ax_n
        block_tile_M, block_tile_N = self.config.block[0], self.config.block[1]
        warp_tile_M, warp_tile_N = self.config.warp[0], self.config.warp[1]
        out_dtype = self.reduce_op.output(0).dtype
        has_output_op = False
        # todo: currently, the accum output is fp32, which cannot reuse the previous smem buffer (f16 dtype)
        # todo: should enhance tvm to resolve it.
        # if self.reduce_op != None and self.reduce_op != self.output_op:
        #     has_output_op = True
        def get_vec(in_dtype):
            if in_dtype == "float32" or in_dtype == "int32":
                vec = 4
            elif in_dtype == "float16":
                vec = 8
            elif in_dtype == "int8":
                vec = 16
            else:
                raise NotImplementedError("dtype {} not supported".format(in_dtype))
            return vec
        vecA = get_vec(self.args[0].dtype)
        vecB = get_vec(self.args[1].dtype)
        vecC = get_vec(self.args[-1].dtype)
        raster = self.config.raster_factor
        # ------------------------ Block and Warp level job partition ------------------------
        chunk_size = config.rstep[0] // wmma_k
        
        warp_row_tiles = warp_tile_M
        warp_col_tiles = warp_tile_N
        block_row_warps = block_tile_M // warp_tile_M
        block_col_warps = block_tile_N // warp_tile_N
        chunk = chunk_size
        stage = config.pipeline_stage
        use_async = (propagate_inter_a and propagate_inter_b) and stage > 1

        # block_row_warps = 2
        # block_col_warps = 2
        # warp_row_tiles = 4
        # warp_col_tiles = 4
        # chunk = 2
        # stage = 1
        # use_async = 0
        # raster = 16
        
        block_i, i, ii = sch.split(i, factors=[None, block_row_warps, warp_row_tiles])
        block_j, j, jj = sch.split(j, factors=[None, block_col_warps, warp_col_tiles])
        ko, ki = sch.split(k, factors=[None, chunk])
        sch.reorder(block_i, block_j, i, j, ko, ki, ii, jj, kernel_i, kernel_j, kernel_k)

        if self.sche.get_sref(ko).stmt.extent <= 128:
            self.sche.unroll(ko)
            sch.annotate(ko, "pragma_unroll_explicit", False) 

        write_sch(sch, "BlockTile")

        sch.bind(block_i, "blockIdx.y")
        sch.bind(block_j, "blockIdx.x")
        sch.bind(i, "threadIdx.y")
        sch.bind(j, "threadIdx.z")
        self.block_size = (self.config.arch.warp_size, sch.get_sref(i).stmt.extent, sch.get_sref(j).stmt.extent)
        self.grid_size = (sch.get_sref(block_j).stmt.extent, sch.get_sref(block_i).stmt.extent, 1)
        write_sch(sch, "thread_bind")

        # ------------------------ Shared memory layout for multiplicand A and B ------------------------

        AS = sch.cache_read(C, 0, "shared")
        BS = sch.cache_read(C, 1, "shared")
        sch.compute_at(AS, ko, preserve_unit_loops=True)
        sch.compute_at(BS, ko, preserve_unit_loops=True)
        write_sch(sch, "cached_shared")
        
        # ------------------------ Schedule output fragment layout ------------------------
        if has_output_op:
            C_shared = sch.cache_write(C, 0, "shared")
        C_warp = sch.cache_write(C, 0, "warp")
        sch.reverse_compute_at(C_warp, j, preserve_unit_loops=True)
        if has_output_op:
            sch.reverse_compute_at(
                C_shared,
                sch.get_loops(C_warp)[-3],
                preserve_unit_loops=True,
            )
        
        self.schedule_compute_inline()

        def schedule_shared_output(block, _vec):
            o_shared_fused = sch.fuse(*sch.get_loops(block)[-4:])
            _loop_extent = sch.get_sref(o_shared_fused).stmt.extent
            if _vec * warp_size > _loop_extent:
                _vec = _loop_extent // warp_size
            oo, o_shared_tx, o_shared_vi = sch.split(
                o_shared_fused, factors=[None, warp_size, _vec]
            )     
            sch.vectorize(o_shared_vi)
            sch.bind(o_shared_tx, "threadIdx.x")
            sch.unroll(oo)
            sch.annotate(oo, "pragma_unroll_explicit", False)

        if has_output_op:
            schedule_shared_output(C_shared, vecC)
        
        write_sch(sch, "schedule_warp")
        write_sch(sch, "schedule_compute_inline")

        a_prmt_func = thread_id_shared_access_64x4_to_16x16_layout_A
        b_prmt_func = thread_id_shared_access_64x4_to_16x16_layout_B

        def A_permutation(*args):
            kernel_i, kernel_j = args[-2], args[-1]
            other_args = args[:-2]
            result = (*other_args, *a_prmt_func(kernel_i, kernel_j))
            return result

        def B_permutation(*args):
            kernel_i, kernel_j = args[-2], args[-1]
            other_args = args[:-2]
            if transpose_B:
                return (*other_args, *b_prmt_func(kernel_i, kernel_j))
            else:
                return (*other_args, *a_prmt_func(kernel_i, kernel_j))

        if not propagate_inter_a:
            sch.transform_layout(AS, ("read", 0),
                                    A_permutation)
        if not propagate_inter_b:
            sch.transform_layout(BS, ("read", 0),
                                    B_permutation)
        
        def cooperative_fetch(block, dims=4, vec=1, use_pragma_unroll=False, force_async_copy=False):
            loops = sch.get_loops(block)[-dims:]
            other_loop = loops[:-1]
            shared_j = loops[-1]
            shared_j, shared_vi = sch.split(shared_j, factors=[None, vec])
            sch.vectorize(shared_vi)
            shared_fused = sch.fuse(*other_loop, shared_j)
            shared_inner, shared_ty, shared_tz, shared_tx = sch.split(
                shared_fused, factors=[None, block_row_warps, block_col_warps, warp_size])
            sch.bind(shared_tx, "threadIdx.x")
            sch.bind(shared_ty, "threadIdx.y")
            sch.bind(shared_tz, "threadIdx.z")
            self.sche.unroll(shared_inner)
            if use_pragma_unroll:
                self.sche.annotate(shared_inner, "pragma_unroll_explicit", False)

        cooperative_fetch(AS, dims=4, vec=vecA, use_pragma_unroll=True, force_async_copy=(propagate_inter_a and not propagate_inter_b))
        cooperative_fetch(BS, dims=4, vec=vecB, use_pragma_unroll=True, force_async_copy=(propagate_inter_b and not propagate_inter_a))
        
        write_sch(sch, "schedule_shared")
        # ------------------------ Warp memory layout for multiplicand A and B ------------------------
        AW = sch.cache_read(C, 0, "warp")
        BW = sch.cache_read(C, 1, "warp")
        sch.compute_at(AW, ki, preserve_unit_loops=True)
        sch.compute_at(BW, ki, preserve_unit_loops=True)
        
        a_index_map = shared_16x16_to_local_64x4_layout_A
        b_index_map = shared_16x16_to_local_64x4_layout_A if transpose_B else shared_16x16_to_local_64x4_layout_B
        c_index_map = shared_16x16_to_local_64x4_layout_C

        def index_map_A(i, k, wmma_m, wmma_k):
            return (i, k, *a_index_map(wmma_m, wmma_k))


        def index_map_B(*args):
            kernel_i, kernel_j = args[-2], args[-1]  
            other_args = args[:-2]    
            result = (*other_args, *b_index_map(kernel_i, kernel_j))
            return result


        def index_map_C(m, n, wmma_m, wmma_n):
            return (m, n, *c_index_map(wmma_m, wmma_n),)


        sch.transform_layout(AW, ("write", 0), index_map_A)
        sch.transform_layout(BW, ("write", 0), index_map_A if not transpose_B else index_map_B)
        sch.transform_layout(C_warp, ("read", 0), index_map_C)
        
        # ------------------------ Tensorize and Pipelining -------------------------
        init_intrin =  HIP_MFMA_fill_16x16_f32_INTRIN
        load_a_intrin = HIP_MFMA_LOAD_16x16_A_SHARED_f16_INTRIN
        load_b_intrin = HIP_MFMA_LOAD_16x16_B_SHARED_f16_INTRIN
        load_b_intrin_trans = HIP_MFMA_LOAD_16x16_B_TRANS_SHARED_f16_INTRIN
        compute_intrin = HIP_MFMA_f16f16f32_INTRIN
        compute_trans_intrin = HIP_MFMA_f16f16f32_TRANS_INTRIN
        store_intrin = HIP_MFMA_STORE_SHARED_16x16_f32_INTRIN if has_output_op else HIP_MFMA_STORE_GLOBAL_16x16_f32_INTRIN
        init_block_b = sch.decompose_reduction(C, ko)
        write_sch(sch, "decompose_reduction")
        init_block_b_loops = sch.get_loops(init_block_b)
        init_block_b_i, init_block_b_j = sch.get_loops(init_block_b)[-4:-2]
        sch.annotate(init_block_b_i, "pragma_unroll_explicit", False)
        sch.annotate(init_block_b_j, "pragma_unroll_explicit", False)
        sch.tensorize(sch.get_loops(init_block_b)[-2], init_intrin)
        sch.tensorize(
            sch.get_loops(AW)[-2], load_a_intrin
        )
        sch.tensorize(
            sch.get_loops(
                BW)[-2], load_b_intrin if not transpose_B else load_b_intrin_trans
        )
        sch.tensorize(
            kernel_i, compute_intrin if not transpose_B else compute_trans_intrin)
        sch.tensorize(
            sch.get_loops(C_warp)[-2],
            store_intrin,
        )
        if raster > 0:
            sch.annotate(init_block_b_loops[-4], ann_key="thread_rasterization", ann_val=raster)
        if stage > 1:
            if stage > 1:
                sch.annotate(ko, ann_key="software_pipeline_stage", ann_val=[0, 0, stage - 1])
                sch.annotate(ko, ann_key="software_pipeline_order", ann_val=[0, 1, 2])

        write_sch(sch, "cache_small_tensor")

        return sch.mod["main"]
    
    def schedule_consistent_rocm_rdna(self):
        from .ladder_intrin import (
            HIP_RDNA_WMMA_fill_16x16_f16_INTRIN,
            HIP_RDNA_WMMA_LOAD_16x16_A_SHARED_f16_INTRIN,
            HIP_RDNA_WMMA_LOAD_16x16_B_SHARED_f16_INTRIN,
            HIP_RDNA_WMMA_LOAD_16x16_B_TRANS_SHARED_f16_INTRIN,
            HIP_RDNA_WMMA_f16f16f16_INTRIN,
            HIP_RDNA_WMMA_f16f16f16_TRANS_INTRIN,
            HIP_RDNA_WMMA_STORE_SHARED_16x16_f16_INTRIN,
        )
        # const val for testing
        warp_size = self.config.arch.warp_size
        compute_dtype = self.reduce_op.output(0).dtype
        sch, config = self.sche, self.config
        write_sch(sch, "original")
        C = sch.get_block(self.reduce_op.name)
        try:
            i, j, kernel_i, kernel_j, k, kernel_k = sch.get_loops(C)
        except ValueError:
            b, i, j, kernel_i, kernel_j, k, kernel_k = sch.get_loops(C)
            sch.bind(b, "blockIdx.z")
        A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n = config.tc_extra_conf.tc_axis
        propagate_inter_a= config.ladder_config.propagate_inter_a
        propagate_inter_b = config.ladder_config.propagate_inter_b 
        assert propagate_inter_a and propagate_inter_b, "currently only support propagate inter a and b"
        transpose_A = A_ax_k < A_ax_m
        transpose_B = B_ax_k > B_ax_n
        M, K, wmma_m, wmma_k = self.args[0].shape if not transpose_A else self.args[0].shape[::-1]
        N, _, wmma_n, _ = self.args[1].shape if transpose_B else self.args[1].shape[::-1]
        block_tile_M, block_tile_N = self.config.block[0], self.config.block[1]
        warp_tile_M, warp_tile_N = self.config.warp[0], self.config.warp[1]
        out_dtype = self.reduce_op.output(0).dtype
        def get_vec(in_dtype):
            if in_dtype == "float32" or in_dtype == "int32":
                vec = 4
            elif in_dtype == "float16":
                vec = 8
            elif in_dtype == "int8":
                vec = 16
            else:
                raise NotImplementedError("dtype {} not supported".format(in_dtype))
            return vec
        vecA = get_vec(self.args[0].dtype)
        vecB = get_vec(self.args[1].dtype)
        vecC = get_vec(self.args[-1].dtype)
        raster = self.config.raster_factor
        # ------------------------ Block and Warp level job partition ------------------------
        chunk_size = config.rstep[0] // wmma_k

        warp_row_tiles = warp_tile_M
        warp_col_tiles = warp_tile_N
        block_row_warps = block_tile_M // warp_tile_M
        block_col_warps = block_tile_N // warp_tile_N
        chunk = chunk_size
        stage = config.pipeline_stage
       
        block_i, i, ii = sch.split(i, factors=[None, block_row_warps, warp_row_tiles])
        block_j, j, jj = sch.split(j, factors=[None, block_col_warps, warp_col_tiles])
        ko, ki = sch.split(k, factors=[None, chunk])
        sch.reorder(block_i, block_j, i, j, ko, ki, ii, jj, kernel_i, kernel_j, kernel_k)

        if self.sche.get_sref(ko).stmt.extent <= 128:
            self.sche.unroll(ko)
            sch.annotate(ko, "pragma_unroll_explicit", False) 

        write_sch(sch, "BlockTile")

        sch.bind(block_i, "blockIdx.y")
        sch.bind(block_j, "blockIdx.x")
        sch.bind(i, "threadIdx.y")
        sch.bind(j, "threadIdx.z")
        self.block_size = (self.config.arch.warp_size, sch.get_sref(i).stmt.extent, sch.get_sref(j).stmt.extent)
        self.grid_size = (sch.get_sref(block_j).stmt.extent, sch.get_sref(block_i).stmt.extent, 1)
        write_sch(sch, "thread_bind")

        # ------------------------ Shared memory layout for multiplicand A and B ------------------------

        AS = sch.cache_read(C, 0, "shared")
        BS = sch.cache_read(C, 1, "shared")
        sch.compute_at(AS, ko, preserve_unit_loops=True)
        sch.compute_at(BS, ko, preserve_unit_loops=True)
        write_sch(sch, "cached_shared")
        
        # ------------------------ Schedule output fragment layout ------------------------
        C_shared = sch.cache_write(C, 0, "shared")
        C_warp = sch.cache_write(C, 0, "warp", alloc_shape=[int(M), int(N), int(wmma_m * 2), int(wmma_n)])
        sch.reverse_compute_at(C_warp, j, preserve_unit_loops=True)
        sch.reverse_compute_at(
            C_shared,
            sch.get_loops(C_warp)[-3],
            preserve_unit_loops=True,
        )
        
        def schedule_shared_output(block):
            o_shared_fused = sch.fuse(*sch.get_loops(block)[-4:])
            oo, o_shared_tx, o_shared_vi = sch.split(
                o_shared_fused, factors=[None, warp_size, vecC]
            )     
            sch.vectorize(o_shared_vi)
            sch.bind(o_shared_tx, "threadIdx.x")
            sch.unroll(oo)
            sch.annotate(oo, "pragma_unroll_explicit", False)

        schedule_shared_output(C_shared)
        
        write_sch(sch, "schedule_warp")
        
        self.schedule_compute_inline()

        write_sch(sch, "schedule_compute_inline")
        
        def cooperative_fetch(block, dims=4, vec=1, use_pragma_unroll=False, force_async_copy=False):
            loops = sch.get_loops(block)[-dims:]
            other_loop = loops[:-1]
            shared_j = loops[-1]
            shared_j, shared_vi = sch.split(shared_j, factors=[None, vec])
            sch.vectorize(shared_vi)
            shared_fused = sch.fuse(*other_loop, shared_j)
            shared_inner, shared_ty, shared_tz, shared_tx = sch.split(
                shared_fused, factors=[None, block_row_warps, block_col_warps, warp_size])
            sch.bind(shared_tx, "threadIdx.x")
            sch.bind(shared_ty, "threadIdx.y")
            sch.bind(shared_tz, "threadIdx.z")
            self.sche.unroll(shared_inner)
            if use_pragma_unroll:
                self.sche.annotate(shared_inner, "pragma_unroll_explicit", False)

        cooperative_fetch(AS, dims=4, vec=vecA, use_pragma_unroll=True, force_async_copy=(propagate_inter_a and not propagate_inter_b))
        cooperative_fetch(BS, dims=4, vec=vecB, use_pragma_unroll=True, force_async_copy=(propagate_inter_b and not propagate_inter_a))
        
        write_sch(sch, "schedule_shared")
        # ------------------------ Warp memory layout for multiplicand A and B ------------------------
        AW = sch.cache_read(C, 0, "warp", alloc_shape=[int(M), int(K), int(wmma_m * 2), int(wmma_k)])
        BW = sch.cache_read(C, 1, "warp", alloc_shape=[int(N), int(K), int(wmma_n * 2), int(wmma_k)])
        sch.compute_at(AW, ki, preserve_unit_loops=True)
        sch.compute_at(BW, ki, preserve_unit_loops=True)
    
        
        # ------------------------ Tensorize and Pipelining -------------------------
        init_intrin =  HIP_RDNA_WMMA_fill_16x16_f16_INTRIN
        load_a_intrin = HIP_RDNA_WMMA_LOAD_16x16_A_SHARED_f16_INTRIN
        load_b_intrin = HIP_RDNA_WMMA_LOAD_16x16_B_SHARED_f16_INTRIN
        load_b_intrin_trans = HIP_RDNA_WMMA_LOAD_16x16_B_TRANS_SHARED_f16_INTRIN
        compute_intrin = HIP_RDNA_WMMA_f16f16f16_INTRIN
        compute_trans_intrin = HIP_RDNA_WMMA_f16f16f16_TRANS_INTRIN
        store_intrin = HIP_RDNA_WMMA_STORE_SHARED_16x16_f16_INTRIN
        init_block_b = sch.decompose_reduction(C, ko)
        write_sch(sch, "decompose_reduction")
        init_block_b_loops = sch.get_loops(init_block_b)
        init_block_b_i, init_block_b_j = sch.get_loops(init_block_b)[-4:-2]
        sch.annotate(init_block_b_i, "pragma_unroll_explicit", False)
        sch.annotate(init_block_b_j, "pragma_unroll_explicit", False)
        sch.tensorize(sch.get_loops(init_block_b)[-2], init_intrin)
        sch.tensorize(
            sch.get_loops(AW)[-2], load_a_intrin
        )
        sch.tensorize(
            sch.get_loops(
                BW)[-2], load_b_intrin if not transpose_B else load_b_intrin_trans
        )
        sch.tensorize(
            kernel_i, compute_intrin if not transpose_B else compute_trans_intrin)
        sch.tensorize(
            sch.get_loops(C_warp)[-2],
            store_intrin,
        )
        if raster > 0:
            sch.annotate(init_block_b_loops[-4], ann_key="thread_rasterization", ann_val=raster)
        if stage > 1:
            if stage > 1:
                sch.annotate(ko, ann_key="software_pipeline_stage", ann_val=[0, 0, stage - 1])
                sch.annotate(ko, ann_key="software_pipeline_order", ann_val=[0, 1, 2])

        write_sch(sch, "cache_small_tensor")

        return sch.mod["main"]
    
    
    def schedule(self) -> tir.Schedule:
        wmma_k = 32 if self.reduce_op.output(0).dtype == "int32" else 16
        if self.config.consistent_config:
            is_a_consistent = self.config.consistent_config.is_a_consistent
            is_b_consistent = self.config.consistent_config.is_b_consistent
        else:
            is_a_consistent = self.args[0].shape[-1] == wmma_k
            is_b_consistent = self.args[1].shape[-1] == wmma_k
        is_consistent = is_a_consistent and is_b_consistent
        if self.config.arch.platform == "CUDA":
            if is_consistent:
                return self.schedule_consistent()
            else:
                if self.config.pipeline_stage == 2:
                    return self.schedule_inconsistent_shared_decode(is_a_consistent, is_b_consistent)
                return self.schedule_inconsistent(is_a_consistent, is_b_consistent)
        elif "ROCm" in self.config.arch.platform:
            if "CDNA" in self.config.arch.platform:
                return self.schedule_consistent_rocm_cdna()
            elif "RDNA" in self.config.arch.platform:
                return self.schedule_consistent_rocm_rdna()
