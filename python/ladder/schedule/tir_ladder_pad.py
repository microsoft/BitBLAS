import numpy as np
from tvm import tir
import os
from ..layout import *
from .tir_base import TIRSchedulerBase
from .utils import write_sch

class TIRLadderMMAPadScheduler2D(TIRSchedulerBase):
    def schedule(self) -> tir.Schedule:
        from tvm.tir.tensor_intrin.cuda import (
            WMMA_FILL_16x16x16_F16_INTRIN,
            WMMA_LOAD_16x16x16_F16_A_INTRIN,
            WMMA_LOAD_16x16x16_F16_B_INTRIN,
            WMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN,
            WMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN,
            WMMA_SYNC_16x16x16_f16f16f16_INTRIN,
            WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN,
            WMMA_STORE_16x16x16_F16_SHARED_INTRIN,
        )
        # const val for testing
        warp_size = 32
        wmma_m, wmma_n, wmma_k = 16, 16, 16
        sch, config = self.sche, self.config
        write_sch(sch, "original")
        C = sch.get_block(self.reduce_op.name)
        A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n = config.tc_extra_conf.tc_axis
        transpose_A = A_ax_k < A_ax_m
        transpose_B = B_ax_k > B_ax_n
        block_tile_M, block_tile_N = self.config.block[0], self.config.block[1]
        warp_tile_M, warp_tile_N = self.config.warp[0], self.config.warp[1]
        out_dtype = self.reduce_op.output(0).dtype
        in_dtype = self.reduce_op.input_tensors[0].dtype

        if in_dtype == "float32":
            vec = 4
        elif in_dtype == "float16":
            vec = 8
        elif in_dtype == "int8":
            vec = 16
        else:
            raise NotImplementedError("dtype {} not supported".format(in_dtype))
        raster = self.config.raster_factor
        # ------------------------ Block and Warp level job partition ------------------------
        chunk_size = config.rstep[0] // wmma_k
        block_row_warps = block_tile_M // warp_tile_M
        block_col_warps = block_tile_N // warp_tile_N
        warp_row_tiles = warp_tile_M // wmma_m
        warp_col_tiles = warp_tile_N // wmma_n
        chunk = chunk_size
        stage = config.pipeline_stage
        use_async = stage > 1
        output_shape = self.reduce_op.output(0).shape
        is_matmul = (output_shape[-1] == 16 and output_shape[-2] == 16)
        if len(output_shape) == 2:  
            M = int(output_shape[0])
            N = int(output_shape[1])
        elif len(output_shape) == 2:
            # nhwc or mn1616
            if is_matmul:
                M = output_shape[0] * output_shape[2]
                N = output_shape[1] * output_shape[3]
            else:
                M = output_shape[0] * output_shape[1] * output_shape[2]
                N = output_shape[3]
        elif len(output_shape) == 6 and is_matmul:
            M = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[-2]
            N = output_shape[3] * output_shape[-1]
        K = int(self.args[1].shape[-1]) if transpose_B else int(self.args[1].shape[-2])
        MPAD = (
            (M + block_row_warps * warp_row_tiles * wmma_m - 1)
            // (block_row_warps * warp_row_tiles * wmma_m)
            * block_row_warps
            * warp_row_tiles
            * wmma_m
        )
        # padding NPAD as the multiple of block_col_warps * warp_col_tiles * wmma_n
        NPAD = (
            (N + block_col_warps * warp_col_tiles * wmma_n - 1)
            // (block_col_warps * warp_col_tiles * wmma_n)
            * block_col_warps
            * warp_col_tiles
            * wmma_n
        )
        # padding KPAD as the multiple of block_col_warps * warp_col_tiles * wmma_k
        KPAD = (
            (K + block_col_warps * warp_col_tiles * wmma_k - 1)
            // (block_col_warps * warp_col_tiles * wmma_k)
            * block_col_warps
            * warp_col_tiles
            * wmma_k
        )
        print("MPAD: ", MPAD)
        print("NPAD: ", NPAD)
        print("KPAD: ", KPAD)
        AS = sch.cache_read(C, 0, "shared")
        BS = sch.cache_read(C, 1, "shared")
        C_shared = sch.cache_write(C, 0, "shared")
        
        self.schedule_compute_inline()
        sch.pad_einsum(C, [MPAD - M, NPAD - N, KPAD - K])
        write_sch(sch, "pad_einsum")
        
        (i, j, k) = sch.get_loops(C)
        i, kernel_i = sch.split(i, factors=[None, wmma_m])
        j, kernel_j = sch.split(j, factors=[None, wmma_n])
        k, kernel_k = sch.split(k, factors=[None, wmma_k])
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
        
        sch.compute_at(AS, ko, preserve_unit_loops=True)
        sch.compute_at(BS, ko, preserve_unit_loops=True)
        write_sch(sch, "cached_shared")
        
        # ------------------------ Schedule output fragment layout ------------------------
        
        C_warp = sch.cache_write(C, 0, "wmma.accumulator")
        sch.reverse_compute_at(C_warp, j, preserve_unit_loops=True)
        
        out_i, out_j = sch.get_loops(C_warp)[-2:]
        out_i, ok_i = sch.split(out_i, factors=[None, wmma_m])
        out_j, ok_j = sch.split(out_j, factors=[None, wmma_n])
        sch.reorder(out_i, out_j, ok_i, ok_j)
        sch.unroll(out_j)
        
        sch.reverse_compute_at(
            C_shared,
            sch.get_loops(C_warp)[-3],
            preserve_unit_loops=True,
        )
        def transform_out(i, j):
            return (i // wmma_m, j // wmma_n, i % wmma_m, j % wmma_n)
        sch.transform_layout(C_warp, ("write", 0), transform_out)
        sch.transform_layout(C_warp, ("read", 0), transform_out)
        
        def schedule_shared_output(block):
            o_shared_fused = sch.fuse(*sch.get_loops(block)[-2:])
            _, o_shared_tx, o_shared_vi = sch.split(
                o_shared_fused, factors=[None, warp_size, vec]
            )     
            sch.vectorize(o_shared_vi)
            sch.bind(o_shared_tx, "threadIdx.x")

        schedule_shared_output(C_shared)
        
        write_sch(sch, "schedule_warp")        
        
        def cooperative_fetch(block, dims=4, vec=vec):
            read_dtype = sch.get_sref(block).stmt.reads[-1].buffer.dtype
            out_dtype = sch.get_sref(block).stmt.writes[-1].buffer.dtype
            if read_dtype == "int8" and out_dtype == "float16":
                vec = 4
            shared_fused = sch.fuse(*sch.get_loops(block)[-dims:])
            shared_ty, shared_tz, shared_inner, shared_tx, shared_vi = sch.split(
                shared_fused, factors=[block_tile_M // warp_tile_M, block_tile_N // warp_tile_N, None, warp_size, vec])
            sch.vectorize(shared_vi)
            sch.bind(shared_tx, "threadIdx.x")
            sch.bind(shared_ty, "threadIdx.y")
            sch.bind(shared_tz, "threadIdx.z")
            sch.storage_align(block, 0, axis=-2, factor=32, offset=8)

        cooperative_fetch(AS, dims=2, vec=1)
        cooperative_fetch(BS, dims=2, vec=1 if transpose_B else vec)
        
        write_sch(sch, "schedule_shared")
        # ------------------------ Warp memory layout for multiplicand A and B ------------------------
        AW = sch.cache_read(C, 0, "wmma.matrix_a")
        BW = sch.cache_read(C, 1, "wmma.matrix_b")
        sch.compute_at(AW, ki, preserve_unit_loops=True)
        sch.compute_at(BW, ki, preserve_unit_loops=True)
        def tricky_extract_cache(block, sub_i, sub_j):
            i, j = sch.get_loops(block)[-2:]
            i, kernel_i = sch.split(i, factors=[None, sub_i])
            j, kernel_j = sch.split(j, factors=[None, sub_j])
            sch.reorder(i, j, kernel_i, kernel_j)
            sch.unroll(i)
            sch.unroll(j)
            return (i, j, kernel_i, kernel_j)


        block_conv_input_frag_loops = tricky_extract_cache(
            AW, wmma_m, wmma_k)
        block_conv_input_frag_loops = tricky_extract_cache(
            BW, wmma_m, wmma_k)


        # ------------------------ Tensorize and Pipelining -------------------------
        init_block_b = sch.decompose_reduction(C, ko)
        write_sch(sch, "decompose_reduction")
        init_block_b_loops = sch.get_loops(init_block_b)
        init_block_b_i, init_block_b_j = sch.get_loops(init_block_b)[-4:-2]
        sch.tensorize(sch.get_loops(init_block_b)[-2], WMMA_FILL_16x16x16_F16_INTRIN)
        sch.tensorize(
            sch.get_loops(AW)[-2], WMMA_LOAD_16x16x16_F16_A_INTRIN
        )
        sch.tensorize(
            sch.get_loops(BW)[-2], WMMA_LOAD_16x16x16_F16_B_INTRIN if not transpose_B else WMMA_LOAD_16x16x16_F16_B_TRANS_INTRIN
        )
        sch.tensorize(kernel_i, WMMA_SYNC_16x16x16_f16f16f16_INTRIN if not transpose_B else WMMA_SYNC_16x16x16_f16f16f16_TRANS_INTRIN)

        sch.tensorize(
            sch.get_loops(C_warp)[-2],
            WMMA_STORE_16x16x16_F16_SHARED_INTRIN,
        )
        write_sch(sch, "tensorize_store")
        if raster > 0:
            sch.annotate(init_block_b_loops[-4], ann_key="thread_rasterization", ann_val=raster)
        if stage > 1:
            sch.annotate(ko, ann_key="software_pipeline_stage",
                         ann_val=[0, 0, stage - 1])
            sch.annotate(ko, ann_key="software_pipeline_order",
                         ann_val=[0, 1, 2])
            if use_async:
                sch.annotate(ko, "software_pipeline_async_stages", [0])

        # ------------------------ Cache small tensors -------------------------------
        # cache_plan = self.make_cache_plan()
        # if len(self.shared_outputs) > 0:
        #     cache_plan.clear() # supports writing to global for now
        # consumer_ops = {t.op for t in self.reduce_op.input_tensors}
        # consumer_ops.add(self.output_op)
        # op_input_map = self.detect_op_inputs(consumer_ops)
        # for tensor in cache_plan:
        #     if tensor.op not in op_input_map[self.output_op]:
        #         continue
        #     tensor_shared = sch.cache_read(C_shared, tensor.name, "local")
        #     sch.compute_at(tensor_shared, sch.get_loops(C_shared)[-1])
        #     self.cooperative_fetch(tensor_shared, dim_offset=2, vector_load=vec)

        write_sch(sch, "cache_small_tensor")

        return sch.mod["main"]
