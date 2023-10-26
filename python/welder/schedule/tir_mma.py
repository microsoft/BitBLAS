import numpy as np
from tvm import tir
import tvm
from ..config import Stride
from ..IRpass import ApplyLayoutPass
from ..layout import *
from .cutlass_intrin import *
from .tir_base import TIRSchedulerBase

import os
# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
log_path = "progress/" + fname
# create log path
count = 0


def write_code(code, path, fname):
    global count
    # if path not exist, create it
    fname = str(count) + "." + fname
    count += 1
    if not os.path.exists(path):
        os.makedirs(path)
    # join path and fname
    fname = os.path.join(path, fname)
    with open(fname, "w") as f:
        f.write(code)

def write_sch(sch, path, fname):
    py_fname = fname + ".py"
    write_code(sch.mod["main"].script(), path, py_fname)
    cu_fname = fname + ".cu"
    write_code(sch.mod.astext(), path, cu_fname)

class TIRCutlassMMAScheduler(TIRSchedulerBase):
    def schedule(self) -> tir.Schedule:
        sch, config = self.sche, self.config
        self.block_size[0] = 32
        self.block_size[1] = int(np.prod(self.config.block)) // int(np.prod(self.config.warp))
        C = sch.get_block(self.reduce_op.name)
        space_loops = sch.get_loops(C)[:len(self.reduce_op.axis)]
        assert(len(self.reduce_op.reduce_axis) == 1)
        ax_K = sch.get_loops(C)[-1]
        A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n = config.tc_extra_conf.tc_axis
        transpose_A = A_ax_k < A_ax_m
        transpose_B = B_ax_k > B_ax_n
        block_tile_M, block_tile_N = self.config.block[C_ax_m], self.config.block[C_ax_n]
        warp_tile_M, warp_tile_N = self.config.warp[C_ax_m], self.config.warp[C_ax_n]
        out_dtype = self.reduce_op.output(0).dtype
        in_dtype = self.reduce_op.input_tensors[0].dtype
        # log_path = f"progress/tir_mma_{block_tile_M}_{block_tile_N}_{warp_tile_M}_{warp_tile_N}"
        is_fpa_intb = self.args[0].dtype == "float16" and self.args[1].dtype == "int8"
        num_args = len(self.args)
        is_lut = False
        if num_args >= 4:
            lut_arg = self.args[2] # assume the 3rd arg is the lut
            lut_shape = np.prod(lut_arg.shape)
            if lut_shape == 16:
                is_lut = True
        # ------------------------ Block and Warp level job partition ------------------------
        def get_vec(in_dtype):
            if in_dtype == "float32":
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
        vecC = get_vec(self.args[2].dtype)

        block_axis = []
        warp_axis = []
        inner_axis = []
        for i, loop in enumerate(space_loops):
            if i in (C_ax_m, C_ax_n):
                bo, wo, wi = sch.split(loop, factors=[None, config.block[i] // config.warp[i], config.warp[i]])
                block_axis.append(bo)
                warp_axis.append(wo)
                inner_axis.append(wi)
            else:
                assert config.block[i] == 1
                block_axis.append(loop)

        chunk_size = config.rstep[0]
        K_outer, K_inner = sch.split(ax_K, factors=[None, chunk_size])

        sch.reorder(*block_axis, *warp_axis, K_outer, *inner_axis, K_inner)
        block_fused = sch.fuse(*block_axis)
        warp_fused = sch.fuse(*warp_axis)
        sch.bind(block_fused, "blockIdx.x")
        sch.bind(warp_fused, "threadIdx.y")

        # ------------------------ Shared memory layout for multiplicand A and B ------------------------
        try:
            if config.use_tc >= "80":
                if transpose_A:
                    layoutA = ColumnMajorTensorOpMultiplicandCongruous(chunk_size, block_tile_M)
                else:
                    layoutA = RowMajorTensorOpMultiplicandCrosswise(block_tile_M, chunk_size)
            elif config.use_tc >= "70":
                if transpose_A:
                    layoutA = ColumnMajorVoltaTensorOpMultiplicandCongruous(chunk_size, block_tile_M)
                else:
                    layoutA = RowMajorVoltaTensorOpMultiplicandCrosswise(block_tile_M, chunk_size)
        except AssertionError:
            if transpose_A:
                layoutA = ColumnMajorLayout(chunk_size, block_tile_M)
            else:
                layoutA = RowMajorLayout(block_tile_M, chunk_size)
        try:
            if config.use_tc >= "80":
                # using ldmatrix 16x16
                assert warp_tile_N % 16 == 0
                if transpose_B:
                    layoutB = ColumnMajorTensorOpMultiplicandCrosswise(block_tile_N, chunk_size)
                else:
                    layoutB = RowMajorTensorOpMultiplicandCongruous(chunk_size, block_tile_N)
            elif config.use_tc >= "70":
                if transpose_B:
                    layoutB = ColumnMajorVoltaTensorOpMultiplicandCrosswise(block_tile_N, chunk_size)
                else:
                    layoutB = RowMajorVoltaTensorOpMultiplicandBCongruous(chunk_size, block_tile_N)
        except AssertionError:
            if transpose_B:
                layoutB = ColumnMajorLayout(block_tile_N, chunk_size)
            else:
                layoutB = RowMajorLayout(chunk_size, block_tile_N)

        AS = sch.cache_read(C, 0, "shared")
        BS = sch.cache_read(C, 1, "shared")
        sch.compute_at(AS, K_outer)
        sch.compute_at(BS, K_outer)
        write_sch(sch, log_path, "compute_at")

        A_stride, B_stride = Stride(), Stride()
        if layoutA.requires_padding():
            A_high_ax = min(A_ax_m, A_ax_k)
            if config.use_tc >= "80" or transpose_A:
                padA = 8
            else:
                padA = 4
            layoutA.set_pad(padA)
            A_stride = Stride(int(np.prod(config.tc_extra_conf.AS_shape[A_high_ax+1:])) + padA, A_high_ax)
        if layoutB.requires_padding():
            B_high_ax = min(B_ax_n, B_ax_k)
            if config.use_tc >= "80" or not transpose_B:
                padB = 8
            else:
                padB = 4
            layoutB.set_pad(padB)
            B_stride = Stride(int(np.prod(config.tc_extra_conf.BS_shape[B_high_ax+1:])) + padB, B_high_ax)

        if vecB == 16 and vecA == 8:
            vecB = 4
        self.cooperative_fetch(AS, 3, A_stride, vector_load=vecA, use_pragma_unroll=True)
        if not is_fpa_intb:
            self.cooperative_fetch(BS, 3, B_stride, vector_load=vecB, use_pragma_unroll=True)


        # ------------------------ Schedule output fragment layout ------------------------
        if config.use_tc >= "80":
            layoutC = FragmentCLayout8x8(warp_tile_M, warp_tile_N)
            cls_code = register_cutlass_warp_mma(warp_tile_M, warp_tile_N, chunk_size, layoutA, layoutB)
        elif config.use_tc >= "70":
            layoutC = voltaFragmentCLayout32x32(warp_tile_M, warp_tile_N)
            cls_code = register_volta_cutlass_warp_mma(warp_tile_M, warp_tile_N, chunk_size, layoutA, layoutB)
        C_warp = sch.cache_write(C, 0, "cutlass.warp.mma")
        sch.reverse_compute_at(C_warp, warp_fused)
        block_init_c = sch.decompose_reduction(C, sch.get_loops(C)[2])
        sch.transform_loop(C_warp, 2, layoutC)
        sch.bind(sch.get_loops(C_warp)[-2], "threadIdx.x")
        oo, vec = sch.split(sch.get_loops(C_warp)[-1], factors=[None, layoutC.get_vectorize()])
        sch.vectorize(vec)
        sch.unroll(oo)
        sch.annotate(oo, "pragma_unroll_explicit", False)
        if not is_fpa_intb:
            self.schedule_compute_inline()
        
        if is_fpa_intb:
            BL0 = sch.cache_read(BS, 0, "local")
            self.schedule_compute_inline()
            sch.compute_at(BL0, K_outer)
            if is_lut:
                BL1 = sch.cache_read(BL0, 1, "local")
            else:
                BL1 = sch.cache_read(BL0, 0, "local")
            BSS = sch.cache_read(BL1, 0, "shared")
            
            sch.compute_at(BL1, K_outer)
            sch.compute_at(BSS, K_outer)
            
            def cooperative_fetch(SS: tir.Block, dim_offset: int, strides: Stride=Stride(), vector_load: int=1, use_pragma_unroll: bool=False):
                loops = self.sche.get_loops(SS)
                assert len(loops) > dim_offset
                axes = loops[-2:]
                if strides.is_valid():
                    self.sche.storage_align(SS, 0, strides.ax, strides.stride - 1, strides.stride)
                ax = axes[-1]
                ax, tv = self.sche.split(ax, factors=[None, vector_load])
                if vector_load > 1:
                    _, tv = self.sche.split(tv, factors=[None, vector_load])
                    self.sche.vectorize(tv)
                ax = self.sche.fuse(axes[-2], ax)
                if self.block_size[0] > 1:
                    ax, tx = self.sche.split(ax, factors=[None, self.block_size[0]])
                    self.sche.bind(tx, "threadIdx.x")
                if self.block_size[1] > 1:
                    ax, ty = self.sche.split(ax, factors=[None, self.block_size[1]])
                    self.sche.bind(ty, "threadIdx.y")
                if self.block_size[2] > 1:
                    ax, tz = self.sche.split(ax, factors=[None, self.block_size[2]])
                    self.sche.bind(tz, "threadIdx.z")
                # self.sche.unroll(ax)
                if use_pragma_unroll:
                    self.sche.annotate(ax, "pragma_unroll_explicit", False)
            cooperative_fetch(BSS, 3, strides=B_stride, vector_load=4, use_pragma_unroll=False)
            cooperative_fetch(BS, 3, strides=B_stride, vector_load=8, use_pragma_unroll=False)
            write_sch(sch, log_path, "cooperative_fetch_BSS")
            
            sch.compute_at(BL0, self.sche.get_loops(BS)[-3])
            sch.compute_at(BL1, self.sche.get_loops(BS)[-3])

            if is_lut:
                block_shared_lut = sch.cache_read(BL0, 0, "shared")
                sch.reverse_compute_at(block_shared_lut, block_fused)
                _, B_shared_tx = sch.split(
                    sch.get_loops(block_shared_lut)[-1], factors=[None, 32])
                sch.bind(B_shared_tx, "threadIdx.x")
            if B_stride.is_valid(): 
                self.sche.storage_align(BS, 0, B_stride.ax, B_stride.stride - 1, B_stride.stride)
            write_sch(sch, log_path, "reverse_compute_at")
            bv = sch.get_loops(BL0)[-1]
            _, bv = sch.split(bv, factors=[None, 4])
            # sch.vectorize(bv)
            sch.compute_at(BL1, self.sche.get_loops(BL0)[-2])
            bv = sch.get_loops(BL1)[-1]
            sch.vectorize(bv)
            bv = sch.get_loops(BS)[-1]
            sch.vectorize(bv)
            write_sch(sch, log_path, "schedule bs")


        # ------------------------ Tensorize and Pipelining -------------------------
        
        sch.tensorize(sch.get_loops(block_init_c)[-2],
            register_cutlass_warp_init_intrin(warp_tile_M, warp_tile_N, out_dtype,
            cls_code, block_tile_M // warp_tile_M, block_tile_N // warp_tile_N)
        )
        sch.tensorize(sch.get_loops(C)[-3],
            register_gemm_intrin(
                config.warp[C_ax_m], config.warp[C_ax_n], chunk_size,
                in_dtype, out_dtype,
                transpose_A, transpose_B,
                layoutA, layoutB)
        )

        if config.use_tc >= "80":
            if chunk_size % 32 != 0:
                sch.annotate(K_outer, "software_pipeline_stage", [0, 0, 1, 1, 1])
                sch.annotate(K_outer, "software_pipeline_order", [0, 1, 2, 3, 4])
            else:
                if is_fpa_intb:
                    sch.annotate(K_outer, "software_pipeline_stage", [0, 0, 1, 1, 1, 1])
                    sch.annotate(K_outer, "software_pipeline_order", [0, 1, 2, 3, 4, 5])
                else:
                    sch.annotate(K_outer, "software_pipeline_stage", [0, 0, 1, 1, 2])
                    sch.annotate(K_outer, "software_pipeline_order", [0, 1, 2, 4, 3])
            sch.annotate(K_outer, "software_pipeline_async_stages", [0])
            self.passes.append((3, tvm.tir.transform.InjectPTXAsyncCopy()))
        elif config.use_tc >= "70":
            if chunk_size % 8 != 0:
                sch.annotate(K_outer, "software_pipeline_stage", [0, 0, 0, 0, 1, 1, 1])
                sch.annotate(K_outer, "software_pipeline_order", [0, 5, 1, 6, 2, 3, 4])
            else:
                sch.annotate(K_outer, "software_pipeline_stage", [0, 0, 0, 0, 1, 1, 2])
                sch.annotate(K_outer, "software_pipeline_order", [0, 5, 1, 6, 2, 4, 3])
            sch.annotate(AS, "tir.manifest_shared_memory_local_stage", 1)
            sch.annotate(BS, "tir.manifest_shared_memory_local_stage", 1)

        layout_pass = ApplyLayoutPass({
            self.reduce_op.input_tensors[0].name+"_shared": layoutA,
            self.reduce_op.input_tensors[1].name+"_shared": layoutB,
            self.reduce_op.name + "_cutlass.warp.mma": layoutC.fragment_offset})
        self.passes.append(layout_pass.get_pass())

        # ------------------------ Cache small tensors -------------------------------
        cache_plan = self.make_cache_plan()
        if len(self.shared_outputs) > 0:
            cache_plan.clear() # supports writing to global for now
        consumer_ops = {t.op for t in self.reduce_op.input_tensors}
        consumer_ops.add(self.output_op)
        op_input_map = self.detect_op_inputs(consumer_ops)
        for tensor in cache_plan:
            if tensor.op not in op_input_map[self.output_op]:
                continue
            tensor_shared = sch.cache_read(C_warp, tensor.name, "shared")
            sch.compute_at(tensor_shared, warp_fused)
            dim_offset = 2 # outer loops are: blck_fused thrd_fused
            self.cooperative_fetch(tensor_shared, dim_offset)
        write_sch(sch, log_path, "cache_small_tensor")
        return sch.mod["main"]
