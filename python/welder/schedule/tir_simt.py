import numpy as np
from tvm import tir
import os
from ..config import Config, Stride
from .tir_base import TIRSchedulerBase

# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/" + fname
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


class TIRSIMTScheduler(TIRSchedulerBase):
    
    def schedule_consistent(self) -> tir.Schedule:
        sch, config = self.sche, self.config
        self.block_size[0] = int(np.prod(config.thread))

        C = sch.get_block(self.reduce_op.name)
        CL = sch.cache_write(C, 0, "local")
        space_loops = sch.get_loops(C)[:len(self.reduce_op.axis)]
        reduce_loops = sch.get_loops(C)[-len(self.reduce_op.reduce_axis):]

        blck_axis = []
        vthd_axis = []
        thrd_axis = []
        tile_axis = []
        for i, loop in enumerate(space_loops):
            if self.reduce_op.axis[i].dom.extent % config.block[i]:
                raise NotImplementedError("Undivisible block in TIR schedule is still buggy.")
            bx, _t = sch.split(loop, factors=[None, config.block[i]])
            blck_axis.append(bx)
            if config.step[i] > 1:
                _t, tn = sch.split(_t, factors=[None, config.step[i]])
                tile_axis.append(tn)
            if config.block[i] <= config.thread[i] * config.step[i]:
                tx = _t
            else:
                vx, tx = sch.split(_t, factors=[None, config.thread[i]])
                vthd_axis.append(vx)
            thrd_axis.append(tx)

        reduce_outer_axis, reduce_inner_axis = [], []
        for i in config.raxis_order:
            loop = reduce_loops[i]
            ro, ri = sch.split(loop, factors=[None, config.rstep[i]])
            reduce_outer_axis.append(ro)
            reduce_inner_axis.append(ri)

        vthd_axis = list(reversed(vthd_axis)) # inner virtual thread first
        axis_order = blck_axis + vthd_axis + thrd_axis + reduce_outer_axis + reduce_inner_axis + tile_axis

        sch.reorder(*axis_order)
        blck_fused = sch.fuse(*blck_axis)
        thrd_fused = sch.fuse(*thrd_axis)
        sch.bind(blck_fused, "blockIdx.x")
        sch.bind(thrd_fused, "threadIdx.x")
        if len(vthd_axis) > 3:
            vthd_axis = vthd_axis[0:2] + [sch.fuse(*vthd_axis[2:])]
        for i, ax in enumerate(vthd_axis):
            sch.bind(ax, "vthread" + ['.x', '.y', '.z'][i])
        for ax in tile_axis:
            sch.unroll(ax)

        cached_stages = []
        for i, input_tensor in enumerate(self.reduce_op.input_tensors):
            SS = sch.cache_read(C, i, "shared")
            cached_stages.append(SS)
            if input_tensor in self.shared_inputs:
                sch.compute_at(SS, blck_fused)
                strides = self.shared_inputs_strides[input_tensor]
                dim_offset = 1
            else:
                sch.compute_at(SS, reduce_outer_axis[-1])
                strides = Stride()
                dim_offset = len(vthd_axis) + len(reduce_outer_axis) + 2 # outer loops are: blck_fused, thrd_fused, vthd_axis, reduce_outer_axis
            if input_tensor.name in config.vectorize and not self._is_from_shared(input_tensor):
                vectorize = config.vectorize[input_tensor.name]
            else:
                vectorize = 1
            self.cooperative_fetch(SS, dim_offset, strides, vectorize)

        sch.reverse_compute_at(CL, thrd_fused)
        if len(tile_axis) > 0:
            for ax in sch.get_loops(CL)[-len(tile_axis):]:
                sch.unroll(ax)
        
        sch.decompose_reduction(C, reduce_outer_axis[0])

        self.schedule_compute_inline()
        

        # ----- cache small tensors -----
        cache_plan = self.make_cache_plan()
        consumer_ops = {t.op for t in self.reduce_op.input_tensors}
        consumer_ops.add(self.output_op)
        op_input_map = self.detect_op_inputs(consumer_ops)
        for tensor in cache_plan:
            block = None
            if tensor.op in op_input_map[self.output_op]:
                block = CL
            else:
                for i, t in enumerate(self.reduce_op.input_tensors):
                    if tensor.op in op_input_map[t.op]:
                        block = cached_stages[i]
                        break
            assert block
            tensor_shared = sch.cache_read(block, tensor.name, "shared")
            if len(self.shared_outputs) > 0:
                tensor_local = sch.cache_read(block, tensor.name + "_shared", "local")
                sch.compute_at(tensor_local, thrd_fused)
                if len(tile_axis) > 0:
                    for ax in sch.get_loops(tensor_local)[-len(tile_axis):]:
                        sch.unroll(ax)
            sch.compute_at(tensor_shared, thrd_fused)
            if tensor in self.shared_inputs_strides:
                strides = self.shared_inputs_strides[tensor]
            else:
                strides = Stride()
            dim_offset = len(vthd_axis) + 2 # outer loops are: blck_fused vthd_axis thrd_fused
            self.cooperative_fetch(tensor_shared, dim_offset, strides)
        write_sch(sch, log_path, "cache_small_tensor")

        return sch.mod["main"]

    def schedule_inconsistent(self, is_a_consistent: bool, is_b_consistent: bool, use_dp4a=False) -> tir.Schedule:
        sch, config = self.sche, self.config
        assert config.block[0] == 1, "inconsistent computation only support gemv case"
        tx = np.prod(config.thread) * np.prod(config.reduce_thread)
        try:
            vec = list(config.vectorize.values())[-1]
        except IndexError:
            vec = 8
        num_warps = config.block[-1] // config.thread[-1]
        warp_size = config.thread[-1] * config.reduce_thread[-1]
        write_sch(sch, log_path, "origin")
        # num_warps = 1
        # warp_size = 32
        # print(f"tx: {tx}, vec: {vec}, num_warps: {num_warps}, warp_size: {warp_size}")
        block_b = sch.get_block(self.reduce_op.name)
        
        i, j, k = sch.get_loops(block_b)   
        block_decode_A = sch.cache_read(block_b, 0, "local")
        block_decode_B = sch.cache_read(block_b, 1, "local")
        # compute inline
        for op in reversed(self.ops):
            if op not in (self.reduce_op, *[arg.op for arg in self.output_args]):
                block = self.sche.get_block(op.name)
                self.sche.compute_inline(block)
        write_sch(sch, log_path, "compute inline") 
        block_shared_local_A = sch.cache_read(block_decode_A, 0, "local")
        block_shared_local_B = sch.cache_read(block_decode_B, 0, "local")
        
        block_local_C = sch.cache_write(block_b, 0, "local")
        write_sch(sch, log_path, "cache_related")
        # reverse inline
        if self.reduce_op != None and self.reduce_op != self.output_op:
            block = self.sche.get_block(self.output_op.name)
            self.sche.reverse_compute_inline(block)
        bx, j = sch.split(j, factors=[None, num_warps])
        k, tx, vk = sch.split(k, factors=[None, warp_size, vec])
        sch.reorder(bx, j, i, k, tx)

        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")
        sch.bind(j, "threadIdx.y")
        
        self.block_size = [sch.get_sref(tx).stmt.extent, sch.get_sref(j).stmt.extent, 1]
        self.grid_size = [sch.get_sref(bx).stmt.extent, 1, 1]
        
        write_sch(sch, log_path, "do_split")

        sch.compute_at(block_decode_A, tx, preserve_unit_loops=True)
        sch.compute_at(block_decode_B, tx, preserve_unit_loops=True)
        
        sch.compute_at(block_shared_local_A, tx, preserve_unit_loops=True)
        sch.compute_at(block_shared_local_B, tx, preserve_unit_loops=True)
        sch.reverse_compute_at(block_local_C, j, preserve_unit_loops=True)
        write_sch(sch, log_path, "compute_at_related")

        block_local_a_v = sch.get_loops(block_shared_local_A)[-1]
        sch.vectorize(block_local_a_v)
        block_local_b_v = sch.get_loops(block_shared_local_B)[-1]
        sch.vectorize(block_local_b_v)
        if use_dp4a:
            vo, vi = sch.split(vk, [None, 4])
        else:
            vo, vi = sch.split(vk, [None, 2])
        write_sch(sch, log_path, "decompose_reduction")
        
        return sch.mod["main"]
    
    def schedule_inconsistent_shared_decode(self, is_a_consistent: bool, is_b_consistent: bool, use_dp4a=False) -> tir.Schedule:
        from welder.schedule.lop3_intrin import (
            LOP3_FAST_DECODE_INT4_TO_FP16_INTRIN,
            LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN,
            LOP3_FAST_DECODE_INT2_TO_INT8_INTRIN_L8,
            LOP3_FAST_DECODE_INT2_TO_INT8_INTRIN_L16,
            LOP3_FAST_DECODE_INT1_TO_INT8_INTRIN_L16,
        )
        sch, config = self.sche, self.config
        assert config.block[0] == 1, "inconsistent computation only support gemv case"
        tx = np.prod(config.thread) * np.prod(config.reduce_thread)
        try:
            vec = list(config.vectorize.values())[-1]
        except IndexError:
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
            vec = get_vec(self.args[0].dtype)

        num_warps = config.block[-1] // config.thread[-1]
        warp_size = config.thread[-1] * config.reduce_thread[-1]
        write_sch(sch, log_path, "origin")
        
        block_b = sch.get_block(self.reduce_op.name)
        A_decode_block = None
        B_decode_block = None
        other_blocks = []
        
        for op in reversed(self.ops):
            if op not in (self.reduce_op, *[arg.op for arg in self.output_args]):
                if not is_b_consistent and is_a_consistent:
                    if 'decode' in op.name or 'decompress' in op.name:
                        B_decode_block = self.sche.get_block(op.name)
                    else:
                        other_blocks.append(self.sche.get_block(op.name))
                else:
                    if op.name == 'A_decode':
                        A_decode_block = self.sche.get_block(op.name)
                    elif op.name == 'B_decode':
                        B_decode_block = self.sche.get_block(op.name)
                    elif op.name == 'mediate0' and is_a_consistent and not is_b_consistent:
                        B_decode_block = self.sche.get_block(op.name)
                    else:
                        block = self.sche.get_block(op.name)
                        other_blocks.append(block)

        i, j, k = sch.get_loops(block_b)    
        block_shared_local_A = sch.cache_read(block_b, 0, "local")
        block_shared_local_B_rescale = sch.cache_read(block_b, 1, "local")
        for block in other_blocks:
            self.sche.compute_inline(block)
        block_shared_local_B_decompress= sch.cache_read(block_shared_local_B_rescale, 0, "local")
        if not is_a_consistent:
            sch.compute_inline(A_decode_block)
        if not is_b_consistent:
            assert B_decode_block
            read_shape = sch.get_sref(B_decode_block).stmt.reads[0].buffer.shape
            write_shape = sch.get_sref(B_decode_block).stmt.writes[0].buffer.shape
            compress_rate = np.prod(write_shape) // np.prod(read_shape) 
            if self.args[0].dtype == 'float16':
                bits = 16 // compress_rate
            elif self.args[0].dtype == 'int8':
                bits = 8 // compress_rate
            sch.compute_inline(B_decode_block)        
        block_shared_local_B_prefetch = sch.cache_read(block_shared_local_B_decompress, 0, "local")
        block_local_C = sch.cache_write(block_b, 0, "local")
        write_sch(sch, log_path, "cache_related")
        # reverse inline
        if self.reduce_op != None and self.reduce_op != self.output_op:
            block = self.sche.get_block(self.output_op.name)
            self.sche.reverse_compute_inline(block)
        bx, j = sch.split(j, factors=[None, num_warps])
        k, tx, vk = sch.split(k, factors=[None, warp_size, vec])
        sch.reorder(bx, j, i, k, tx)

        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")
        sch.bind(j, "threadIdx.y")
        
        self.block_size = [sch.get_sref(tx).stmt.extent, sch.get_sref(j).stmt.extent, 1]
        self.grid_size = [sch.get_sref(bx).stmt.extent, 1, 1]
        
        write_sch(sch, log_path, "do_split")

        sch.compute_at(block_shared_local_A, tx, preserve_unit_loops=True)
        sch.compute_at(block_shared_local_B_rescale, tx, preserve_unit_loops=True)
        sch.compute_at(block_shared_local_B_decompress, tx, preserve_unit_loops=True)
        sch.compute_at(block_shared_local_B_prefetch, tx, preserve_unit_loops=True)
        sch.reverse_compute_at(block_local_C, j, preserve_unit_loops=True)
        write_sch(sch, log_path, "compute_at_related")

        block_local_a_v = sch.get_loops(block_shared_local_A)[-1]
        sch.vectorize(block_local_a_v)

        block_local_b_v = sch.get_loops(block_shared_local_B_prefetch)[-1]
        sch.vectorize(block_local_b_v)
        if use_dp4a:
            vo, vi = sch.split(vk, [None, 4])
        write_sch(sch, log_path, "decompose_reduction")
        if B_decode_block:
            try:
                if self.args[0].dtype == 'float16':
                    sch.tensorize(sch.get_loops(block_shared_local_B_decompress)[-1], LOP3_FAST_DECODE_INT4_TO_FP16_INTRIN)
                elif self.args[0].dtype == 'int8':
                    # compute the decode bits.
                    if bits == 4:
                        pass
                        # sch.tensorize(sch.get_loops(block_shared_local_B_decompress)[-1], LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN)
                    elif bits == 2:
                        loop = sch.get_loops(block_shared_local_B_decompress)[-1]
                        loop_extent = sch.get_sref(loop).stmt.extent
                        if loop_extent == 16:
                            sch.tensorize(loop, LOP3_FAST_DECODE_INT2_TO_INT8_INTRIN_L16)
                    elif bits == 1:
                        sch.tensorize(sch.get_loops(block_shared_local_B_decompress)[-1], LOP3_FAST_DECODE_INT1_TO_INT8_INTRIN_L16)
            except Exception as e:
                print(e)
        
        write_sch(sch, log_path, "tensorize_lop3")
        return sch.mod["main"]
        
    def schedule(self) -> tir.Schedule:
        if len(self.reduce_op.input_tensors) > 1:
            input0_dtype = self.args[0].dtype
            input1_dtype = self.args[1].dtype
            reduce_op = self.reduce_op
            if self.config.consistent_config:
                is_a_consistent = self.config.consistent_config.is_a_consistent
                is_b_consistent = self.config.consistent_config.is_b_consistent
            else:
                reduce_input0_dtype = reduce_op.input_tensors[0].dtype
                reduce_input1_dtype = reduce_op.input_tensors[1].dtype
                is_a_consistent = reduce_input0_dtype == input0_dtype
                is_b_consistent = reduce_input1_dtype == input1_dtype
            is_consitent = is_a_consistent and is_b_consistent
            use_dp4a = input0_dtype == 'int8' and self.reduce_op.output(0).dtype == "int32"
            if use_dp4a:
                if self.config.compute_capability == "80":
                    return self.schedule_inconsistent_shared_decode(is_a_consistent, is_b_consistent, use_dp4a=True)
                return self.schedule_inconsistent(is_a_consistent, is_b_consistent, use_dp4a=True)
            if is_consitent:
                return self.schedule_consistent()
            else:
                print(
                    f"the computation is inconsistent, is_a_consistent: {is_a_consistent}, is_b_consistent: {is_b_consistent}")
                print(f"self.config.compute_capability {self.config.compute_capability}")
                if self.config.compute_capability == "80":
                    return self.schedule_inconsistent_shared_decode(is_a_consistent, is_b_consistent)
                return self.schedule_inconsistent(is_a_consistent, is_b_consistent)
        else:
            return self.schedule_consistent()
