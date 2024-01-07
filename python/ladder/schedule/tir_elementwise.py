import numpy as np
from tvm import tir
import tvm
import os
from ..config import Config, Stride
from .tir_base import TIRSchedulerBase
from tvm import te
from .utils import write_sch


class TIRElementWiseScheduler(TIRSchedulerBase):
    def create_schedule(self) -> tir.Schedule:
        workload = te.create_prim_func(self.args)
        ir_module = tvm.IRModule({"main": workload})
        return tir.Schedule(ir_module)

    def schedule(self) -> tir.Schedule:
        
        sch, config = self.sche, self.config
        self.block_size[0] = int(np.prod(config.thread))
        write_sch(sch, "origin")

        output_block_name = self.output_op.name
        
        C = sch.get_block(output_block_name)
        space_loops = sch.get_loops(C)[:len(self.output_op.axis)]

        blck_axis = []
        vthd_axis = []
        thrd_axis = []
        tile_axis = []
        for i, loop in enumerate(space_loops):
            if self.output_op.axis[i].dom.extent % config.block[i]:
                raise NotImplementedError("Undivisible block in TIR schedule is still buggy.")
            bx, _t = sch.split(loop, factors=[None, config.block[i]])
            vx, _t = sch.split(
                _t, factors=[None, config.thread[i] * config.step[i]])
            tx, tn = sch.split(_t, factors=[None, config.step[i]])
            blck_axis.append(bx)
            vthd_axis.append(vx)
            thrd_axis.append(tx)
            tile_axis.append(tn)
        vthd_axis = list(reversed(vthd_axis))  # inner virtual thread first
        axis_order = blck_axis + vthd_axis + thrd_axis + tile_axis
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

        write_sch(sch, "unroll")

        if len(self.shared_inputs):
            raise NotImplementedError("Shared memory is not implemented yet.")
                
        write_sch(sch, "cache_input")
        
        self.schedule_compute_inline()

        # ----- cache small tensors -----
        # not implemented yet       
        # cache_plan = self.make_cache_plan()
        # consumer_ops = {t.op for t in self.output_op.input_tensors}
        # consumer_ops.add(self.output_op)
        # op_input_map = self.detect_op_inputs(consumer_ops)
        # for tensor in cache_plan:
        #     block = None
        #     for i, t in enumerate(self.output_op.input_tensors):
        #         if tensor.op in op_input_map[t.op]:
        #             block = cached_stages[i]
        #             break
        #     assert block
        #     tensor_shared = sch.cache_read(block, tensor.name, "shared")
        #     if len(self.shared_outputs) > 0:
        #         tensor_local = sch.cache_read(block, tensor.name + "_shared", "local")
        #         sch.compute_at(tensor_local, thrd_fused)
        #         if len(tile_axis) > 0:
        #             for ax in sch.get_loops(tensor_local)[-len(tile_axis):]:
        #                 sch.unroll(ax)
        #     sch.compute_at(tensor_shared, thrd_fused)
        #     if tensor in self.shared_inputs_strides:
        #         strides = self.shared_inputs_strides[tensor]
        #     else:
        #         strides = Stride()
        #     dim_offset = len(vthd_axis) + 2 # outer loops are: blck_fused vthd_axis thrd_fused
        #     self.cooperative_fetch(tensor_shared, dim_offset, strides)
        write_sch(sch, "cache_small_tensor")

        return sch.mod["main"]
