import numpy as np
import tvm
from tvm import te

from ..config import Config, Stride
from .te_base import TESchedulerBase

# for debugging.
import os
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

class TEElementWiseScheduler(TESchedulerBase):
    def schedule(self) -> te.Schedule:
        sch, config = self.sche, self.config
        for op in self.ops:
            if op is not self.output_op:
                sch[op].compute_inline()
        out = self.output_op
        # block = [8, 4, 16, 16]
        # thread = [4, 4, 1, 8]
        # step = [1, 1, 1, 2]
        thread = config.thread
        block = config.block
        step = config.step

        self.block_size[0] = int(np.prod(thread))
        write_code(
            str(tvm.lower(sch, self.args, simple_mode=True)), log_path, 'origin.py')
        blck_axis = []
        vthd_axis = []
        thrd_axis = []
        tile_axis = []
        for i, axis in enumerate(sch[out].op.axis):
            bx, _t = sch[out].split(axis, factor=block[i])
            vx, _t = sch[out].split(_t, factor=thread[i] * step[i])
            tx, tn = sch[out].split(_t, factor=step[i])
            blck_axis.append(bx)
            vthd_axis.append(vx)
            thrd_axis.append(tx)
            tile_axis.append(tn)
        write_code(
            str(tvm.lower(sch, self.args, simple_mode=True)), log_path, 'split.py')
        vthd_axis = list(reversed(vthd_axis)) # inner virtual thread first
        axis_order = blck_axis + vthd_axis + thrd_axis + tile_axis
        sch[out].reorder(*axis_order)
        write_code(
            str(tvm.lower(sch, self.args, simple_mode=True)), log_path, 'reorder.py')
        blck_fused = sch[out].fuse(*blck_axis)
        write_code(
            str(tvm.lower(sch, self.args, simple_mode=True)), log_path, 'block_fused.py')
        thrd_fused = sch[out].fuse(*thrd_axis)
        write_code(
            str(tvm.lower(sch, self.args, simple_mode=True)), log_path, 'thrd_fused.py')
        sch[out].bind(blck_fused, te.thread_axis("blockIdx.x"))
        for va in vthd_axis:
            sch[out].bind(va, te.thread_axis("vthread"))
        sch[out].bind(thrd_fused, te.thread_axis("threadIdx.x"))

        for tn in tile_axis:
            sch[out].unroll(tn)
        write_code(
            str(tvm.lower(sch, self.args, simple_mode=True)), log_path, 'unroll.py')
        cache_plan = {}
        for op in self.none_reduce_ops:
            for tensor in op.input_tensors:
                if self.requires_cache(tensor, op):
                    if tensor not in cache_plan:
                        cache_plan[tensor] = []
                    cache_plan[tensor].append(op)

        for tensor, consumers in cache_plan.items():
            tensor_shared = sch.cache_read(tensor, "shared", consumers)
            sch[tensor_shared].compute_at(sch[out], thrd_fused)

            if tensor in self.shared_inputs_strides:
                strides = self.shared_inputs_strides[tensor]
            else:
                strides = Stride()

            self.cooperative_fetch(tensor_shared, strides)

            if len(self.shared_outputs) == 0: continue
            tensor_local = sch.cache_read(tensor_shared, "local", consumers)
            sch[tensor_local].compute_at(sch[out], thrd_fused)
        
        return sch
