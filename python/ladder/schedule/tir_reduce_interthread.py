import numpy as np
from tvm import tir
import tvm
import os
from ..config import Config, Stride
from .tir_base import TIRSchedulerBase
from tvm import te
from .utils import write_sch
import logging

logger = logging.getLogger(__name__)


class TIRReduceInterThreadScheduler(TIRSchedulerBase):
    def create_schedule(self) -> tir.Schedule:
        workload = te.create_prim_func(self.args)
        ir_module = tvm.IRModule({"main": workload})
        return tir.Schedule(ir_module)

    def schedule_consistent(self) -> tir.Schedule:
        sch, config = self.sche, self.config
        assert config.block[0] == 1, "tir computation only support gemv case"
        tx = np.prod(config.thread) * np.prod(config.reduce_thread)
        try:
            vec = list(config.vectorize.values())[-1]
        except IndexError:
            vec = 8
        num_warps = int(np.prod(self.config.thread))
        warp_size = int(np.prod(self.config.reduce_thread))
        write_sch(sch, "origin")
        block_b = sch.get_block(self.reduce_op.name)

        # compute inline
        for op in reversed(self.ops):
            if op not in (self.reduce_op, *[arg.op for arg in self.output_args]):
                block = self.sche.get_block(op.name)
                self.sche.compute_inline(block)

        i, j, k = sch.get_loops(block_b)
        block_shared_local_A = sch.cache_read(block_b, 0, "local")
        block_shared_local_B = sch.cache_read(block_b, 1, "local")
        block_local_C = sch.cache_write(block_b, 0, "local")
        write_sch(sch, "cache_related")
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

        write_sch(sch, "do_split")

        sch.compute_at(block_shared_local_A, tx, preserve_unit_loops=True)
        sch.compute_at(block_shared_local_B, tx, preserve_unit_loops=True)
        sch.reverse_compute_at(block_local_C, j, preserve_unit_loops=True)
        write_sch(sch, "compute_at_related")

        block_local_a_v = sch.get_loops(block_shared_local_A)[-1]
        sch.vectorize(block_local_a_v)
        block_local_b_v = sch.get_loops(block_shared_local_B)[-1]
        sch.vectorize(block_local_b_v)
        write_sch(sch, "decompose_reduction")

        return sch.mod["main"]

    def schedule_inconsistent(
        self, is_a_consistent: bool, is_b_consistent: bool, use_dp4a=False
    ) -> tir.Schedule:
        from ladder.schedule.lop3_intrin import (
            LOP3_FAST_DECODE_INT4_TO_FP16_INTRIN,
            LOP3_FAST_DECODE_INT2_TO_INT8_INTRIN_L16,
            LOP3_FAST_DECODE_INT1_TO_INT8_INTRIN_L16
        )

        sch, config = self.sche, self.config
        assert config.block[0] == 1, "inconsistent computation only support gemv case"
        tx = np.prod(config.thread) * np.prod(config.reduce_thread)

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

        if config.ladder_compute_type == "mxfp":
            vecA = get_vec(self.args[0].dtype)
            vecB = get_vec(self.args[1].dtype)
            vec = min(vecA, vecB)
        else:
            try:
                vec = list(config.vectorize.values())[-1]
            except IndexError:
                vec = get_vec(self.args[0].dtype)
        num_warps = int(np.prod(self.config.thread))
        warp_size = int(np.prod(self.config.reduce_thread))
        write_sch(sch, "origin")

        block_b = sch.get_block(self.reduce_op.name)

        i, j, k = sch.get_loops(block_b)

        A_decode_block = None
        B_decode_block = None
        other_blocks = []
        for op in reversed(self.ops):
            if op not in (self.reduce_op, *[arg.op for arg in self.output_args]):
                if op.name == "A_decode":
                    A_decode_block = self.sche.get_block(op.name)
                elif op.name == "B_decode":
                    B_decode_block = self.sche.get_block(op.name)
                elif op.name == "mediate0" and is_a_consistent and not is_b_consistent:
                    B_decode_block = self.sche.get_block(op.name)
                else:
                    block = self.sche.get_block(op.name)
                    other_blocks.append(block)

        if not is_a_consistent:
            block_decode_A = sch.cache_read(block_b, 0, "local")
        block_decode_B = sch.cache_read(block_b, 1, "local")

        write_sch(sch, "cache_read_decode")

        if A_decode_block:
            sch.compute_inline(A_decode_block)
        if B_decode_block:
            read_shape = sch.get_sref(B_decode_block).stmt.reads[0].buffer.shape
            write_shape = sch.get_sref(B_decode_block).stmt.writes[0].buffer.shape
            compress_rate = np.prod(write_shape) // np.prod(read_shape)
            if self.args[0].dtype == "float16":
                bits = 16 // compress_rate
            elif self.args[0].dtype == "int8":
                bits = 8 // compress_rate
            sch.compute_inline(B_decode_block)

        write_sch(sch, "inline_decode")
        if not is_a_consistent:
            block_shared_local_A = sch.cache_read(block_decode_A, 0, "local")
        else:
            block_shared_local_A = sch.cache_read(block_b, 0, "local")

        write_sch(sch, "compute inline")
        block_shared_local_B = sch.cache_read(block_decode_B, 0, "local")
        block_local_C = sch.cache_write(block_b, 0, "local")
        write_sch(sch, "cache_related")

        # compute inline
        for block in other_blocks:
            self.sche.compute_inline(block)
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

        write_sch(sch, "do_split")

        if not is_a_consistent:
            sch.compute_at(block_decode_A, tx, preserve_unit_loops=True)
        sch.compute_at(block_decode_B, tx, preserve_unit_loops=True)

        sch.compute_at(block_shared_local_A, tx, preserve_unit_loops=True)
        sch.compute_at(block_shared_local_B, tx, preserve_unit_loops=True)
        sch.reverse_compute_at(block_local_C, j, preserve_unit_loops=True)
        write_sch(sch, "compute_at_related")

        block_local_a_v = sch.get_loops(block_shared_local_A)[-1]
        sch.vectorize(block_local_a_v)
        block_local_b_v = sch.get_loops(block_shared_local_B)[-1]
        sch.vectorize(block_local_b_v)
        if use_dp4a:
            vo, vi = sch.split(vk, [None, 4])
        if B_decode_block and self.config.fast_decoding:
            try:
                if self.args[0].dtype == "float16":
                    sch.tensorize(
                        sch.get_loops(block_decode_B)[-1],
                        LOP3_FAST_DECODE_INT4_TO_FP16_INTRIN,
                    )
                elif self.args[0].dtype == "int8":
                    # compute the decode bits.
                    if bits == 4:
                        pass
                        # sch.tensorize(sch.get_loops(block_shared_local_B_decompress)[-1], LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN)
                    elif bits == 2:
                        loop = sch.get_loops(block_decode_B)[-1]
                        loop_extent = sch.get_sref(loop).stmt.extent
                        if loop_extent == 16:
                            sch.tensorize(
                                loop, LOP3_FAST_DECODE_INT2_TO_INT8_INTRIN_L16
                            )
                    elif bits == 1:
                        sch.tensorize(
                            sch.get_loops(block_decode_B)[-1],
                            LOP3_FAST_DECODE_INT1_TO_INT8_INTRIN_L16,
                        )
            except Exception as e:
                logger.debug(f"tensorize decode block failed: {e}")
        write_sch(sch, "decompose_reduction")

        return sch.mod["main"]

    def schedule_inconsistent_lut(
        self, is_a_consistent: bool, is_b_consistent: bool
    ) -> tir.Schedule:
        assert is_a_consistent and not is_b_consistent
        sch, config = self.sche, self.config
        assert config.block[0] == 1, "inconsistent computation only support gemv case"
        tx = np.prod(config.thread) * np.prod(config.reduce_thread)
        vec = 8
        num_warps = int(np.prod(self.config.thread))
        warp_size = int(np.prod(self.config.reduce_thread))
        write_sch(sch, "origin")
        # num_warps = 1
        # warp_size = 32
        block_b = sch.get_block(self.reduce_op.name)
        i, j, k = sch.get_loops(block_b)

        B_decode_block = None
        other_blocks = []
        for op in reversed(self.ops):
            if op not in (self.reduce_op, *[arg.op for arg in self.output_args]):
                if (
                    op.name == "B_decode"
                    or op.name == "mediate0"
                    or op.name == "B_decompress"
                ):
                    B_decode_block = self.sche.get_block(op.name)
                else:
                    block = self.sche.get_block(op.name)
                    other_blocks.append(block)

        write_sch(sch, "cache_read_decode")

        sch.compute_inline(B_decode_block)

        # compute inline
        for block in other_blocks:
            self.sche.compute_inline(block)

        block_shared_local_A = sch.cache_read(block_b, 0, "local")
        block_shared_local_B = sch.cache_read(block_b, 2, "local")
        block_local_C = sch.cache_write(block_b, 0, "local")
        write_sch(sch, "cache_related")

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

        write_sch(sch, "do_split")

        sch.compute_at(block_shared_local_A, tx, preserve_unit_loops=True)
        sch.compute_at(block_shared_local_B, tx, preserve_unit_loops=True)
        sch.reverse_compute_at(block_local_C, j, preserve_unit_loops=True)
        write_sch(sch, "compute_at_related")

        block_local_a_v = sch.get_loops(block_shared_local_A)[-1]
        sch.vectorize(block_local_a_v)
        block_local_b_v = sch.get_loops(block_shared_local_B)[-1]
        sch.vectorize(block_local_b_v)

        write_sch(sch, "decompose_reduction")

        return sch.mod["main"]

    def schedule_inconsistent_shared_decode(
        self, is_a_consistent: bool, is_b_consistent: bool, use_dp4a=False
    ) -> tir.Schedule:
        from ladder.schedule.lop3_intrin import (
            LOP3_FAST_DECODE_INT4_TO_FP16_INTRIN,
            LOP3_FAST_DECODE_INT2_TO_FP16_INTRIN_L8,
            LOP3_FAST_DECODE_INT1_TO_FP16_INTRIN_L8,
            LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN,
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

        num_warps = int(np.prod(self.config.thread))
        warp_size = int(np.prod(self.config.reduce_thread))
        write_sch(sch, "origin")

        block_b = sch.get_block(self.reduce_op.name)
        decode_block = None
        other_blocks = []

        for op in reversed(self.ops):
            if op not in (self.reduce_op, *[arg.op for arg in self.output_args]):
                if "decode" in op.name or "decompress" in op.name:
                    decode_block = self.sche.get_block(op.name)
                else:
                    other_blocks.append(self.sche.get_block(op.name))

        i, j, k = sch.get_loops(block_b)
        block_shared_local_A = sch.cache_read(block_b, 0, "local")
        block_shared_local_B_rescale = sch.cache_read(block_b, 1, "local")

        block_shared_local_B_decompress = sch.cache_read(
            block_shared_local_B_rescale, 0, "local"
        )
        if decode_block != None:
            read_shape = sch.get(decode_block).reads[0].buffer.shape
            write_shape = sch.get(decode_block).writes[0].buffer.shape
            compress_rate = np.prod(write_shape) // np.prod(read_shape)
            bits = 8 // compress_rate
            sch.compute_inline(decode_block)
        block_shared_local_B_prefetch = sch.cache_read(
            block_shared_local_B_decompress, 0, "local"
        )
        for block in other_blocks:
            self.sche.compute_inline(block)
        block_local_C = sch.cache_write(block_b, 0, "local")
        write_sch(sch, "cache_related")
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

        write_sch(sch, "do_split")

        sch.compute_at(block_shared_local_A, tx, preserve_unit_loops=True)
        sch.compute_at(block_shared_local_B_rescale, tx, preserve_unit_loops=True)
        sch.compute_at(block_shared_local_B_decompress, tx, preserve_unit_loops=True)
        sch.compute_at(block_shared_local_B_prefetch, tx, preserve_unit_loops=True)
        sch.reverse_compute_at(block_local_C, j, preserve_unit_loops=True)
        write_sch(sch, "compute_at_related")

        block_local_a_v = sch.get_loops(block_shared_local_A)[-1]
        sch.vectorize(block_local_a_v)

        block_local_b_v = sch.get_loops(block_shared_local_B_prefetch)[-1]
        sch.vectorize(block_local_b_v)
        if use_dp4a:
            vo, vi = sch.split(vk, [None, 4])
        write_sch(sch, "decompose_reduction")
        if decode_block and self.config.fast_decoding:
            try:
                if self.args[0].dtype == "float16":
                    if bits == 4:
                        sch.tensorize(
                            sch.get_loops(block_shared_local_B_decompress)[-1],
                            LOP3_FAST_DECODE_INT4_TO_FP16_INTRIN,
                        )
                    elif bits == 2:
                        sch.tensorize(
                            sch.get_loops(block_shared_local_B_decompress)[-1],
                            LOP3_FAST_DECODE_INT2_TO_FP16_INTRIN_L8,
                        )
                    elif bits == 1:
                        sch.tensorize(
                            sch.get_loops(block_shared_local_B_decompress)[-1],
                            LOP3_FAST_DECODE_INT1_TO_FP16_INTRIN_L8,
                        )
                elif self.args[0].dtype == "int8":
                    # compute the decode bits.
                    if bits == 4:
                        sch.tensorize(sch.get_loops(block_shared_local_B_decompress)[-1], LOP3_FAST_DECODE_INT4_TO_INT8_INTRIN)
                    elif bits == 2:
                        loop = sch.get_loops(block_shared_local_B_decompress)[-1]
                        loop_extent = sch.get_sref(loop).stmt.extent
                        if loop_extent == 16:
                            sch.tensorize(
                                loop, LOP3_FAST_DECODE_INT2_TO_INT8_INTRIN_L16
                            )
                    elif bits == 1:
                        sch.tensorize(
                            sch.get_loops(block_shared_local_B_decompress)[-1],
                            LOP3_FAST_DECODE_INT1_TO_INT8_INTRIN_L16,
                        )
            except Exception as e:
                logger.debug(f"tensorize decode block failed: {e}")

        write_sch(sch, "tensorize_lop3")
        return sch.mod["main"]

    def schedule(self) -> tir.Schedule:
        if len(self.reduce_op.input_tensors) > 1:
            num_args = len(self.args)
            is_lut = False
            if num_args >= 4:
                lut_arg = self.args[2]  # assume the 3rd arg is the lut
                lut_shape = np.prod(lut_arg.shape)
                if lut_shape == 16:
                    is_lut = True
            input0_dtype = self.args[0].dtype
            input1_dtype = self.args[1].dtype
            reduce_op = self.reduce_op
            reduce_input0_dtype = reduce_op.input_tensors[0].dtype
            reduce_input1_dtype = reduce_op.input_tensors[1].dtype
            # TODO(lei): The rule should be updated.
            if self.config.consistent_config:
                is_a_consistent = self.config.consistent_config.is_a_consistent
                is_b_consistent = self.config.consistent_config.is_b_consistent
            else:
                is_a_consistent = reduce_input0_dtype == input0_dtype
                is_b_consistent = reduce_input1_dtype == input1_dtype
            is_consistent = is_a_consistent and is_b_consistent
            use_dp4a = (
                input0_dtype == "int8" and self.reduce_op.output(0).dtype == "int32"
            )
            if is_consistent:
                return self.schedule_consistent()
            else:
                logger.debug(
                    f"the computation is inconsistent, is_a_consistent: {is_a_consistent}, is_b_consistent: {is_b_consistent}"
                )
                if is_lut:
                    return self.schedule_inconsistent_lut(
                        is_a_consistent, is_b_consistent
                    )
                if use_dp4a:
                    if self.config.compute_capability == "80":
                        return self.schedule_inconsistent_shared_decode(
                            is_a_consistent, is_b_consistent, use_dp4a=True
                        )
                    return self.schedule_inconsistent(
                        is_a_consistent, is_b_consistent, use_dp4a=True
                    )

                if self.config.compute_capability == "80":
                    return self.schedule_inconsistent_shared_decode(
                        is_a_consistent, is_b_consistent
                    )
                return self.schedule_inconsistent(is_a_consistent, is_b_consistent)
        else:
            return self.schedule_consistent()
