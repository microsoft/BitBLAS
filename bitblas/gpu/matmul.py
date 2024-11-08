# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: disable=missing-docstring, invalid-name
"""A GEMM schedule rule for GPU operators."""
from dataclasses import dataclass
from typing import Optional, List
from contextlib import suppress
from tvm import tir
from tvm.target import Target
from tvm.tir.stmt import ForKind

from ..base import analysis
from ..base.analysis import get_coalesced_veclen
from .base import GPUScheduleRule
from . import utils
from .matmul_analysis import (
    auto_inline_consumer_chain,
    auto_inline_producers,
    get_in_out_dtypes,
    get_index_map,
    normalize_to_matmul,
    _collect_producers,
    get_reduction_blocks,
)
from .matmul_mma import MatmulTensorizationMMA
from .matmul_wmma import (
    MatmulInt8Tensorization,
    MatmulTensorizationWMMA,
)
from .matmul_mfma import MatmulTensorizationMFMA
from functools import reduce
import logging

logger = logging.getLogger(__name__)


class Matmul(GPUScheduleRule):
    """The schedule rule for matmul-like computation"""

    @dataclass
    class Config:
        block_size_x: int = 8
        block_size_y: int = 8
        vthread_x: int = 1
        vthread_y: int = 1
        micro_size_x: int = 4
        micro_size_y: int = 4
        micro_size_k: int = 8
        vector_size: int = 1
        unroll: int = 256  # 0 means no unroll
        use_shared: bool = True
        storage_align: bool = False
        inner_x: bool = False

    def get_configs(self, target: Target) -> Config:
        """Get the schedule config for the target"""
        if target.kind.name in {"cuda", "rocm", "hip"}:
            return Matmul.Config(
                block_size_x=8,
                block_size_y=16,
                vthread_x=1,
                vthread_y=1,
                micro_size_x=4,
                micro_size_y=4,
                micro_size_k=16,
                vector_size=2,
                unroll=256,
                use_shared=True,
                storage_align=True,
                inner_x=False,
            )
        elif target.kind.name == "opencl" and "android" in str(target.host):
            return Matmul.Config(
                block_size_x=8,
                block_size_y=8,
                vthread_x=1,
                vthread_y=1,
                micro_size_x=8,
                micro_size_y=2,
                micro_size_k=16,
                vector_size=8,
                unroll=64,
                use_shared=False,
                storage_align=False,
                inner_x=True,
            )
        else:
            return Matmul.Config()

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Optional[tir.Schedule]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None
        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        main_block = reduction_blocks[0]
        block_stmt = sch.get(main_block)
        sch = normalize_to_matmul(sch, main_block)
        if sch is None:
            return None

        # Step 1. Check hardware supports tensorization.
        # Tensorization config:
        # If any value of I, J, K is fixed and less than this threshold,
        # tensorization rule will not be applied.
        #TODO check matrix core support, now there is a trick: MI250 can use Matrix core.
        minimal_tensorize_threshold = 64
        block_stmt = sch.get(main_block)
        if target.kind.name == "cuda" and utils.get_sm_version(target) >= 70:
            apply_tensorization: bool = True
            # the batch dimension is not taken into consideration.
            # Analyze read/write buffers and choose correct tensorizer: int8 or fp16.
            in_dtype, out_dtype = get_in_out_dtypes(block_stmt)
            if in_dtype not in ["int8", "float16"]:
                apply_tensorization = False
            for item_var in block_stmt.iter_vars[1:]:
                extent = item_var.dom.extent
                if isinstance(extent,
                              tir.expr.IntImm) and extent.value <= minimal_tensorize_threshold:
                    apply_tensorization = False
            if apply_tensorization:
                if in_dtype == "int8" and out_dtype == "int32":
                    tensorize_sch = MatmulInt8Tensorization().apply(func, target, _)
                elif utils.get_sm_version(target) >= 80:
                    # For A100(sm_80) or more advanced gpu, use MMA tensorization.
                    tensorize_sch = MatmulTensorizationMMA().apply(func, target, _)
                else:
                    # For other GPUs, use WMMA tensorization.
                    tensorize_sch = MatmulTensorizationWMMA().apply(func, target, _)
                if tensorize_sch is not None:
                    return tensorize_sch
        elif target.kind.name == "hip":
            apply_tensorization: bool = True
            # the batch dimension is not taken into consideration.
            # Analyze read/write buffers and choose correct tensorizer: int8 or fp16.
            in_dtype, out_dtype = get_in_out_dtypes(block_stmt)
            if in_dtype not in ["int8", "float16"]:
                apply_tensorization = False
            for item_var in block_stmt.iter_vars[1:]:
                extent = item_var.dom.extent
                if isinstance(extent,
                              tir.expr.IntImm) and extent.value <= minimal_tensorize_threshold:
                    apply_tensorization = False
            if apply_tensorization:
                # For MI250
                tensorize_sch = MatmulTensorizationMFMA().apply(func, target, _)
                if tensorize_sch is not None:
                    return tensorize_sch

        # Step 2. Get schedule config.
        config = self.get_configs(target)

        # Step 3. Schedule matmul
        y_kernel_size = config.vthread_y * config.block_size_y * config.micro_size_y
        x_kernel_size = config.vthread_x * config.block_size_x * config.micro_size_x
        if config.inner_x:
            sch.pad_einsum(
                main_block,
                [1, y_kernel_size, x_kernel_size, config.micro_size_k],
            )
            batch, y, x, k = sch.get_loops(main_block)
        else:
            sch.pad_einsum(
                main_block,
                [1, x_kernel_size, y_kernel_size, config.micro_size_k],
            )
            batch, x, y, k = sch.get_loops(main_block)
        by, vy, ty, yi = sch.split(
            y, [None, config.vthread_y, config.block_size_y, config.micro_size_y])
        bx, vx, tx, xi = sch.split(
            x, [None, config.vthread_x, config.block_size_x, config.micro_size_x])
        ko, ki = sch.split(k, factors=[None, config.micro_size_k])
        sch.reorder(by, bx, vy, vx, ty, tx, ko, ki, yi, xi)
        by = sch.fuse(batch, by)
        sch.bind(bx, "blockIdx.x")
        sch.bind(by, "blockIdx.y")
        sch.bind(vy, "vthread.y")
        sch.bind(vx, "vthread.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        inner_loop = config.micro_size_x if config.inner_x else config.micro_size_y
        if inner_loop % config.vector_size == 0:
            _, v = sch.split(xi, [None, config.vector_size])
            sch.vectorize(v)

        if config.unroll > 0:
            sch.annotate(tx, ann_key="pragma_auto_unroll_max_step", ann_val=config.unroll)
            sch.annotate(tx, ann_key="pragma_unroll_explicit", ann_val=1)

        l2g = sch.cache_write(main_block, 0, "local")
        sch.reverse_compute_at(l2g, tx, preserve_unit_loops=True)
        if config.micro_size_x % config.vector_size == 0:
            _, v = sch.split(sch.get_loops(l2g)[-1], [None, config.vector_size])
            sch.vectorize(v)

        if config.use_shared:

            def _cooperative_fetch(index, vec_len):
                block = sch.cache_read(main_block, index, "shared")
                num_loops = len(sch.get_loops(block))
                sch.compute_at(block, ko, preserve_unit_loops=True)
                loops = sch.get_loops(block)[-num_loops:]
                ty, tx, _, vec = sch.split(
                    sch.fuse(*loops),
                    factors=[config.block_size_y, config.block_size_x, None, vec_len],
                )
                sch.vectorize(vec)
                sch.bind(ty, "threadIdx.y")
                sch.bind(tx, "threadIdx.x")
                if config.storage_align:
                    sch.storage_align(block, 0, axis=1, factor=8, offset=vec_len)
                return block

            a_g2s = _cooperative_fetch(0, vec_len=config.vector_size)
            b_g2s = _cooperative_fetch(1, vec_len=config.vector_size)

            auto_inline_producers(sch, a_g2s)
            auto_inline_producers(sch, b_g2s)
        else:
            auto_inline_producers(sch, main_block)

        auto_inline_consumer_chain(sch, l2g)
        sch.decompose_reduction(main_block, ko)

        # Step 4. Check if there are unbound blocks. Execute fallback scheduling to them.
        def is_scheduled(block: tir.schedule.BlockRV) -> bool:
            loops = sch.get_loops(block)
            loop_kinds = {sch.get(loop).kind for loop in loops}
            return loop_kinds != {ForKind.SERIAL}

        blocks = sch.get_child_blocks(root_block)
        max_threads_per_block = utils.max_threads_per_block(target)  # noqa: F841
        for block in blocks:
            if is_scheduled(block):
                continue
            # no axis of the block is bound to thread or block
            s_loops = sch.get_loops(block)
            bx, tx = sch.split(
                sch.fuse(*s_loops),
                factors=[
                    None,
                    256,
                ],
            )
            sch.bind(bx, "blockIdx.x")
            sch.bind(tx, "threadIdx.x")

        return sch

    def apply_config(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        config,
    ) -> tir.Schedule:
        if "dequantize_info" in func.attrs:
            return self.sch_dequantize_in_register_with_config(func, config)
        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        # in some case conv template will use this rule, but the tile config is not
        # analyzed by matmul expr.
        if len(config.block) != 2:
            logger.debug(f"Warning: block config {config.block} is not valid for matmul, skip.")
            return None

        main_block = reduction_blocks[0]

        block_stmt = sch.get(main_block)

        # cuda core prefer b is [k, j] layout without swizzling.
        index_maps = get_index_map(block_stmt, ["n", "n", "n"])
        if index_maps is None:
            return None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        # Step 0. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
        block = sch.reindex(main_block, ("read", 0))
        sch.transform_layout(block, ("write", 0), a_index_map)
        block = sch.reindex(main_block, ("read", 1))
        sch.transform_layout(block, ("write", 0), b_index_map)
        block = sch.reindex(main_block, ("write", 0))
        sch.transform_layout(block, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)

        # Step 2. Get schedule config.
        block_row_warps = config.block[0] // (config.thread[0] * config.step[0])
        block_col_warps = config.block[1] // (config.thread[1] * config.step[1])
        thread_row_tiles = config.thread[0] // (config.step[0] * 2)
        thread_col_tiles = config.thread[1] // (config.step[1] * 2)
        vthread_row_tiles = (config.step[0] * 2)  # expand vtrhead to avoid load band conflict
        vthread_col_tiles = (config.step[1] * 2)  # expand vtrhead to avoid load band conflict
        chunk = config.rstep[0]

        # Step 3. Schedule matmul
        BM = block_row_warps * vthread_row_tiles * thread_row_tiles
        BN = block_col_warps * vthread_col_tiles * thread_col_tiles
        BK = chunk

        sch.pad_einsum(
            main_block,
            [1, BM, BN, BK],
        )
        batch, y, x, k = sch.get_loops(main_block)
        by, vy, ty, yi = sch.split(y, [None, vthread_row_tiles, block_row_warps, thread_row_tiles])
        bx, vx, tx, xi = sch.split(x, [None, vthread_col_tiles, block_col_warps, thread_col_tiles])
        ko, ki = sch.split(k, factors=[None, BK])
        sch.reorder(by, bx, vy, vx, ty, tx, ko, ki, yi, xi)
        by = sch.fuse(batch, by)
        sch.bind(bx, "blockIdx.x")
        sch.bind(by, "blockIdx.y")
        sch.bind(vy, "vthread.y")
        sch.bind(vx, "vthread.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

        def prod(iterable):
            return reduce(lambda x, y: x * y, iterable, 1)

        l2g = sch.cache_write(main_block, 0, "local")
        sch.reverse_compute_at(l2g, tx, preserve_unit_loops=True)

        def _cooperative_fetch(index, vec_len):
            block = sch.cache_read(main_block, index, "shared")
            num_loops = len(sch.get_loops(block))
            block_local = sch.cache_read(main_block, index, "local")
            sch.compute_at(block_local, ki, preserve_unit_loops=True)
            sch.compute_at(block, ko, preserve_unit_loops=True)
            loops = sch.get_loops(block)[-num_loops:]
            _, ty, tx, vec = sch.split(
                sch.fuse(*loops),
                factors=[None, block_row_warps, block_col_warps, vec_len],
            )

            auto_inline_producers(sch, block)

            def is_trivial_load(block):
                # avoid vectorize under global[v2, v1]] shared[v1, v2] case
                reads = sch.get(block).reads
                writes = sch.get(block).writes
                if len(reads) != 1 or len(writes) != 1:
                    return False
                return all(
                    read.region[-1] == write.region[-1] for read, write in zip(reads, writes))

            if is_trivial_load(block):
                sch.vectorize(vec)

            sch.bind(ty, "threadIdx.y")
            sch.bind(tx, "threadIdx.x")

            _, vec = sch.split(
                sch.fuse(*sch.get_loops(block_local)[-2:]),
                [None, vec_len // prod(config.step)],
            )
            sch.vectorize(vec)

            return block

        for i, input_region in enumerate(sch.get(main_block).reads):
            _buffer_name = input_region.buffer.name.replace("_reindex", "").replace("_pad", "")
            if _buffer_name not in config.cached_tensors:
                logger.warning(
                    f"Warning: {_buffer_name} is not in cached_tensors {config.cached_tensors}, skip."
                )
                continue

            # otherwise cooperative fetch in shared memory.
            vectorize = config.vectorize.get(_buffer_name, 1)

            _cooperative_fetch(i, vec_len=vectorize)

        auto_inline_consumer_chain(sch, l2g)

        _, vec = sch.split(
            sch.fuse(*sch.get_loops(l2g)[-2:]), [None, vectorize // prod(config.step)])
        sch.vectorize(vec)

        sch.decompose_reduction(main_block, ko)
        return sch

    def sch_dequantize_in_register_with_config(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        config,
    ) -> tir.Schedule:
        '''
        For devices without async copy, we can use a simple dequantize schedule without shared memory prefetch.
            quantized weight
                |
                V
            dequantized in register
                |
                V
            save into shared memory
                |
                V
            compute
        '''
        from .intrin import get_lop3_intrin_group

        import_source: List[str] = []

        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        # in some case conv template will use this rule, but the tile config is not
        # analyzed by matmul expr.
        if len(config.block) != 2:
            logger.debug(f"Warning: block config {config.block} is not valid for matmul, skip.")
            return None

        # Check Dequantize Info
        dequantize_info = func.attrs["dequantize_info"]

        def check_dequantize_info(dequantize_info):
            conditions = []
            # currently only support weight only dequantization
            conditions.append(len(dequantize_info) == 1)
            # TODO(@lei) check if the dequantize value name is weight
            return all(conditions)

        assert check_dequantize_info(dequantize_info)

        (weight_decode_info,) = list(dequantize_info.values())

        def check_weight_decode_info(weight_decode_info):
            conditions = []
            # check source format in ["int", "fp", "nf"]
            conditions.append("source_format" in weight_decode_info)
            conditions.append(weight_decode_info["source_format"]["format"] in
                              ["uint", "int", "fp", "nf", "fp_e4m3"])
            # check source bits in [1, 2, 4, 8]
            conditions.append(weight_decode_info["source_format"]["bits"] in [1, 2, 4, 8])
            # check target format in ["float16", "int8"]
            conditions.append("target_format" in weight_decode_info)
            conditions.append(weight_decode_info["target_format"] in ["float16", "int8"])
            return all(conditions)

        assert check_weight_decode_info(weight_decode_info), "Invalid Weight Decode Info"

        main_block = reduction_blocks[0]

        block_stmt = sch.get(main_block)

        # dequant must be 'n' 't' 'n' layout for fast decoding.
        index_maps = get_index_map(block_stmt, ["n", "t", "n"])
        if index_maps is None:
            return None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        # Step 0. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
        block = sch.reindex(main_block, ("read", 0))
        sch.transform_layout(block, ("write", 0), a_index_map)
        block = sch.reindex(main_block, ("read", 1))
        sch.transform_layout(block, ("write", 0), b_index_map)
        block = sch.reindex(main_block, ("write", 0))
        sch.transform_layout(block, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)

        # Step 2. Get schedule config.
        block_row_warps = config.block[0] // (config.thread[0] * config.step[0])
        block_col_warps = config.block[1] // (config.thread[1] * config.step[1])
        thread_row_tiles = config.thread[0] // (config.step[0])
        thread_col_tiles = config.thread[1] // (config.step[1])
        vthread_row_tiles = (config.step[0])  # expand vthread to avoid load band conflict
        vthread_col_tiles = (config.step[1])  # expand vthread to avoid load band conflict
        chunk = config.rstep[0]
        shared_scope = config.shared_scope

        num_ty = block_row_warps
        num_tx = block_col_warps

        # Step 3. Schedule matmul
        BM = block_row_warps * vthread_row_tiles * thread_row_tiles
        BN = block_col_warps * vthread_col_tiles * thread_col_tiles

        # TODO(lei): this is a hack.
        def find_valid_number(k, chunk, magic=16):
            # Start with the largest possible number smaller than chunk that is divisible by 16
            num = (chunk // magic) * magic
            # Iterate downwards to find a number divisible by both 16 and k
            while num > 0:
                if k % num == 0:
                    return num
                num -= magic

            return None  # If no such number is found

        K = func.buffer_map[func.params[0]].shape[-1]
        # This is hack to handle unaligned K and BK
        BK = find_valid_number(K, chunk)
        # Align Factor (Notes: This is also a hack.)
        align_factor = 4  # used to expand the vectorization factor
        sch.pad_einsum(
            main_block,
            [1, BM, BN, BK],
        )
        batch, y, x, k = sch.get_loops(main_block)
        by, vy, ty, yi = sch.split(y, [None, vthread_row_tiles, block_row_warps, thread_row_tiles])
        bx, vx, tx, xi = sch.split(x, [None, vthread_col_tiles, block_col_warps, thread_col_tiles])
        ko, ki, kii = sch.split(k, factors=[None, (BK // align_factor), align_factor])
        sch.reorder(by, bx, vy, vx, ty, tx, ko, ki, kii, yi, xi)
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

        def prod(iterable):
            return reduce(lambda x, y: x * y, iterable, 1)

        l2g = sch.cache_write(main_block, 0, "local")
        sch.reverse_compute_at(l2g, tx, preserve_unit_loops=True)

        def _cooperative_fetch(index, vec_len, align_factor=2):
            block = sch.cache_read(main_block, index, "shared")
            num_loops = len(sch.get_loops(block))
            block_local = sch.cache_read(main_block, index, "local")
            sch.compute_at(block_local, ki, preserve_unit_loops=True)
            sch.compute_at(block, ko, preserve_unit_loops=True)
            loops = sch.get_loops(block)[-num_loops:]
            _, ty, tx, vec = sch.split(
                sch.fuse(*loops),
                factors=[None, block_row_warps, block_col_warps, vec_len],
            )

            auto_inline_producers(sch, block)

            def is_trivial_load(block):
                # avoid vectorize under global[v2, v1]] shared[v1, v2] case
                reads = sch.get(block).reads
                writes = sch.get(block).writes
                if len(reads) != 1 or len(writes) != 1:
                    return False
                return all(
                    read.region[-1] == write.region[-1] for read, write in zip(reads, writes))

            if is_trivial_load(block):
                sch.vectorize(vec)

            sch.bind(ty, "threadIdx.y")
            sch.bind(tx, "threadIdx.x")

            fused = sch.fuse(*sch.get_loops(block_local)[-2:])
            _, vec = sch.split(
                fused,
                [None, align_factor],
            )
            sch.vectorize(vec)

            return block

        for i, input_region in enumerate(sch.get(main_block).reads[:1]):
            _buffer_name = input_region.buffer.name.replace("_reindex", "").replace("_pad", "")
            if _buffer_name not in config.cached_tensors:
                logger.warning(
                    f"Warning: {_buffer_name} is not in cached_tensors {config.cached_tensors}, skip."
                )
                continue

            # otherwise cooperative fetch in shared memory.
            vectorize = config.vectorize.get(_buffer_name, 1)

            _cooperative_fetch(i, vec_len=vectorize, align_factor=align_factor)

        def decode_fetch_to_shared(block, idx):
            # step1. create memory hierarchy
            # global -> local -> shared
            block_shared = sch.cache_read(block, idx, shared_scope)
            sch.compute_at(block_shared, ko, preserve_unit_loops=True)

            decode_factor = get_coalesced_veclen(sch.get(block_shared))
            _, B_shared_vi, _ = sch.split(
                sch.get_loops(block_shared)[-1], factors=[None, 1, decode_factor])
            block_shared_local = sch.cache_read(block_shared, 0, "local")
            # global -> dequantzed_local -> shared
            # step2. inline to local block
            weight_dequantize_block = sch.get_block(weight_decode_info["decode_block"])
            weight_producers = _collect_producers(sch, weight_dequantize_block)
            auto_inline_producers(sch, block_shared_local, weight_producers)

            # get target dequantize buffer's idx
            def get_idx():
                # for LUT dequantize, the expr is LUT(w), the idx is 1
                # maybe we can use a more general and structural based way
                # to analysis the idx
                if weight_decode_info["source_format"]["format"] == "nf":
                    return 1
                return 0

            b_idx = get_idx()
            # global -> prefetch_local -> dequantzed_local -> shared
            block_shared_local_local = sch.cache_read(block_shared_local, b_idx, "local")

            sch.compute_at(block_shared_local, B_shared_vi, preserve_unit_loops=True)
            sch.compute_at(block_shared_local_local, B_shared_vi, preserve_unit_loops=True)

            dequantize_block_local = block_shared_local
            if ("with_scaling" in weight_decode_info and weight_decode_info["with_scaling"]):
                block_local_scales = sch.cache_read(dequantize_block_local, b_idx + 1, "local")
                sch.compute_at(block_local_scales, B_shared_vi, preserve_unit_loops=True)
                # pop the scale block
                auto_inline_producers(sch, block_local_scales)

            if ("with_zeros" in weight_decode_info and weight_decode_info["with_zeros"]):
                block_local_zeros = sch.cache_read(dequantize_block_local, b_idx + 2, "local")
                sch.compute_at(block_local_zeros, B_shared_vi, preserve_unit_loops=True)
                auto_inline_producers(sch, block_local_zeros)

            for producer in weight_producers:
                with suppress(Exception):
                    auto_inline_producers(sch, producer)
                    sch.compute_inline(producer)

            # fast type conversion
            if ("fast_decoding" in weight_decode_info and weight_decode_info["fast_decoding"]):
                source_bit = weight_decode_info["source_format"]["bits"]
                out_dtype = weight_decode_info["target_format"]
                lop3_intrin_info = get_lop3_intrin_group(
                    out_dtype=out_dtype,
                    storage_dtype=weight_decode_info["storage_dtype"],
                    source_format=weight_decode_info["source_format"]["format"],
                    source_bit=source_bit,
                    with_scaling=weight_decode_info["with_scaling"],
                    with_zeros=weight_decode_info["with_zeros"],
                    zeros_mode=weight_decode_info["zeros_mode"],
                )
                sch.tensorize(
                    sch.get_loops(dequantize_block_local)[-1],
                    lop3_intrin_info["compute"],
                )
                import_source.append(lop3_intrin_info["c_source"])

            union_len = (2 + 2)
            B_shared_fused = sch.fuse(*sch.get_loops(block_shared)[-union_len:-2])

            _, B_shared_ty, B_shared_tx = sch.split(B_shared_fused, factors=[None, num_ty, num_tx])
            sch.bind(B_shared_tx, "threadIdx.x")
            sch.bind(B_shared_ty, "threadIdx.y")
            sch.vectorize(sch.get_loops(block_shared)[-1])
            sch.vectorize(sch.get_loops(block_shared_local_local)[-1])

            # cache small tensors, e.g. LUT
            if b_idx:
                block_shared_lut = sch.cache_read(dequantize_block_local, 0, shared_scope)
                sch.reverse_compute_at(block_shared_lut, bx)
                _, B_shared_tx = sch.split(
                    sch.get_loops(block_shared_lut)[-1], factors=[None, num_tx])
                sch.bind(B_shared_tx, "threadIdx.x")
            return block_shared_local

        _ = decode_fetch_to_shared(main_block, 1)

        def fetch_to_local(block, index, align_factor=2):
            # read_b to load
            block_local = sch.cache_read(block, index, "local")
            sch.compute_at(block_local, ki, preserve_unit_loops=True)
            fused = sch.fuse(*sch.get_loops(block_local)[-2:])
            _, vec = sch.split(
                fused,
                [None, align_factor],
            )
            sch.vectorize(vec)
            return block_local

        fetch_to_local(main_block, 1, align_factor=align_factor)

        auto_inline_consumer_chain(sch, l2g)

        l2g_vec = get_coalesced_veclen(sch.get(l2g))
        _, vec = sch.split(sch.fuse(*sch.get_loops(l2g)[-2:]), [None, l2g_vec])
        sch.vectorize(vec)

        sch.decompose_reduction(main_block, ko)
        # plan import source
        if len(import_source) > 0:
            sch.annotate(
                ty,
                ann_key="pragma_import_c",
                ann_val=("\n").join(import_source),
            )
        return sch
