from typing import Dict, List, Tuple

import numpy as np

from ..arch import Arch
from ..config import Config, Stride, TileDict, LadderConfig
from ..graph import IRNode, Node
from .common import factorize, get_all_factors
from .default import DefaultPolicy


class LadderPolicy(DefaultPolicy):
    def __init__(self, output_nodes: List[Node], arch: Arch) -> None:
        super().__init__(output_nodes, arch)
        self.wmma_m = 16
        self.wmma_n = 16
        compute_dtype = str(self.output_nodes[-1]._dtypes[-1])
        self.wmma_k = 32 if compute_dtype == "int32" else 16

    def _compute_tc_strides(self, node: IRNode, tile: List[int], rstep: Dict[str, int]={}) -> Tuple[Stride, Stride, Stride]:
        shapes = node.propogate_reduction_inputs(tile, rstep)
        AS_shape, BS_shape = shapes.values()
        CS_shape = tile
        A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n = node.infer_tensorcore_axis()
        # applying strides
        offset = 0
        output_dtype = node.outputs[0].dst_node._dtypes[0]
        if len(node.raxis) == 1:
            if output_dtype.bits == 16:
                offset = 8
            elif output_dtype.bits == 8:
                offset = 16
        A_high_ax = min(A_ax_m, A_ax_k)
        B_high_ax = min(B_ax_n, B_ax_k)
        C_high_ax = min(C_ax_m, C_ax_n)
        A_stride = Stride(stride=np.prod(AS_shape[A_high_ax+1:]) + offset, ax=A_high_ax)
        B_stride = Stride(stride=np.prod(BS_shape[B_high_ax+1:]) + offset, ax=B_high_ax)
        C_stride = Stride(stride=np.prod(CS_shape[C_high_ax+1:]) + offset, ax=C_high_ax)
        return A_stride, B_stride, C_stride

    def infer_node_smem_usage(self, td: TileDict, node: IRNode):
        value, cached_tensors = super().infer_node_smem_usage(td, node)
        ladder_configs = node.get_tag("ladder_config")
        if ladder_configs:
            pipeline_stage = ladder_configs[2] if len(ladder_configs) > 2 else 1
            value *= pipeline_stage
        return value, cached_tensors

    def _assign_reduce_step(self, node):
        if not node.get_tag("tensorCoreConfig"):
            return super()._assign_reduce_step(node)
        result = {}
        output_shape = node.reduce_op.output(0).shape
        output_dtype = node.reduce_op.output(0).dtype
        ladder_configs = node.get_tag("ladder_config")
        pipeline_stage = ladder_configs[2] if len(ladder_configs) > 2 else 1
        # assume A is always not transposed.
        is_matmul = (output_shape[-1] == self.wmma_n and output_shape[-2] == self.wmma_n)

        if len(output_shape) == 2:
            M = output_shape[0]
            N = output_shape[1]
        elif len(output_shape) == 4:
            if is_matmul:
                M = output_shape[0] * output_shape[2]
                N = output_shape[1] * output_shape[3]
            else:
                M = output_shape[0] * output_shape[1] * output_shape[2]
                N = output_shape[3]
        elif len(output_shape) == 5:
            # it's batched matmul
            M = output_shape[1] * output_shape[3]
            N = output_shape[2] * output_shape[4]
        elif len(output_shape) == 6 and is_matmul:
            M = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[-2]
            N = output_shape[3] * output_shape[-1]
        else:
            print(output_shape)
            raise NotImplementedError

        A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n = node.infer_tensorcore_axis()
        transpose_B = B_ax_k > B_ax_n        
        kernel_shape = node.reduce_op.input_tensors[1].shape
        input_shape = node.args[0].shape
        if len(kernel_shape) == 2:
            if transpose_B:
                K = kernel_shape[1]
            else:
                K = kernel_shape[0]
        elif len(kernel_shape) == 4:
            if transpose_B:
                K = kernel_shape[1] * kernel_shape[3]
            else:
                K = kernel_shape[0] * kernel_shape[2]
        elif len(kernel_shape) == 5:
            if transpose_B:
                K = kernel_shape[2] * kernel_shape[4]
            else:
                K = kernel_shape[1] * kernel_shape[3]

        if len(input_shape) == 2:
            AK = input_shape[1]
        elif len(input_shape) == 4:
            AK = input_shape[1] * input_shape[3]
        elif len(input_shape) == 5:
            AK = input_shape[2] * input_shape[-1]
        elif len(input_shape) == 6:
            is_nhwc = input_shape[1] == input_shape[2]
            if is_nhwc:
                AK = input_shape[3] * input_shape[-1]
            else:
                AK = input_shape[1] * input_shape[-1]
        print(input_shape)
        print(f"Considering a gemm problem M N K CHANNEL", M, N, K)

        if len(node.raxis) == 1:
            for k in node.raxis:
                if output_dtype == 'float16':
                    result[k] = 32
                elif output_dtype == 'int32' or output_dtype == 'int8':
                    result[k] = 64
                else:
                    raise NotImplementedError
        else:
            for i, k in enumerate(node.raxis):
                if i == 0:
                    if output_dtype == 'int32' or output_dtype == 'int8':
                        result[k] = 64
                        continue
                    if AK % 32 != 0 and AK % 16 == 0:
                        result[k] = 16
                        continue
                    if node.raxis[k] % 2 == 0:
                        if M <= 128 or N <= 512:
                            if N <= 64 and (K % 96 == 0 and K > 96) and pipeline_stage == 1:
                                result[k] = 96
                            elif K > 64 and K % 64 == 0 and pipeline_stage == 1:
                                result[k] = 64
                            else:
                                result[k] = 32
                        else:
                            result[k] = 32
                else:
                    result[k] = 1
        return result

    def _expand_reduce_axis(self, td):
        return

    def get_node_reduce_step_candidates(self, node):
        if not node.get_tag("tensorCoreConfig"):
            return super().get_node_reduce_step_candidates(node)
        else:
            # must be a a multiple of wmma_k
            return {k : [x * self.wmma_k for x in get_all_factors(node.raxis[k] // self.wmma_k)] for k in node.raxis}

    def check_tile_shape_isvalid(self, td: TileDict):
        for node in self.ordered_nodes:
            if node.get_tag("tensorCoreConfig"):
                ax_m, ax_n = node.get_tag("tensorCoreConfig")
                block_m, block_n = td.tile_map[node][ax_m], td.tile_map[node][ax_n]
                wmma_invalid = [block_m % wmma_m or block_n % wmma_n for wmma_m, wmma_n in [(16, 16)]]
                if all(wmma_invalid):
                    return False
                if any([x and (y % x) for x, y in zip(td.tile_map[node], node.get_space_dim())]):
                    return False
        return super().check_tile_shape_isvalid(td)

    def compute_node_stride_map(self, node: IRNode, td: TileDict):
        if not node.get_tag("tensorCoreConfig"):
            return super().compute_node_stride_map(node, td)
        AS_stride, BS_stride, C_stride = self._compute_tc_strides(node, td.get_tile(node), td.get_rstep(node))
        A_stride, B_stride, _ = self._compute_tc_strides(node, td.get_tile(node))
        output_strides = {int(edge.src_id + len(node.inputs)): C_stride for edge in node.outputs}
        tensor_strides = {}
        # when connected to shared input, should use full stride without rstep
        for i, (stride, stride_full) in enumerate(zip([AS_stride, BS_stride], [A_stride, B_stride])):
            name = node.reduce_op.input_tensors[i].name
            tensor_strides[name] = stride

            arg_names = [arg.name for arg in node.args]
            if name in arg_names:
                input_id = arg_names.index(name)
                src_node = node.inputs[input_id].src_node
                if not src_node.is_placeholder():
                    tensor_strides[name] = stride_full

        return output_strides, tensor_strides
    
    def _compute_thread_raster_factor(self, node: IRNode, td: TileDict):
        tile = td.get_tile(node)
        rstep = td.get_rstep(node)
        raster_factor = 0
        input_shape = node.args[0].shape
        kernel_shape = node.args[1].shape
        output_shape = node.reduce_op.output(0).shape
        _dtype = self.output_nodes[-1]._dtypes[-1]
        size_per_element = _dtype.bits // 8
        # assume A is always not transposed.
        if len(output_shape) == 4:
            M = output_shape[0] * output_shape[2]
            N = output_shape[1] * output_shape[3]
        elif len(output_shape) == 2:
            M = output_shape[0]
            N = output_shape[1]
        elif len(output_shape) == 5:
            M = output_shape[1] * output_shape[3]
            N = output_shape[2] * output_shape[4]
        elif len(output_shape) == 6:
            # nhwc1616
            M = output_shape[0] * output_shape[-2]
            N = output_shape[1] * output_shape[2] * output_shape[3] * output_shape[-1]
        else:
            raise NotImplementedError

        A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n = node.infer_tensorcore_axis()
        transpose_B = B_ax_k > B_ax_n        
        kernel_shape = node.args[1].shape
        if len(kernel_shape) == 2:
            if transpose_B:
                K = kernel_shape[1]
            else:
                K = kernel_shape[0]
        elif len(kernel_shape) == 4:
            if transpose_B:
                K = kernel_shape[1] * kernel_shape[3]
            else:
                K = kernel_shape[0] * kernel_shape[2]
        elif len(kernel_shape) == 5:
            if transpose_B:
                K = kernel_shape[2] * kernel_shape[4]
            else:
                K = kernel_shape[1] * kernel_shape[3]
        total_size = (M * N  + M * K + N * K) * size_per_element
        if total_size < 6 * 1024 * 1024 and total_size > 0:
            print(f"total size {total_size} is too small")
            return raster_factor
        raster_factor = int(self.arch.compute_max_core ** 0.5)
        return raster_factor
    
    def _assign_block_size(self, node: Node, td: TileDict, block_size: int):
        if not node.get_tag("tensorCoreConfig"):
            return super()._assign_block_size(node, td, block_size)
        ax_m, ax_n = node.get_tag("tensorCoreConfig")
        if block_size % self.arch.warp_size != 0:
            return None
        tile, rsteps = td.get_tile(node), td.get_rstep(node)
        warps = block_size // self.arch.warp_size
        ndim = len(tile)
        wmma = [self.wmma_m, self.wmma_n, self.wmma_k]
        wmma_tile = [1 for i in range(ndim)]
        wmma_tile[ax_m] = wmma[0]
        wmma_tile[ax_n] = wmma[1]
        space = [tile[i] // wmma_tile[i] for i in range(ndim)]
        if tile[ax_m] % wmma_tile[ax_m] != 0 or tile[ax_n] % wmma_tile[ax_n]:
            return None
        if np.prod(space) % warps != 0:
            return None
        factors = factorize(np.prod(space) // warps)

        def _score(node, thread): # small is better
            score = 0
            block_tile = [int(np.ceil(tile[i] / thread[i])) for i in range(ndim)]
            shape = node.propogate_inputs(block_tile)
            for edge in node.inputs:
                score += np.prod(shape[edge.dst_id]) / self.arch.bandwidth[1]
            return score

        warp_tile = wmma_tile.copy()
        for factor in reversed(factors):
            score_map = {}
            for i in range(ndim):
                if tile[i] % (warp_tile[i] * factor) != 0:
                    continue
                warp_tile[i] *= factor
                score_map[i] = (_score(node, warp_tile), i)
                warp_tile[i] //= factor
            if len(score_map) == 0:
                return None
            dim_order = sorted(score_map.keys(), key=lambda x:score_map[x])
            warp_tile[dim_order[0]] *= factor

        ladder_configs = node.get_tag("ladder_config")
        propagate_inter_a, propagate_inter_b = ladder_configs[0:2]
        pipeline_stage = ladder_configs[2] if len(ladder_configs) > 2 else 1
        codegen_dict = Config()
        codegen_dict.use_tc = self.arch.compute_capability
        codegen_dict.block = tile
        codegen_dict.use_ladder = True
        codegen_dict.warp = warp_tile
        codegen_dict.rstep = [int(rsteps[ax]) for ax in node.raxis]
        codegen_dict.cached_tensors = td.cached_tensors_map[node]
        codegen_dict.wmma = wmma
        codegen_dict.raster_factor = self._compute_thread_raster_factor(node, td)
        codegen_dict.schedule_stages = [stage.name for stage in node._schedule_compute_stages]
        codegen_dict.complete_config(node)
        codegen_dict.pipeline_stage = pipeline_stage
        codegen_dict.ladder_config = LadderConfig(propagate_inter_a, propagate_inter_b, pipeline_stage)
        return codegen_dict
