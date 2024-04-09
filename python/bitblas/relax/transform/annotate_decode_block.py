# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Dict, Tuple
from tvm.ir import IRModule
from tvm.ir.transform import PassContext, module_pass
from tvm import tir
from tvm.tir.schedule import BlockRV
from mlc_llm.quantization import quantization_schemes, GroupQuantizationSpec
from bitblas.gpu.gemv import is_gemv
from bitblas.gpu.matmul_analysis import (
    get_reduction_blocks,
    get_index_map,
    get_root_block,
    get_dequantize_block,
)
from bitblas.base import (
    normalize_prim_func,
    try_inline_contiguous_spatial,
)


# Define a module pass to annotate dequantization information
@module_pass(opt_level=0, name="AnnotateDecodeInformation")
class AnnotateDecodeInformation:

    def __init__(self, spec: str = "q4f16_0"):
        # Validate and store the specified quantization scheme
        if spec not in quantization_schemes:
            raise ValueError(f"Quantization scheme {spec} not found")
        self.quantize_scheme = quantization_schemes[spec]

    def detect_matmul(self, func: tir.PrimFunc) -> bool:
        """Detect if the given function represents a matrix multiplication."""
        sch = tir.Schedule(func)
        root_block = get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        # Identify reduction blocks to infer matmul operations
        reduction_blocks = get_reduction_blocks(sch, blocks)
        if not reduction_blocks:
            return False

        # Check for index map patterns typical of matmul operations
        main_block = reduction_blocks[0]
        main_block_stmt = sch.get(main_block)
        index_maps = get_index_map(main_block_stmt)
        _is_matmul = index_maps is not None

        block_infos = normalize_prim_func(sch)
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        block_info = block_infos[0]
        _is_gemv = True
        if len(block_info.iters) not in [2, 3]:
            # either [B, S, R] = [B, S, R] * [B, R]
            # or [S, R] = [S, R] * [R]
            _is_gemv = False
        if _is_gemv:
            _is_gemv = is_gemv(sch, block_info)
        return _is_matmul or _is_gemv

    def transform_module(self, mod: IRModule, _: PassContext) -> IRModule:
        """Annotate dequantize information for all applicable functions in the module."""
        for g_var, func in mod.functions.items():
            if not isinstance(func, tir.PrimFunc) or g_var.name_hint == "main":
                continue

            if not self.detect_matmul(func):
                continue  # Process only if matmul is detected

            sch = tir.Schedule(func)
            root_block = get_root_block(sch)
            blocks = sch.get_child_blocks(root_block)
            dequantize_block = get_dequantize_block(sch, blocks)
            if dequantize_block is None:
                continue  # Skip if no dequantize block is found

            # Prepare dequantize info annotation
            dequantize_info = self.prepare_dequantize_info(sch, dequantize_block)

            # Annotate function with dequantize information
            mod[g_var] = func.with_attr("dequantize_info", dequantize_info)
        return mod

    def prepare_dequantize_info(self, sch: tir.Schedule, dequantize_block: BlockRV) -> Dict:
        """Generate dequantize information for a given block."""
        block_stmt = sch.get(dequantize_block)
        block_name = block_stmt.name_hint
        dequantize_info = {block_name: {"decode_block": block_name, "fast_decoding": False}}

        quantize_spec = self.quantize_scheme.linear_weight
        if isinstance(quantize_spec, GroupQuantizationSpec):
            dequantize_info[block_name].update({
                "with_scaling": True,
                "group_size": quantize_spec.group_size,
            })

        # Determine source format based on quantization mode
        quantize_mod = quantize_spec.mode
        bits, source_format = self.parse_quantize_mode(quantize_mod)
        dequantize_info[block_name]["source_format"] = {
            "bits": bits,
            "format": source_format,
        }

        # Set storage and target data types
        storage_dtype = self.get_storage_dtype(block_stmt, source_format)
        dequantize_info[block_name]["storage_dtype"] = storage_dtype
        dequantize_info[block_name]["target_format"] = quantize_spec.dtype

        return dequantize_info

    def parse_quantize_mode(self, quantize_mod: str) -> Tuple[int, str]:
        """Extract bits and format from quantization mode."""
        if quantize_mod.startswith("int"):
            return int(quantize_mod[3:]), "int"
        elif quantize_mod.startswith("uint"):
            return int(quantize_mod[4:]), "uint"
        raise ValueError(f"Unsupported mode {quantize_mod}")

    def get_storage_dtype(self, block_stmt: BlockRV, source_format: str) -> str:
        """Determine storage data type based on source format."""
        return (block_stmt.reads[0].buffer.dtype
                if "nf" not in source_format else block_stmt.reads[1].buffer.dtype)
