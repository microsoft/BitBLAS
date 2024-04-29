# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Optional, Tuple, Union, List, Dict
from tvm.ir import IRModule
from tvm.ir.transform import PassContext, module_pass
from tvm import relax
from tvm import tir
from enum import Enum
from tvm.ir import GlobalVar
from tvm.tir import IndexMap
from tvm.target import Target
from tvm.tir import IterVar
from tvm.tir.schedule.schedule import BlockRV
from tvm.relax import PyExprMutator
from tvm.relax.expr import Call
from bitblas.gpu.matmul_analysis import (
    get_tensorized_func_and_tags,
    get_propagate_map,
    find_last_producer_from_buffer,
    find_arg_idx_from_buffer_chain,
    layout_propagate_chain,
)
from tvm.dlight.base import (
    analysis,)
from dataclasses import dataclass


def get_reduction_blocks(sch, blocks) -> bool:
    # Get the main computation block
    def is_reduction(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.CommReduce, IterVar.DataPar}

    def is_spatial(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.DataPar}

    # NOTE: We assume there is only one reduction block in the function
    # all blocks are required to be spatial or reduction
    if not all([is_reduction(block) or is_spatial(block) for block in blocks]):
        return None

    # There is only one reduction block
    reduction_blocks = [block for block in blocks if is_reduction(block)]
    if len(reduction_blocks) != 1:
        return None

    return reduction_blocks


class TransformKind(Enum):
    NonTransform = 0
    InterWarpTransform = 1
    IntraWarpTransform = 2


def check_sm_version(arch: str) -> int:
    sm_version = arch.replace("sm_", "")
    return int(sm_version) if sm_version.isdigit() else -1


def get_in_out_dtypes(block: tir.Block) -> Tuple[str]:
    """
    Detect In/Out data types for the given block based on the analysis if read/write buffers.
    """
    assert len(block.reads) > 0 and len(block.writes) > 0
    in_dtype = block.reads[0].buffer.dtype
    out_dtype = block.writes[0].buffer.dtype
    return (in_dtype, out_dtype)


@dataclass
class LayoutTransformHint:
    """
    A dataclass to store the layout transformation hint.
    """

    transform_level: TransformKind
    inter_warp_layout: IndexMap
    intra_warp_layout: IndexMap
    apply_arg_idx: int


@module_pass(opt_level=0, name="InsertLayoutTransform")
class WeightOnlyLayoutPropagation:

    def __init__(
        self,
        transform_level: Union[int, TransformKind] = TransformKind.InterWarpTransform,
        target: Optional[Target] = None,
        faster_conversion: bool = False,
    ) -> None:
        if isinstance(transform_level, int):
            transform_level = TransformKind(transform_level)
        assert transform_level in [
            TransformKind.NonTransform,
            TransformKind.InterWarpTransform,
            TransformKind.IntraWarpTransform,
        ]
        # transform_level 1: only transform the inter-warp memory layout
        # transform_level 2: transform the inter-warp memory layout and the intra-warp memory layout
        self.transform_level = transform_level
        self.target = Target.current() if target is None else target
        # fast type conversion on nvidia gpu also requires weight permutation
        self.faster_conversion = faster_conversion
        # layout transform info to sync the layout in both graph and tir
        self.layout_transform_hints: Dict[str, List[LayoutTransformHint]] = {}

    def detect_propagate_matmul(self, func: tir.PrimFunc, target: Target):
        _, tags = get_tensorized_func_and_tags(func, target, skip_normalize=True, allow_gemv=True)
        if tags is None:
            return False, None
        return True, tags["intrin_info"]

    def transform_matmul(self, g_var: GlobalVar, func: tir.PrimFunc, intrin_info):
        from tvm.tir.tensor_intrin.cuda import (  # pylint: disable=import-outside-toplevel
            get_mma_intrin_group,)

        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None or len(reduction_blocks) != 1:
            return False
        (main_block,) = reduction_blocks

        intrin_group = get_mma_intrin_group(
            load_scope="shared",
            store_scope="shared",
            a_dtype=intrin_info["in_dtype"],
            b_dtype=intrin_info["in_dtype"],
            out_dtype=intrin_info["out_dtype"],
            trans_a=False,
            trans_b=intrin_info["trans_b"],
        )

        _, inter_j, inter_k = intrin_group["micro_kernel"]

        # weight only propagation
        target_scope = ("read", 1)
        weight_buffer = sch.get(main_block).reads[1].buffer

        # checkout whether the weight buffer has dynamic symbol
        def check_dynamic_symbol(buffer):
            return any([isinstance(axis, tir.Var) for axis in buffer.shape])

        if check_dynamic_symbol(weight_buffer):
            print("[BitBLAS] Weight buffer has dynamic symbol, skip weight propagation.")
            return False

        transformed_block = find_last_producer_from_buffer(sch, main_block, weight_buffer)
        if transformed_block is None:
            return False
        if transformed_block != main_block:
            target_scope = ("read", 0)

        reindex_block = sch.cache_read(transformed_block, target_scope[1], "global")

        # create inter-warp memory layout index map
        inter_warp_layout = IndexMap.from_func(
            lambda i, j: (i // inter_j, j // inter_k, i % inter_j, j % inter_k))

        inter_warp_layout = layout_propagate_chain(
            sch,
            main_block,
            sch.get(main_block).reads[1].buffer,
            reindex_block,
            inter_warp_layout,
        )

        sch.transform_layout(
            reindex_block,
            ("read", 0),
            lambda i, j: inter_warp_layout.map_indices([i, j]),
        )
        arg_idx = find_arg_idx_from_buffer_chain(sch, reindex_block,
                                                 sch.get(reindex_block).reads[0].buffer)

        intra_warp_layout = None
        if self.transform_level.value >= TransformKind.IntraWarpTransform.value:
            intra_warp_layout, _ = get_propagate_map(intrin_info["trans_b"])
            intra_warp_layout = layout_propagate_chain(
                sch,
                main_block,
                sch.get(main_block).reads[1].buffer,
                reindex_block,
                intra_warp_layout,
            )
            sch.transform_layout(
                reindex_block,
                ("read", 0),
                lambda i, j, ii, jj: (
                    i,
                    j,
                    *intra_warp_layout.map_indices([ii, jj]),
                ),
            )

        self.layout_transform_hints[g_var] = [
            LayoutTransformHint(
                transform_level=self.transform_level,
                inter_warp_layout=inter_warp_layout,
                intra_warp_layout=intra_warp_layout,
                apply_arg_idx=arg_idx,
            )
        ]

        return sch.mod["main"]

    def transform_module(  # pylint: disable=missing-function-docstring
        self,
        mod: IRModule,
        _: PassContext,
    ) -> IRModule:
        if self.target.kind.name != "cuda":
            # currently weight propagation only support nvidia gpus
            return mod

        propagate_candidates = {}
        propagated_funcs = {}  # some funcs may not be able to transform
        candidates_intrin_info = {}
        decoded_funcs = {}
        for g_var, func in mod.functions_items():
            if not isinstance(func, tir.PrimFunc):
                continue
            if g_var.name_hint != "main":
                # Note: this can be applied to any function which can be transformed to matmul (e.g., conv2d)
                # for mlc we only consider matmul
                # detect the pattern
                is_matmul, intrin_info = self.detect_propagate_matmul(func, self.target)

                if (func.attrs is not None and "dlight.do_not_tensorize" in func.attrs.keys()):
                    # currently we only support tensorize propagation
                    continue

                if is_matmul:
                    if "dequantize_info" in func.attrs:
                        decoded_funcs[g_var] = func
                    if self.transform_level != TransformKind.NonTransform:
                        # lift tags to the function as it has intrinsic information that can be reused.
                        propagate_candidates[g_var] = func
                        candidates_intrin_info[g_var] = intrin_info

        for g_var, func in propagate_candidates.items():
            updated_func = self.transform_matmul(g_var, func, candidates_intrin_info[g_var])
            if updated_func:
                updated_func = updated_func.with_attrs({
                    "transform_kind": self.transform_level.value,
                    "weight_transform_kind": True,
                })
                propagated_funcs[g_var] = updated_func
                mod[g_var] = updated_func

        @relax.expr_functor.mutator
        class TensorCoreLayoutMutator(PyExprMutator):
            """Mutator that performs transformation."""

            def __init__(
                self,
                transform_level: TransformKind = TransformKind.NonTransform,
                layout_transform_hints: Optional[Dict[str, List[LayoutTransformHint]]] = None,
            ):
                if layout_transform_hints is None:
                    layout_transform_hints = {}
                super().__init__()
                self.transform_level = transform_level
                self.layout_transform_hints = layout_transform_hints

            def tc_layout_transform(self, call_node: Call) -> Call:
                if self.transform_level == TransformKind.NonTransform:
                    return super().visit_call_(call_node)
                g_var = call_node.args[0]
                if g_var not in propagated_funcs:
                    return super().visit_call_(call_node)
                args = list(call_node.args[1])
                # assume we only have weight propagation currently
                (weight_layout_hint,) = self.layout_transform_hints[g_var]
                weight = args[weight_layout_hint.apply_arg_idx]
                weight = self.builder_.emit(
                    relax.op.layout_transform(
                        weight,
                        index_map=lambda i, j: weight_layout_hint.inter_warp_layout.map_indices(
                            [i, j]),
                    ))
                if self.transform_level.value >= TransformKind.IntraWarpTransform.value:
                    weight = self.builder_.emit(
                        relax.op.layout_transform(
                            weight,
                            index_map=lambda i, j, ii, jj: (
                                i,
                                j,
                                *weight_layout_hint.intra_warp_layout.map_indices([ii, jj]),
                            ),
                        ))

                call_node = self.builder_.emit(
                    relax.call_tir(
                        g_var,
                        args[:weight_layout_hint.apply_arg_idx] + [weight] +
                        args[weight_layout_hint.apply_arg_idx + 1:],
                        out_sinfo=call_node.struct_info,
                    ))
                return call_node

            def visit_call_(self, call_node: Call):
                return self.tc_layout_transform(call_node)

            def transform(
                self,
                mod: IRModule,
            ):
                for gv, func in mod.functions_items():
                    if isinstance(func, relax.Function):
                        updated_func = self.visit_expr(func)
                        self.builder_.update_func(gv, updated_func)
                new_mod = self.builder_.get()
                new_mod = new_mod.with_attrs(mod.attrs) if mod.attrs else new_mod
                for gv, func in new_mod.functions_items():
                    mod.update_func(gv, func)
                return mod

        mod = TensorCoreLayoutMutator(
            transform_level=self.transform_level,
            layout_transform_hints=self.layout_transform_hints,
        ).transform(mod)

        @relax.expr_functor.mutator
        class FastTypeConversionLayoutMutator(PyExprMutator):
            """Mutator that performs transformation."""

            def __init__(self, faster_conversion: bool = False):
                super().__init__()
                self.faster_conversion = faster_conversion

            def lop3_layout_transform(self, call_node: Call) -> Call:
                if not self.faster_conversion:
                    return super().visit_call_(call_node)

                from bitblas.ops.impl import tir_interleave_weight

                g_var = call_node.args[0]
                if g_var not in decoded_funcs:
                    return super().visit_call_(call_node)

                args = list(call_node.args[1])
                func = decoded_funcs[g_var]
                if "dequantize_info" not in func.attrs:
                    return super().visit_call_(call_node)
                dequantize_info = dict(func.attrs["dequantize_info"])
                assert len(dequantize_info) == 1
                (weight_dequantize_info,) = dequantize_info.values()

                sch = tir.Schedule(func)
                dequantize_block = sch.get_block(weight_dequantize_info["decode_block"])

                # weight is the first read buffer if format in ["int", "uint"], otherwise the second read buffer, nf .etc
                source_format = weight_dequantize_info["source_format"]["format"]
                source_bits = weight_dequantize_info["source_format"]["bits"]
                target_dtype = weight_dequantize_info["target_format"]

                if source_format in ["int", "uint"]:
                    weight_buffer = sch.get(dequantize_block).reads[0].buffer
                elif source_format in ["nf"]:
                    weight_buffer = sch.get(dequantize_block).reads[1].buffer
                else:
                    raise ValueError(f"Unsupported source format {source_format}")

                # update func with dequantize_info
                dequantize_info["fast_decoding"] = True
                self.builder_.update_func(g_var,
                                          func.with_attrs({"dequantize_info": dequantize_info}))

                weight_idx = find_arg_idx_from_buffer_chain(sch, dequantize_block, weight_buffer)
                weight = args[weight_idx]

                weight_shape = weight_buffer.shape
                # reshape the weight shape to 2d
                reshape_weight = self.builder_.emit(
                    relax.op.reshape(weight, (-1, weight_shape[-1])))
                # register g_var to the func
                lop3_interleave_func = tir_interleave_weight(
                    N=reshape_weight.struct_info.shape[0],
                    QK=reshape_weight.struct_info.shape[1],
                    bits=source_bits,
                    target_dtype=target_dtype,
                    storage_dtype=reshape_weight.struct_info.dtype,
                )
                interleave_gvar = self.builder_.add_func(
                    lop3_interleave_func.without_attr("global_symbol"),
                    "tir_interleave_weight",
                )
                lop3_interleave_weight = self.builder_.emit(
                    relax.call_tir(
                        interleave_gvar,
                        [reshape_weight],
                        out_sinfo=reshape_weight.struct_info,
                    ),)
                reshape_weight = self.builder_.emit(
                    relax.op.reshape(lop3_interleave_weight, weight_shape))
                call_node = self.builder_.emit(
                    relax.call_tir(
                        g_var,
                        args[:weight_idx] + [reshape_weight] + args[weight_idx + 1:],
                        out_sinfo=call_node.struct_info,
                    ),)

                return call_node

            def visit_call_(self, call_node: Call):
                return self.lop3_layout_transform(call_node)

            def transform(
                self,
                mod: IRModule,
            ):
                for gv, func in mod.functions_items():
                    if isinstance(func, relax.Function):
                        updated_func = self.visit_expr(func)
                        self.builder_.update_func(gv, updated_func)
                new_mod = self.builder_.get()
                new_mod = new_mod.with_attrs(mod.attrs) if mod.attrs else new_mod
                for gv, func in new_mod.functions_items():
                    mod.update_func(gv, func)
                return mod

        mod = FastTypeConversionLayoutMutator(
            faster_conversion=self.faster_conversion).transform(mod)
        mod = relax.transform.LegalizeOps()(mod)
        return mod
