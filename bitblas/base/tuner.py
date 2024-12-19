# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas import tvm
from typing import List, Optional, Dict, Literal, Callable, Union
from tvm import tir, IRModule
from tvm.tir import PrimFunc
from .analysis import find_var_from_func
from bitblas.base.arch import CUDA, CDNA
from bitblas.base.roller.policy import TensorCorePolicy, DefaultPolicy
from bitblas.gpu.matmul_analysis import get_tensorized_func_and_tags
import itertools
from tvm.ir.supply import GlobalVarSupply
from bitblas.base.base_scheduler import BaseScheduler
from bitblas.base.utils import apply_and_build as tir_apply_and_build
from bitblas.tl.tuner import apply_and_build as tl_apply_and_build
from bitblas.utils import retrieve_func_from_module
import logging

logger = logging.getLogger(__name__)


def fast_tune_tir(
    func: PrimFunc,
    target: tvm.target.Target,
    topk: int = 10,
    parallel_build: bool = True,
    data_distribution: Literal["uniform", "onefill"] = "uniform",
):
    # check the function is a primfunc
    if not isinstance(func, tir.PrimFunc):
        raise ValueError("Only support func is PrimFunc")  # pragma: no cover

    if target.kind.name not in ["cuda", "hip"]:
        logger.error("Only support CUDA and hip target")
        return None, None

    specilized_func = func
    if func.attrs is not None and "opt_shapes" in func.attrs:
        opt_shapes = func.attrs["opt_shapes"]
        # should be int value
        if not all([isinstance(v.value, int) for v in opt_shapes.values()]):
            logger.error("The opt_shapes should be int value")
            return None, None
        # currently only support one dynamic range
        if len(opt_shapes) > 1:
            logger.error("Currently only support one dynamic range")
            return None, None

        for buffer in func.buffer_map.values():
            for axis in buffer.shape:
                if (isinstance(axis, tvm.tir.Var) and axis.name not in opt_shapes):
                    raise NotImplementedError(
                        "Currently do not support fast tune with none-dynamic range set")
        if opt_shapes:
            for name, shape in opt_shapes.items():
                var = find_var_from_func(func, name)
                specilized_func = func.specialize({
                    var: shape.astype(var.dtype)
                }).with_attr("is_specialized")

    if target.kind.name == "cuda":
        arch = CUDA(target)
    elif target.kind.name == "hip":
        arch = CDNA(target)
    else:
        raise ValueError(f"Unsupported target: {target.kind.name}")

    policy = DefaultPolicy(func=func, arch=arch)
    try:
        specilized_func, tags = get_tensorized_func_and_tags(specilized_func, arch.target)
    except Exception as e_msg:
        logger.debug("Get tensorized func and tags failed: ", e_msg)
        tags = None
    if tags:
        policy = TensorCorePolicy(func=specilized_func, arch=arch, tags=tags)

    configs = policy.emit_config(topk)

    if len(configs) == 0:
        raise ValueError("No valid config generated")

    cpresults, best = tir_apply_and_build(
        func,
        configs,
        arch,
        parallel_build=parallel_build,
        data_distribution=data_distribution,
    )

    return cpresults, best


def fast_tune_tilelang(
    scheduler: BaseScheduler,
    target: tvm.target.Target,
    topk: int = 10,
    parallel_build: bool = True,
    data_distribution: Literal["uniform", "onefill"] = "uniform",
):
    if target.kind.name not in ["cuda", "hip"]:
        logger.error("Only support CUDA and hip target")
        return None, None

    arch: Union[CUDA, CDNA] = None
    if target.kind.name == "cuda":
        arch = CUDA(target)
    elif target.kind.name == "hip":
        arch = CDNA(target)
    else:
        raise ValueError(f"Unsupported target: {target.kind.name}")

    specialized_scheduler = scheduler
    if scheduler.has_dynamic_range():
        specialized_scheduler = scheduler.specialize_from_dynamic_range()
    tuning_configs = specialized_scheduler.get_hardware_aware_configs(arch, topk)
    assert len(tuning_configs) > 0, "No tuning config found for this operator."
    cpresults, best = tl_apply_and_build(
        scheduler,
        tuning_configs,
        arch=arch,
        parallel_build=parallel_build,
        data_distribution=data_distribution)
    return cpresults, best


def fast_tune(
    func_or_scheduler: Union[PrimFunc, BaseScheduler],
    target: tvm.target.Target,
    topk: int = 10,
    parallel_build: bool = True,
    data_distribution: Literal["uniform", "onefill"] = "uniform",
):
    if isinstance(func_or_scheduler, tir.PrimFunc):
        return fast_tune_tir(func_or_scheduler, target, topk, parallel_build, data_distribution)
    elif isinstance(func_or_scheduler, BaseScheduler):
        return fast_tune_tilelang(func_or_scheduler, target, topk, parallel_build,
                                  data_distribution)
    else:
        raise ValueError("Not supported type: ", type(func_or_scheduler))


# always use the first function as the base
def collect_buffers_to_declare(func):
    params = []
    # collect dynamic symbolic
    dyn_symbolic: List[tvm.tir.Var] = []
    buffers_to_declare = []
    for param in func.params:
        if param not in func.buffer_map:
            continue
        buffer = func.buffer_map[param]
        for axis in buffer.shape:
            if isinstance(axis, tvm.tir.Var) and axis not in dyn_symbolic:
                dyn_symbolic.append(axis)
        buffers_to_declare.append(buffer)
        params.append(buffer.data)

    # the args should be buffers + dynamic symbolic
    params += list(dyn_symbolic)

    return params, buffers_to_declare


def refactor_specialized_func(g_var, func, params, buffers_to_declare):
    body = func.body
    attrs = func.attrs
    global_symbol = g_var
    opt_shapes: Optional[Dict[str, int]] = None
    if "opt_shapes" in func.attrs:
        opt_shapes = func.attrs["opt_shapes"]

    assert opt_shapes is not None, "The opt_shapes should not be None"

    def serialize_name(opt_shapes: Dict):
        return "_opt_" + "_".join([f"{k}_{v}" for k, v in opt_shapes.items()])

    global_symbol += serialize_name(opt_shapes)
    ret_type = func.ret_type
    for buf in buffers_to_declare:
        body = tvm.tir.DeclBuffer(buf, body=body)

    # device func must be private
    device_func = tvm.tir.PrimFunc(
        params, body, ret_type, attrs=attrs).without_attr("global_symbol")
    return global_symbol, device_func


def create_dispatch_func(g_var: str, func: tir.PrimFunc, refactored_funcs: List[str]):
    global_symbol = g_var
    attrs = func.attrs
    buffer_map = func.buffer_map
    params = func.params
    ret_type = func.ret_type

    # collect dynamic symbolic
    dyn_symbolic: List[tvm.tir.Var] = []
    _invoke_params = []
    for param in func.params:
        if param not in func.buffer_map:
            continue
        buffer = func.buffer_map[param]
        for axis in buffer.shape:
            if isinstance(axis, tvm.tir.Var) and axis not in dyn_symbolic:
                dyn_symbolic.append(axis)
        _invoke_params.append(buffer.data)
    _invoke_params += list(dyn_symbolic)

    func_range: List[int] = []
    global_symbols = []
    for g_var, refactor_func in refactored_funcs:
        opt_shapes = refactor_func.attrs["opt_shapes"]
        func_range.append(list(opt_shapes.values())[0])
        global_symbols.append(g_var)

    # TODO(lei): general the dispatch function to support multiple dynamic symbolics
    assert len(dyn_symbolic) == 1, "Only support one dynamic symbolics currently"

    ib = tvm.tir.ir_builder.create()
    syb = list(dyn_symbolic)[-1]
    last_range = 0
    for i, (_range, g_var) in enumerate(zip(func_range, global_symbols)):
        if i == 0:
            with ib.if_scope(syb <= _range):
                ib.emit(tvm.tir.Call(None, g_var, _invoke_params))
        else:
            with ib.if_scope(tvm.tir.all(syb > last_range, syb <= _range)):
                ib.emit(tvm.tir.Call(None, g_var, _invoke_params))
        last_range = _range
    with ib.if_scope(syb > last_range):
        ib.emit(tvm.tir.Call(None, g_var, _invoke_params))
    stmt = ib.get()
    dispatch_func = tvm.tir.PrimFunc(params, stmt, ret_type, buffer_map, attrs).with_attrs({
        "tir.is_global_func": True,
        "global_symbol": global_symbol
    })
    return dispatch_func


def create_dispatch_mod(g_var: str, original_func: tir.PrimFunc,
                        specialized_funcs: List[tir.PrimFunc], function_symbols) -> IRModule:
    dispatch_mod: IRModule = tvm.IRModule()
    g_var_supply = GlobalVarSupply(dispatch_mod)
    refactored_funcs = []
    for f_var, func in zip(function_symbols, specialized_funcs):
        params, buffers_to_declare = collect_buffers_to_declare(func)
        global_symbol, device_func = refactor_specialized_func(f_var, func, params,
                                                               buffers_to_declare)
        global_symbol = g_var_supply.fresh_global(global_symbol, add_prefix=False)
        dispatch_mod[global_symbol] = device_func
        refactored_funcs.append((global_symbol, device_func))
    dispatch_func = create_dispatch_func(g_var, original_func, refactored_funcs=refactored_funcs)
    dispatch_mod.update(tvm.IRModule.from_expr(dispatch_func))
    return dispatch_mod


def fast_tune_with_dynamic_range_tir(
    func: tir.PrimFunc,
    target: tvm.target.Target,
    topk: int = 10,
    parallel_build: bool = True,
    global_symbol: Optional[str] = None,
    dynamic_range: Optional[Dict[str, List[int]]] = None,
    kernel_name_generator: Optional[Callable] = None,
) -> IRModule:
    if dynamic_range is None:
        dynamic_range = {}
    if target.kind.name != "cuda":
        logger.error("Only support CUDA target")
        return None
    if not global_symbol:
        global_symbol = func.attrs["global_symbol"]

    # set opt_shapes for the primfunc with dynamic symbolic
    opt_shapes: Dict[str, List[int]] = {}
    for buffer in func.buffer_map.values():
        for axis in buffer.shape:
            if isinstance(axis, tvm.tir.Var):
                if axis.name in dynamic_range:
                    opt_shapes[axis.name] = dynamic_range[axis.name]
                else:
                    raise ValueError(f"[BitBLAS] The axis {axis.name} is not in dynamic_range")
    func = func.with_attr("opt_shapes", opt_shapes)

    if "opt_shapes" not in func.attrs:
        logger.error(
            "[BitBLAS] The primfunc has no opt_shapes, please set opt_shapes for the primfunc")
        return None
    else:
        # should be list value
        if not all([isinstance(v, tvm.ir.Array) for v in func.attrs["opt_shapes"].values()]):
            logger.error("The opt_shapes should be list value")
            return None

    logger.info("Start fast tuning with dynamic range")
    opt_shapes = func.attrs["opt_shapes"]

    # Step 1.Calculate the Cartesian product using itertools.product
    product_list = list(itertools.product(*(opt_shapes[key] for key in opt_shapes)))

    # Convert the Cartesian product to a list of dictionaries
    specialize_items: List[Dict] = [dict(zip(opt_shapes.keys(), values)) for values in product_list]

    function_symbols: List[str] = []
    specilized_tuned_funcs: List[tir.PrimFunc] = []
    for item in specialize_items:
        func = func.with_attr("opt_shapes", item)
        _, best = fast_tune(func, target, topk, parallel_build)
        if best is None:
            return None
        specialized_func = best.sch.mod["main"]
        function_symbol = global_symbol
        if kernel_name_generator is not None:
            scheduled_mod = best.sch.mod
            best_hint = best.config
            assert len(scheduled_mod.get_global_vars()) == 1, (
                "The optimized module should only have one global variable for default schedule.")
            assert "main" in scheduled_mod, (
                "The optimized module should have a function named 'main' for default schedule.")
            default_kernal_name = kernel_name_generator.generate(best_hint)
            specialized_func = scheduled_mod["main"].with_attr("global_symbol", default_kernal_name)
            function_symbol = default_kernal_name

        function_symbols.append(function_symbol)
        specilized_tuned_funcs.append(specialized_func)

    assert global_symbol is not None, "The global_symbol should not be None"
    assert len(function_symbols) == len(specilized_tuned_funcs), (
        "The length of global_symbols should be equal to the length of specilized_tuned_funcs")
    return create_dispatch_mod(global_symbol, func, specilized_tuned_funcs, function_symbols)


def fast_tune_with_dynamic_range_tilelang(
    scheduler: BaseScheduler,
    target: tvm.target.Target,
    topk: int = 10,
    parallel_build: bool = True,
    global_symbol: Optional[str] = None,
    dynamic_range: Optional[Dict[str, List[int]]] = None,
    kernel_name_generator: Optional[Callable] = None,
) -> IRModule:
    if dynamic_range is None:
        dynamic_range = {}
    if not global_symbol:
        global_symbol = scheduler.global_symbol

    if target.kind.name != "cuda":
        logger.error("Only support CUDA target")
        return None

    # set opt_shapes for the primfunc with dynamic symbolic
    opt_shapes: Dict[str, List[int]] = {}
    opt_shapes = dynamic_range

    logger.info("Start fast tuning with dynamic range")

    # Step 1.Calculate the Cartesian product using itertools.product
    product_list = list(itertools.product(*(opt_shapes[key] for key in opt_shapes)))
    # Convert the Cartesian product to a list of dictionaries
    specialize_items: List[Dict] = [dict(zip(opt_shapes.keys(), values)) for values in product_list]
    function_symbols: List[str] = []
    specilized_tuned_funcs: List[tir.PrimFunc] = []
    for item in specialize_items:
        # Fast Tune with specialized function
        # Step 1. Send m(dynamic symbolic) -> scheduler(dispatch different scheduler based on input shape)
        # Step 2. Scheduler -> tuning and return the best tile hints
        # Step 3. Apply into a dynamic version (must be aligned with the same scheduler as Step 1)
        # So we should we should have a general scheduler for operators
        # For example, MatmulDispatcher, Conv2DDispatcher, etc.
        # The dispatcher should have a method to dispatch the specialized tilelang template
        # for static shape with default configuration, we handle the dispatch within with default schedule
        # for static shape with customized configuration, we handle the dispatch within with apply config
        # which is similar to what we did at /root/BitBLAS/bitblas/base/utils.py

        # get specialized scheduler
        unit_scheduler = scheduler.set_dynamic_range(dynamic_range=item)
        _, best = fast_tune(unit_scheduler, target, topk, parallel_build)
        if best is None:
            return None
        specialized_func = retrieve_func_from_module(best.sch.mod)
        function_symbol = global_symbol
        if kernel_name_generator is not None:
            scheduled_mod = best.sch.mod
            best_hint = best.config
            default_kernal_name = kernel_name_generator.generate(best_hint)
            prim_func = retrieve_func_from_module(scheduled_mod)
            specialized_func = prim_func.with_attr("global_symbol", default_kernal_name)
            function_symbol = default_kernal_name

        function_symbols.append(function_symbol)
        specilized_tuned_funcs.append(specialized_func)

    assert global_symbol is not None, "The global_symbol should not be None"
    assert len(function_symbols) == len(specilized_tuned_funcs), (
        "The length of global_symbols should be equal to the length of specilized_tuned_funcs")

    default_func = scheduler.with_default_config()  # only for kernel config analysis
    return create_dispatch_mod(global_symbol, default_func, specilized_tuned_funcs,
                               function_symbols)


def fast_tune_with_dynamic_range(
    func_or_scheduler: tir.PrimFunc,
    target: tvm.target.Target,
    topk: int = 10,
    parallel_build: bool = True,
    global_symbol: Optional[str] = None,
    dynamic_range: Optional[Dict[str, List[int]]] = None,
    kernel_name_generator: Optional[Callable] = None,
) -> IRModule:
    if isinstance(func_or_scheduler, tir.PrimFunc):
        return fast_tune_with_dynamic_range_tir(func_or_scheduler, target, topk, parallel_build,
                                                global_symbol, dynamic_range, kernel_name_generator)
    elif isinstance(func_or_scheduler, BaseScheduler):
        return fast_tune_with_dynamic_range_tilelang(func_or_scheduler, target, topk,
                                                     parallel_build, global_symbol, dynamic_range,
                                                     kernel_name_generator)
    else:
        raise ValueError("Not supported type: ", type(func_or_scheduler))
