# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas import tvm
import os
from tvm.contrib.popen_pool import PopenPoolExecutor, StatusKind
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from typing import List, Tuple, Optional, Union, Literal
from tvm import tir, IRModule
from tvm.runtime import Module
from tvm.tir import Schedule
from tvm.relax.expr import Function
import bitblas
from .analysis import get_root_block, get_reduction_blocks
from bitblas.base.arch import TileDevice
from bitblas.base.roller.policy import TensorCorePolicy, DefaultPolicy
from bitblas.base.roller.hint import Hint
from bitblas.gpu.matmul_analysis import get_tensorized_func_and_tags
from bitblas.common import MAX_ERROR_MESSAGE_LENGTH
import tempfile
from bitblas.utils import (
    tensor_replace_dp4a,
    tensor_remove_make_int4,
    tensor_remove_make_int2,
    retrieve_func_from_module,
)
from bitblas.utils.tensor_adapter import (
    np_float2np_bf16,)
import logging

logger = logging.getLogger(__name__)


def get_rasterization_code(pannel_width: int = 8) -> str:
    return f"""
        const int MAX_BLOCK_N = {pannel_width};
        const auto baseBlockIdx = blockIdx.x + gridDim.x *blockIdx.y;
        const auto totalPanel = (gridDim.x * gridDim.y +MAX_BLOCK_N * gridDim.x - 1) / (MAX_BLOCK_N * gridDim.x);
        const auto totalBlock = gridDim.x * gridDim.y;
        const auto panelIdx = baseBlockIdx / (MAX_BLOCK_N *gridDim.x);
        const auto strideLd = panelIdx + 1 < totalPanel ?MAX_BLOCK_N : (totalBlock - panelIdx * (MAX_BLOCK_N *gridDim.x)) / gridDim.x;
        const auto bx = (panelIdx & 1) ? gridDim.x -(baseBlockIdx - panelIdx * MAX_BLOCK_N * gridDim.x) /strideLd - 1 : (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) / strideLd;
        const auto by = (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) % strideLd + panelIdx * MAX_BLOCK_N;
        const auto bz = blockIdx.z;
        const dim3 blockIdx(bx, by, bz);
    """


class CompileResult:
    """
    Class to store the result of compilation
    """

    def __init__(self, config, sch, mod: Module):
        self.config = config
        self.sch = sch
        self.mod = mod
        self.code = mod.imported_modules[0].get_source() if mod else None
        self.latency = 1e9
        self.time_evaluator = None

    def profile(self, data_distribution="uniform"):
        func = retrieve_func_from_module(self.sch.mod)
        device = self.config.arch.device
        profile_tensors = get_dummy_input_arrays(func, device, distribution=data_distribution)
        latency = self.time_evaluator(*profile_tensors).mean * 1e3
        return latency


def get_roller_hints_from_func(func_or_module: Union[tir.PrimFunc, IRModule],
                               arch: TileDevice,
                               topk: int = 10,
                               tensorcore_only: bool = False,
                               allow_gemv: bool = False) -> Optional[List[Hint]]:
    func = None
    if isinstance(func_or_module, tir.PrimFunc):
        func = func_or_module
    elif isinstance(func_or_module, IRModule):
        func = retrieve_func_from_module(func_or_module)
    else:
        raise ValueError("Not supported type: ", type(func_or_module))

    assert func is not None, "The function should not be None"

    if tensorcore_only:
        try:
            tensorized_func, tags = get_tensorized_func_and_tags(
                func, arch.target, allow_gemv=allow_gemv)
        except Exception as e_msg:
            logger.debug("Get tensorized func and tags failed: ", e_msg)
            tags = None
        if tags and tensorized_func:
            policy = TensorCorePolicy(func=tensorized_func, arch=arch, tags=tags)
            return policy.emit_config(topk)
        else:
            return None
    else:
        policy = DefaultPolicy(func=func, arch=arch)
        tensorized_func = None
        try:
            tensorized_func, tags = get_tensorized_func_and_tags(
                func, arch.target, allow_gemv=allow_gemv)
        except Exception as e_msg:
            logger.debug("Get tensorized func and tags failed: ", e_msg)
            tags = None
        if tags and tensorized_func:
            policy = TensorCorePolicy(func=tensorized_func, arch=arch, tags=tags)
        return policy.emit_config(topk)


def _apply_config(
        func: tir.PrimFunc,
        config=None,  # todo(lei): update typing
) -> Optional[tir.Schedule]:
    """
    find rules:
    case 1. if the main block has no reduce op, then use the Elementwise rule.
    case 2. if the config enabled tensorcore, then use the TensorCore rule.
    case 3. if any([t > 1 for t in config.reduce_thread]), we should use the InnerThread Reduction Rule.
    case 4. else we should use general reduction rule.
    """
    logger.debug("Apply config {}".format(config))

    sch = tir.Schedule(func)
    root_block = get_root_block(sch)
    blocks = sch.get_child_blocks(root_block)
    reduction_blocks = get_reduction_blocks(sch, blocks)

    if not reduction_blocks:
        return bitblas.gpu.ElementWise().apply_config(func, config)
    elif config.use_tc:
        if config.arch.sm_version >= 80:
            # For A100(sm_80) or more advanced gpu, use MMA tensorization.
            return bitblas.gpu.MatmulTensorizationMMA().apply_config(func, config)
        else:
            # For other GPUs, use WMMA tensorization.
            return bitblas.gpu.MatmulTensorizationWMMA().apply_config(func, config)
    else:
        _reduction_rules = []

        _reduction_rules.append(bitblas.gpu.GEMV())
        if not any([t > 1 for t in config.reduce_thread]):
            # Matrix multiplication template doesn't support inner thread reduction
            _reduction_rules.append(bitblas.gpu.Matmul())
        _reduction_rules.append(bitblas.gpu.GeneralReduction())

        for rule in _reduction_rules:
            sch = rule.apply_config(func, config)
            try:
                sch = rule.apply_config(func, config)
            except Exception as e_msg:
                logger.debug("Apply config failed: ", e_msg)
                continue
            if sch is not None:
                return sch
    return None


def get_dummy_input_arrays(
    func: Union[tir.PrimFunc, Function],
    device: tvm.runtime.Device,
    distribution: Literal["uniform", "onefill"] = "uniform",
):

    def var_wrapper(v):
        if isinstance(v, tvm.tir.Var):
            assert "opt_shapes" in func.attrs
            assert v.name in func.attrs["opt_shapes"]
            return func.attrs["opt_shapes"][v.name].value
        elif isinstance(v, tvm.tir.IntImm):
            return v.value
        else:
            raise RuntimeError("Not supported type: ", type(v))

    profile_tensors = []
    for param in func.params:
        if isinstance(func, tir.PrimFunc):
            if param not in func.buffer_map:
                # in case of dynamic symbolic may in params
                continue
            arg = func.buffer_map[param]
        elif isinstance(func, Function):
            arg = param.struct_info
        else:
            raise ValueError("Not supported type: ", type(func))

        def map_numpy_type(intype):
            typemap = {
                'e4m3_float8': 'float8_e4m3fn',
                'e5m2_float8': 'float8_e5m2',
            }
            if intype in typemap:
                return typemap[intype]
            else:
                return intype

        numpy_dtype = map_numpy_type(arg.dtype)
        if distribution == "uniform":
            data_np = np.random.rand(*[var_wrapper(i) for i in arg.shape])
            if arg.dtype == "bfloat16":
                profile_tensors.append(
                    tvm.nd.empty(data_np.shape, device=device, dtype=arg.dtype).copyfrom(
                        np_float2np_bf16(data_np.astype(np.float32))))
            else:
                profile_tensors.append(tvm.nd.array(data_np.astype(numpy_dtype), device=device))
        elif distribution == "onefill":
            data_np = np.ones(*[var_wrapper(i) for i in arg.shape])
            if arg.dtype == "bfloat16":
                profile_tensors.append(
                    tvm.nd.empty(data_np.shape, device=device,
                                 dtype=arg.dtype).copyfrom(np_float2np_bf16(data_np)))
            else:
                profile_tensors.append(tvm.nd.array(data_np.astype(numpy_dtype), device=device))
        else:
            raise ValueError("Not supported distribution: ", distribution)
    return profile_tensors


def apply_and_build_parallel(func,
                             configs,
                             arch,
                             num_repeats=3,
                             max_workers=10,
                             timeout=60,
                             data_distribution="uniform") -> CompileResult:
    cpresults = []

    max_workers = min(len(configs), os.cpu_count(), max_workers)

    # apply config in thread parallel
    _sched: List[Schedule] = []

    def _apply_schedule(f, c):
        try:
            sch = _apply_config(f, c)
        except Exception as apply_schedule_error:
            logger.debug("Apply schedule failed: {}".format(apply_schedule_error))
            sch = None
        return sch

    with ThreadPoolExecutor(max_workers=max_workers) as scheduler:
        futures = {scheduler.submit(_apply_schedule, func, config) for config in configs}
        for future in as_completed(futures, timeout=timeout):
            _sched.append(future.result())

    builder = PopenPoolExecutor(max_workers=max_workers, timeout=timeout)

    # build in process parallel
    def _build(context) -> str:
        idx, mod, arch = context
        if mod is None:
            return idx, None, None
        # TODO(lei):
        # this is a trick to implement rasteration, will be removed in the future
        config = configs[idx]

        @tvm.register_func(func_name="tvm_callback_cuda_postproc", override=True)
        def tvm_callback_cuda_postproc(code, _):
            code = tensor_replace_dp4a(code)
            code = tensor_remove_make_int4(code)
            code = tensor_remove_make_int2(code)
            return code

        with tvm.transform.PassContext(config={
                "tir.use_async_copy": True,
                "tir.disable_cse_tir": True,
                **config.pass_context
        }):
            rt_mod = tvm.build(mod, target=arch.target)

        from tvm.contrib.tar import tar  # pylint: disable=import-outside-toplevel

        artifact_path = os.path.join(tempfile.mkdtemp(), "tvm_tmp_mod." + tar.output_format)
        code = rt_mod.imported_modules[0].get_source()
        rt_mod.export_library(artifact_path, fcompile=tar)
        return idx, code, artifact_path

    _mods = [sch.mod if sch is not None else None for sch in _sched]

    for map_result in builder.map_with_error_catching(
            _build,
        [(i, mod, arch) for i, mod in enumerate(_mods)],
    ):
        if map_result.status == StatusKind.TIMEOUT:
            logger.debug("LocalBuilder: Timeout")
        elif map_result.status == StatusKind.EXCEPTION:
            local_build_error = str(map_result.value)
            if len(local_build_error) > MAX_ERROR_MESSAGE_LENGTH:
                local_build_error = (
                    local_build_error[:MAX_ERROR_MESSAGE_LENGTH // 2] + "\t...\t" +
                    local_build_error[-MAX_ERROR_MESSAGE_LENGTH // 2:])
            logger.debug("LocalBuilder: An exception occurred {}".format(local_build_error))
            continue
        elif map_result.status == StatusKind.COMPLETE:
            idx, code, artifact_path = map_result.value
            sch = _sched[idx]
            config = configs[idx]
            if artifact_path is None:
                ARTIFACT_NOT_FOUND = f"Apply config {config} failed, artifact path is None"
                logger.debug(ARTIFACT_NOT_FOUND)
                continue
            rt_mod = tvm.runtime.load_module(artifact_path)
            cpresult = CompileResult(config, sch, rt_mod)
            timer_cuda_mod = rt_mod.time_evaluator(
                rt_mod.entry_name, arch.device, number=num_repeats)
            cpresult.time_evaluator = timer_cuda_mod
            cpresult.code = code
            cpresults.append(cpresult)
        else:
            raise ValueError(f"Unreachable: unexpected result: {map_result}")

    del builder

    best = None
    best_latency = 1e9
    for cpresult in cpresults:
        config = cpresult.config
        try:
            latency = cpresult.profile(data_distribution=data_distribution)
        except Exception as e_mesg:
            logger.debug(f"Evaluation with config failed {e_mesg}")
            continue
        logger.info("Evaluation with config {}".format(config))
        logger.info("Time cost of this config: {:.3f} ms".format(latency))

        cpresult.latency = latency
        if latency < best_latency:
            best_latency = latency
            best = cpresult

    return cpresults, best


def apply_and_build(
    func,
    configs,
    arch,
    parallel_build=False,
    data_distribution="uniform",
) -> Tuple[List[CompileResult], CompileResult]:
    max_workers = 10 if parallel_build else 1
    return apply_and_build_parallel(
        func, configs, arch, max_workers=max_workers, data_distribution=data_distribution)
