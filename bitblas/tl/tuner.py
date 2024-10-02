# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas import tvm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Literal
from tvm import tir, IRModule
from tvm.runtime import Module
from tvm.tir import Schedule
import tvm.tl as tl
from bitblas.ops.base_scheduler import BaseScheduler
from bitblas.base.arch import CUDA
from bitblas.base.utils import get_dummy_input_arrays
from bitblas.base.roller.policy import TensorCorePolicy, DefaultPolicy
from bitblas.gpu.matmul_analysis import get_tensorized_func_and_tags
from bitblas.utils import tensor_replace_dp4a, tensor_remove_make_int4, tensor_remove_make_int2
from bitblas.common import MAX_ERROR_MESSAGE_LENGTH

import logging
import tempfile

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
        func = self.sch.mod["main"]
        device = self.config.arch.device
        profile_tensors = get_dummy_input_arrays(func, device, distribution=data_distribution)
        latency = self.time_evaluator(*profile_tensors).mean * 1e3
        return latency


def _apply_config(
    scheduler: BaseScheduler,
    config=None,
) -> Optional[IRModule]:
    """
    find rules:
    case 1. if the main block has no reduce op, then use the Elementwise rule.
    case 2. if the config enabled tensorcore, then use the TensorCore rule.
    case 3. if any([t > 1 for t in config.reduce_thread]), we should use the InnerThread Reduction Rule.
    case 4. else we should use general reduction rule.
    """
    logger.debug("Scheduler Apply config {}".format(config))
    scheduled_func = scheduler.apply_config(**config.get_config_params())
    if scheduled_func is None:
        return None
    else:
        return tvm.IRModule.from_expr(scheduled_func)


def apply_and_build_parallel(scheduler,
                             configs,
                             arch,
                             num_repeats=3,
                             max_workers=10,
                             timeout=30,
                             data_distribution="uniform") -> CompileResult:
    cpresults = []

    max_workers = min(len(configs), os.cpu_count(), max_workers)

    # apply config in thread parallel
    _scheduled_ir_modules: List[Schedule] = []

    def _submit_config(f, c):
        try:
            scheduled_ir_module = _apply_config(f, c)
        except Exception as apply_schedule_error:
            logger.debug("Apply schedule failed: {}".format(apply_schedule_error))
            scheduled_ir_module = None
        return scheduled_ir_module

    with ThreadPoolExecutor(max_workers=max_workers) as _scheduler:
        futures = {_scheduler.submit(_submit_config, scheduler, config) for config in configs}
        for future in as_completed(futures, timeout=timeout):
            _scheduled_ir_modules.append(future.result())

    # build in process parallel
    def _build(context):
        idx, mod, arch = context
        if mod is None:
            return idx, None, None

        config = configs[idx]
        assert config is not None

        @tvm.register_func(func_name="tvm_callback_cuda_postproc", override=True)
        def tvm_callback_cuda_postproc(code, _):
            code = tensor_replace_dp4a(code)
            code = tensor_remove_make_int4(code)
            code = tensor_remove_make_int2(code)
            return code

        # Check only have one function in the module
        if len(mod.functions) > 1:
            raise ValueError("Only support one function in the module")

        tl_prim_func = list(mod.functions.values())[0]

        with tvm.transform.PassContext(
                config={
                    "tir.use_async_copy": True,
                    "tir.disable_cse_tir": True,
                    **(config.pass_context if config.pass_context else {})
                }):
            rt_mod = tl.lower(tl_prim_func, arch.target, runtime_only=True)

        from tvm.contrib.tar import tar  # Import the tar module

        artifact_path = os.path.join(tempfile.mkdtemp(), "tvm_tmp_mod." + tar.output_format)
        code = rt_mod.imported_modules[0].get_source()
        rt_mod.export_library(artifact_path, fcompile=tar)
        return idx, code, artifact_path

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_build, (i, mod, arch)): i
            for i, mod in enumerate(_scheduled_ir_modules)
        }

        for future in as_completed(future_to_idx, timeout=timeout):
            idx = future_to_idx[future]
            try:
                idx, code, artifact_path = future.result()
                ir_module = _scheduled_ir_modules[idx]
                sch = tvm.tir.Schedule(ir_module)
                config = configs[idx]

                if artifact_path is None:
                    ARTIFACT_NOT_FOUND = f"Apply config {config} failed, artifact path is None"
                    print(ARTIFACT_NOT_FOUND)
                    continue

                rt_mod = tvm.runtime.load_module(artifact_path)
                cpresult = CompileResult(config, sch, rt_mod)
                timer_cuda_mod = rt_mod.time_evaluator(
                    rt_mod.entry_name, arch.device, number=num_repeats)
                cpresult.time_evaluator = timer_cuda_mod
                cpresult.code = code
                cpresults.append(cpresult)

            except Exception as e:
                local_build_error = str(e)
                if len(local_build_error) > MAX_ERROR_MESSAGE_LENGTH:
                    local_build_error = (
                        local_build_error[:MAX_ERROR_MESSAGE_LENGTH] + "\t...\t" +
                        local_build_error[-MAX_ERROR_MESSAGE_LENGTH:])
                print(f"An exception occurred for index {idx}: {local_build_error}")

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
    scheduler,
    configs,
    arch,
    parallel_build=False,
    data_distribution="uniform",
) -> Tuple[List[CompileResult], CompileResult]:
    max_workers = 10 if parallel_build else 1
    return apply_and_build_parallel(
        scheduler, configs, arch, max_workers=max_workers, data_distribution=data_distribution)


def fast_tune(
    func: tir.PrimFunc,
    target: tvm.target.Target,
    topk: int = 10,
    parallel_build: bool = True,
    data_distribution: Literal["uniform", "onefill"] = "uniform",
):
    # check the function is a primfunc
    if not isinstance(func, tir.PrimFunc):
        raise ValueError("Only support func is PrimFunc")  # pragma: no cover

    if target.kind.name != "cuda":
        logger.error("Only support CUDA target")
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
                if isinstance(axis, tvm.tir.Var) and axis.name not in opt_shapes:
                    raise NotImplementedError(
                        "Currently do not support fast tune with none-dynamic range set")
        if opt_shapes:
            raise NotImplementedError(
                "Currently do not support fast tune with none-dynamic range set")

    arch = CUDA(target)

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

    cpresults, best = apply_and_build(
        func,
        configs,
        arch,
        parallel_build=parallel_build,
        data_distribution=data_distribution,
    )

    return cpresults, best
