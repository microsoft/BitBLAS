# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Apply ScheduleRules onto an IRModule to generate default schedules without tuning,
or a space for MetaSchedule tuning
"""
from typing import List, Optional, Dict
import os
import shutil
import tempfile
import os.path as osp
import tvm
from tvm import tir
from tvm import meta_schedule as ms
from tvm.ir import IRModule
from tvm.ir.transform import PassContext, module_pass
from tvm.target import Target
from .roller.policy import DefaultPolicy, TensorCorePolicy
from .roller.arch import CUDA
from .schedule_rule import ScheduleRule
from ..gpu.matmul_analysis import get_tensorized_func_and_tags
from ..base.analysis import check_func_with_dynamic
from .utils import apply_and_build, fast_tune, fast_tune_with_dynamic_range


def _is_scheduled(func: tir.PrimFunc) -> bool:
    if not isinstance(func, tir.PrimFunc):
        return False
    if not func.attrs:
        return False
    if "tir.is_scheduled" not in func.attrs:
        return False
    return func.attrs["tir.is_scheduled"] == 1


@module_pass(opt_level=0, name="ApplyDefaultSchedule")
class ApplyDefaultSchedule:  # pylint: disable=too-few-public-methods
    """A IRModule pass that applies a list of ScheduleRules to all PrimFuncs in the module."""

    def __init__(self, *rules: ScheduleRule):
        """Construct a new ApplyDefaultSchedule pass.

        Parameters
        ----------
        *rules : ScheduleRule
            The ScheduleRules to apply to all PrimFuncs in the module.
        """
        self.rules = list(rules)

    def transform_module(  # pylint: disable=missing-function-docstring
        self,
        mod: IRModule,
        _: PassContext,
    ) -> IRModule:
        target = Target.current(allow_none=False)

        updated_functions = {}
        for g_var, func in mod.functions_items():
            if isinstance(func, tir.PrimFunc) and not _is_scheduled(func):
                sch = _apply_rules(func, target, self.rules, tunable=False)
                if sch is not None:
                    assert len(sch) == 1
                    updated_functions[g_var] = sch[0].mod["main"].with_attr("tir.is_scheduled", 1)
        for g_var, func in updated_functions.items():
            mod[g_var] = func
        return mod


@module_pass(opt_level=0, name="ApplyFastTuning")
class ApplyFastTuning:  # pylint: disable=too-few-public-methods
    """A IRModule pass that applies a list of ScheduleRules to all PrimFuncs in the module."""

    def __init__(
        self,
        topk: int = 10,
        target: Optional[Target] = None,
        parallel_build: bool = True,
        meta_database_dir: str = None,
        whitelist: List[str] = [],
        dynamic_range: Dict[str, List[int]] = {},
    ):
        """Construct a new ApplyFastTuning pass.

        Parameters
        ----------
        meta_database : str
            The path of database.
        dynamic_range : Dict[str, List[int]]
            Use for generate kernel based on dynamic range.
        """
        self.topk = topk
        self.target = Target.current() if target is None else target
        self.parallel_build = parallel_build
        self.meta_database_dir = meta_database_dir
        self.whitelist = whitelist
        self.dynamic_range = dynamic_range
        self.temp_dir = tempfile.TemporaryDirectory()
        print(f"[BitBLAS] Using meta database dir {self.temp_dir}")
        path_workload = osp.join(self.temp_dir.name, "database_workload.json")
        path_tuning_record = osp.join(self.temp_dir.name, "database_tuning_record.json")
        self.cache_meta_database = ms.database.JSONDatabase(
            path_workload, path_tuning_record, module_equality="structural"
        )
    
    def _in_white_list(self, func_name: str) -> bool:
        if len(self.whitelist) == 0:
            return True
        for name in self.whitelist:
            if name in func_name:
                return True
        return False

    def transform_module(  # pylint: disable=missing-function-docstring
        self,
        mod: IRModule,
        _: PassContext,
    ) -> IRModule:
        target = self.target
        updated_functions = {}

        for g_var, func in mod.functions_items():
            if isinstance(func, tir.PrimFunc) and not _is_scheduled(func):
                if not self._in_white_list(g_var.name_hint):
                    continue
                print(f"[BitBLAS] Start to apply fast tuning for {g_var}")
                normalize_mod_func_ = tvm._ffi.get_global_func("tvm.meta_schedule.normalize_mod")
                _normalized_func_mod = normalize_mod_func_(func)

                if self.cache_meta_database.has_workload(_normalized_func_mod):
                    tuning_record = self.cache_meta_database.query_tuning_record(
                        _normalized_func_mod,
                        target,
                        g_var.name_hint,
                    )
                    if tuning_record:
                        trace = tuning_record.trace
                        sch = tvm.tir.Schedule(func)
                        trace.apply_to_schedule(sch, remove_postproc=False)
                        print(f"[BitBLAS] Find Cache for {g_var}")
                        updated_functions[g_var] = sch.mod["main"].with_attr("tir.is_scheduled", 1)
                        continue
                
                if check_func_with_dynamic(func):

                    dispatch_mod = fast_tune_with_dynamic_range(
                        func,
                        target=target,
                        topk=self.topk,
                        parallel_build=self.parallel_build,
                        global_symbol=g_var.name_hint,
                        dynamic_range=self.dynamic_range,
                    )
  
                    if dispatch_mod:
                        for g, f in dispatch_mod.functions_items():
                            if g.name_hint == g_var.name_hint:
                                # avoid duplicated global symbol
                                updated_functions[g_var] = f.without_attr("global_symbol").with_attr("tir.is_scheduled", 1)
                            else:
                                updated_functions[g] = f.with_attr("tir.is_scheduled", 1)
                        # cannot reuse meta database as it canot be recorvered from the trace
                        workload = self.cache_meta_database.commit_workload(_normalized_func_mod)
                else:
                    # otherwise is static shape analysis
                    _, best = fast_tune(
                        func, target=target, topk=self.topk, parallel_build=self.parallel_build
                    )

                    if best is not None:
                        updated_functions[g_var] = best.sch.mod["main"].with_attr("tir.is_scheduled", 1)
                        workload = self.cache_meta_database.commit_workload(_normalized_func_mod)
                        # only record the best schedule
                        self.cache_meta_database.commit_tuning_record(
                            ms.database.TuningRecord(
                                best.sch.trace,
                                workload,
                                [best.latency],
                                target,
                                ms.arg_info.ArgInfo.from_prim_func(func=best.sch.mod["main"]),
                            )
                        )

        for g_var, func in updated_functions.items():
            mod[g_var] = func

        # copy database
        if self.meta_database_dir is not None:
            if not osp.exists(self.meta_database_dir):
                os.makedirs(self.meta_database_dir)
            # TODO(lei): maybe another way to copy the database
            shutil.copytree(self.temp_dir.name, self.meta_database_dir, dirs_exist_ok=True)

        return mod

    def __del__(self):
        # clean up the temp cache
        self.temp_dir.cleanup()


def _apply_rules(
    func: tir.PrimFunc,
    target: Target,
    rules: List[ScheduleRule],
    tunable: bool,
) -> Optional[List[tir.Schedule]]:
    for rule in rules:
        space = rule.apply(func, target, tunable)
        if space is None:
            continue
        if isinstance(space, tir.Schedule):
            space = [space]
        return space
    return None
