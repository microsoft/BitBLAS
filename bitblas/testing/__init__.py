# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys
import inspect
import pytest
from bitblas.base import DefaultPolicy, TensorCorePolicy
from bitblas.gpu.matmul_analysis import get_tensorized_func_and_tags


# pytest.main() wrapper to allow running single test file
def main():
    test_file = inspect.getsourcefile(sys._getframe(1))
    sys.exit(pytest.main([test_file] + sys.argv[1:]))


def debug_with_schedule(func, arch, sch_rule):
    policy = DefaultPolicy(func=func, arch=arch)
    try:
        tensorized_func, tags = get_tensorized_func_and_tags(func, arch.target)
    except Exception:
        tags = None
    if tags:
        policy = TensorCorePolicy(func=tensorized_func, arch=arch, tags=tags)
    configs = policy.emit_config(1)
    return sch_rule.apply_config(func, configs[0])
