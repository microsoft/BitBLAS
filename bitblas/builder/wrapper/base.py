# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from abc import ABC, abstractmethod

PREDEF_ARRTIBUTE_SET_DYNAMIC_MEMORY = """
    cudaFuncSetAttribute({}, cudaFuncAttributeMaxDynamicSharedMemorySize, {});
"""

PREDEF_INIT_FUNC = """
extern "C" void init() {{
    {}
}}
"""

PREDEF_HOST_FUNC = """
extern "C" void call({}) {{
{}
}}
"""


class BaseWrapper(ABC):

    @abstractmethod
    def wrap(self, *args, **kwargs):
        raise NotImplementedError
