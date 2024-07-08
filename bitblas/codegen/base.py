# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas.ops.operator import OperatorConfig
from abc import ABC, abstractmethod


class Backend(ABC):
    """
        input: OperatorConfig
        The duty of backend:
            - is OperatorConfig is Available For our Backend
            - Generate CUDA Source for compilation
    """

    def __init__(self, config: OperatorConfig):
        self.config = config

    @abstractmethod
    def compile(self, config):
        pass

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass

    @abstractmethod
    def optimize(self, *args, **kwargs):
        pass
