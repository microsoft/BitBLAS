# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas.ops.operator import OperatorConfig, Operator


class OperatorCache(object):
    """
    A cache manager for operator instances, such as Matmul, Convolution, etc.,
    keyed by their configuration objects. Supports adding, retrieving, and
    checking the existence of operator instances based on their unique configurations.
    """

    def __init__(self):
        """
        Initializes the cache.
        """
        self.cache = {}

    def add(self, config: OperatorConfig, op_inst: Operator):
        """
        Adds an operator instance to the cache with the given configuration.

        Parameters:
        - config: A hashable configuration object that uniquely identifies the operator instance.
        - op_inst: The instance of the operator to cache.
        """
        self.cache[config] = op_inst

    def get(self, config: OperatorConfig):
        """
        Retrieves an operator instance from the cache based on the given configuration.

        Parameters:
        - config: The configuration object that uniquely identifies the operator instance.

        Returns:
        The cached operator instance if present; otherwise, None.
        """
        return self.cache.get(config, None)

    def exists(self, config):
        """
        Checks if an operator instance with the given configuration exists in the cache.

        Parameters:
        - config: The configuration object that uniquely identifies the operator instance.

        Returns:
        True if the instance exists in the cache; otherwise, False.
        """
        return config in self.cache

global_operator_cache = OperatorCache()
