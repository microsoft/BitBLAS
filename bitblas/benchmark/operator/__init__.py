# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from os import path, makedirs
from time import perf_counter
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from bitblas.ops import Operator, OperatorConfig
from bitblas.utils import get_default_cache_path
from bitblas import auto_detect_nvidia_target
from bitblas import tvm as tvm
from bitblas.cache import OperatorCache
import logging

logger = logging.getLogger(__name__)


class BitblasOperatorBenchmarkBase(ABC):
    # Separate benchmark sets for different operators, where the last key represents
    # the dynamic profing shape
    benchmark_sets: Dict[str, List[Tuple[Operator, OperatorConfig, Optional[int]]]] = {}

    # Currently we only support NVIDIA target for benchmarking
    benchmark_target: str = auto_detect_nvidia_target()

    # Benchmark results: a list of tuples, each containing latency and tuning time
    benchmark_results: Dict[str, List[Tuple[Optional[float], Optional[float]]]] = {}

    # Enable hardware-aware tuning
    enable_hardware_aware_tuning: bool = False

    # Log path
    log_path: Optional[str] = path.join(get_default_cache_path(), "benchmark")

    # Operator cache
    operator_cache: OperatorCache = OperatorCache()

    @abstractmethod
    def prepare_benchmark_sets(self):
        pass

    def generate_op_unit(
        self,
        config: OperatorConfig,
        dynamic_profiling_shape: Optional[Dict[str, int]] = None,
    ) -> Tuple[Operator, OperatorConfig, Optional[Dict[str, int]]]:
        """Generate a benchmark element for an operator."""
        return self.get_operator(), config, dynamic_profiling_shape

    def add_benchmark_set(
        self,
        name: str,
        benchmark_set: List[Tuple[Operator, OperatorConfig, Optional[Dict[str, int]]]],
    ):
        """Add a benchmark set to the collection."""
        if name in self.benchmark_sets:
            self.benchmark_sets[name].extend(benchmark_set)
        else:
            self.benchmark_sets[name] = benchmark_set

    def run(self, report=True, serialize=True, enable_tuning: bool = False):
        """Run the benchmark process."""

        if not path.exists(self.log_path):
            makedirs(self.log_path)

        if enable_tuning:
            self.enable_tuning()

        self.prepare_benchmark_sets()
        self.benchmark()

        if report:
            self.report()

        if serialize:
            self.serialize_results()

        self.cleanup()

    @abstractmethod
    def report(self):
        """Generate a report of the benchmark results."""
        raise NotImplementedError

    def cleanup(self):
        """Clean up the benchmark sets."""
        self.benchmark_sets.clear()

    def benchmark(self):
        """Run benchmarks on all benchmark sets."""
        for name, benchmark_set in self.benchmark_sets.items():
            self.benchmark_results[name] = [
                self.run_benchmark(op, config, opt) for op, config, opt in benchmark_set
            ]

    def make_operator(self, operator: Operator, config: OperatorConfig) -> Operator:
        """Make an operator instance."""
        return operator(config, target=self.benchmark_target)

    def run_benchmark(
        self,
        operator: Operator,
        config: OperatorConfig,
        dynamic_profiling_shape: Optional[Dict[str, int]] = None,
    ) -> Optional[float]:
        """Run a single benchmark."""

        if self.operator_cache.exists(config):
            logger.info(f"Operator {config} found in cache")
            op_inst = self.operator_cache.get(config)
            latency = op_inst.profile_latency(dynamic_symbolic_constraints=dynamic_profiling_shape)
            op_inst.cleanup()
            return latency, None

        op_inst = self.make_operator(operator, config)
        tuning_time = None

        if self.enable_hardware_aware_tuning:
            start = perf_counter()
            op_inst.hardware_aware_finetune(topk=20, parallel_build=True)
            tuning_time = perf_counter() - start

        self.operator_cache.add(config, op_inst)

        latency = op_inst.profile_latency(dynamic_symbolic_constraints=dynamic_profiling_shape)

        op_inst.cleanup()

        return latency, tuning_time

    @abstractmethod
    def get_operator(self) -> Operator:
        """Get the operator to be benchmarked."""
        raise NotImplementedError

    @abstractmethod
    def get_operator_config(self) -> OperatorConfig:
        """Get the configuration for the operator."""
        raise NotImplementedError

    def get_benchmark_sets(self,
                           name: Optional[str] = None) -> List[Tuple[Operator, OperatorConfig]]:
        """Retrieve benchmark sets by name, or all if name is None."""
        if name is None:
            return self.benchmark_sets
        else:
            assert (name in self.benchmark_sets), f"Operator {name} not found in benchmark sets"
            return self.benchmark_sets[name]

    @abstractmethod
    def serialize_results(self) -> None:
        """Serialize the benchmark results."""
        pass

    def enable_tuning(self):
        """Enable hardware-aware tuning."""
        self.enable_hardware_aware_tuning = True

    def disable_tuning(self):
        """Disable hardware-aware tuning."""
        self.enable_hardware_aware_tuning = False

    def set_log_path(self, log_path: str):
        """Set the log path."""
        self.log_path = log_path

    def set_benchmark_target(self, target: str):
        """Set the benchmark target."""
        self.benchmark_target = target

    def set_benchmark_results(self, results: Dict[str, List[Tuple[Optional[float],
                                                                  Optional[float]]]]):
        """Set the benchmark results."""
        self.benchmark_results = results
