# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from bitblas.ops import Operator, OperatorConfig
from bitblas import auto_detect_nvidia_target

class BitblasOperatorBenchmarkBase(ABC):
    
    # separate benchmark sets for different operators set
    benchmark_sets: Dict[str, List[Tuple[Operator, OperatorConfig]]] = {}

    # currently we only support nvidia target for benchmarking
    benchmark_target: str = auto_detect_nvidia_target()

    # benchmark results
    benchmark_results: Dict[str, List[Optional[float]]] = {}
    
    @abstractmethod
    def prepare_benchmark_sets(self):
        pass
    
    def add_benchmark_set(self, name:str, benchmark_set:List[Tuple[Operator, OperatorConfig]]):
        if name in self.benchmark_sets:
            self.benchmark_sets[name].extend(benchmark_set)
        else:
            self.benchmark_sets[name] = benchmark_set
    
    def run(self):
        self.prepare_benchmark_sets()
        self.benchmark()
        print("Benchmark results:", self.benchmark_results)
        self.report()
        self.cleanup()

    def report(self):
        return NotImplementedError

    def cleanup(self):
        # clean up the benchmark sets
        self.benchmark_sets.clear()

    def benchmark(self):
        for name, benchmark_set in self.benchmark_sets.items():
            self.benchmark_results[name] = []
            for operator, config in benchmark_set:
                self.benchmark_results[name].append(self.run_benchmark(operator, config))

    def run_benchmark(self, operator:Operator, config:OperatorConfig) -> Optional[float]:
        op_inst = operator(config, target=self.benchmark_target)
        return op_inst.profile_latency() 
    
    @abstractmethod
    def get_operator(self) -> Operator:
        raise NotImplementedError

    @abstractmethod
    def get_operator_config(self) -> OperatorConfig:
        raise NotImplementedError

    def get_benchmark_sets(self, name:Optional[str]=None) -> List[Tuple[Operator, OperatorConfig]]:
        if name is None:
            return self.benchmark_sets
        else:
            assert name in self.benchmark_sets, f"Operator {name} not found in benchmark sets"
            return self.benchmark_sets[name]
