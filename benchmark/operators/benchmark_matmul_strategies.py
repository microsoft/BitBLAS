# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas.benchmark import BitblasOperatorBenchmarkBase
from bitblas import Matmul, MatmulConfig
from bitblas.ops.general_matmul import OptimizeStrategy
from bitblas.utils import get_commit_id
from bitblas import set_log_level
from tabulate import tabulate
import json
from os import path, makedirs
from typing import Dict, List, Union
import argparse

set_log_level("DEBUG")


class BitblasMatmulOpsBenchmarkCompareStategies(BitblasOperatorBenchmarkBase):

    BENCHMARK_RESULTS_FILE = "benchmark_results.json"
    BENCHMARK_SHAPES_FILE = "benchmark_shapes.json"
    BENCHMARK_DEVICE_FILE = "benchmark_device.json"

    config_map = {
        "FP16xUINT4_ACCFP16_NT_STRATEGY_GEMV": {
            "A_dtype": "float16",
            "W_dtype": "uint4",
            "accum_dtype": "float16",
            "optimize_stratety": OptimizeStrategy.SingleBatchDecodeOnly,
        },
        "FP16xUINT4_ACCFP16_NT_STRATEGY_ContigiousBatching": {
            "A_dtype": "float16",
            "W_dtype": "uint4",
            "accum_dtype": "float16",
            "optimize_stratety": OptimizeStrategy.ContigousBatching,
        },
    }

    OPT_SHAPES = [1, 16, 32, 64, 128, 256, 512, 4096]
    CURRENT_COMMIT_ID = get_commit_id()

    def __init__(self):
        super().__init__()

    def prepare_set_group_4x(self, name: str, N, K) -> List:
        assert name in self.config_map, f"Operator {name} not found in config map"
        optimize_strategy = self.config_map[name]["optimize_stratety"]
        return [
            self.generate_op_unit(
                self.generate_operator_config(
                    name, [1, 16, 32, 64, 128, 256, 512] if optimize_strategy
                    == OptimizeStrategy.SingleBatchDecodeOnly else [16, 32, 64, 128, 256, 512], N,
                    K)),
        ]

    def prepare_benchmark_sets(self):
        """Prepare benchmark sets."""
        self.add_benchmark_set(
            "FP16xUINT4_ACCFP16_NT_STRATEGY_GEMV",
            [
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_GEMV", 16384, 16384),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_GEMV", 3200, 3200),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_GEMV", 8640, 3200),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_GEMV", 3200, 8640),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_GEMV", 5120, 5120),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_GEMV", 13824, 5120),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_GEMV", 5120, 13824),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_GEMV", 6656, 6656),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_GEMV", 17920, 6656),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_GEMV", 6656, 17920),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_GEMV", 1024, 8192),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_GEMV", 8192, 8192),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_GEMV", 28672, 8192),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_GEMV", 8192, 28672),
            ],
        )

        self.add_benchmark_set(
            "FP16xUINT4_ACCFP16_NT_STRATEGY_ContigiousBatching",
            [
                *self.prepare_set_group_4x(
                    "FP16xUINT4_ACCFP16_NT_STRATEGY_ContigiousBatching",
                    16384,
                    16384,
                ),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_ContigiousBatching",
                                           3200, 3200),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_ContigiousBatching",
                                           8640, 3200),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_ContigiousBatching",
                                           3200, 8640),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_ContigiousBatching",
                                           5120, 5120),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_ContigiousBatching",
                                           13824, 5120),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_ContigiousBatching",
                                           5120, 13824),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_ContigiousBatching",
                                           6656, 6656),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_ContigiousBatching",
                                           17920, 6656),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_ContigiousBatching",
                                           6656, 17920),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_ContigiousBatching",
                                           1024, 8192),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_ContigiousBatching",
                                           8192, 8192),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_ContigiousBatching",
                                           28672, 8192),
                *self.prepare_set_group_4x("FP16xUINT4_ACCFP16_NT_STRATEGY_ContigiousBatching",
                                           8192, 28672),
            ],
        )

    def generate_operator_config(self, name: str, M, N, K) -> MatmulConfig:
        """Generate configuration for the given operator."""
        if name not in self.config_map:
            raise ValueError(f"Operator {name} not found in config map")
        return self.get_operator_config()(
            M=M,
            N=N,
            K=K,
            **self.config_map[name],
        )

    def serialize_results(self) -> None:
        """Serialize benchmark results into JSON files."""
        commit_id_path = f"CommitID_{self.CURRENT_COMMIT_ID}"
        log_commit_path = path.join(self.log_path, commit_id_path)

        if not path.exists(log_commit_path):
            makedirs(log_commit_path)

        # Save benchmark results into JSON
        self._save_json(
            self.benchmark_results,
            path.join(log_commit_path, self.BENCHMARK_RESULTS_FILE),
        )

        # Save benchmark shapes into JSON
        shapes: Dict[List[List[Union[List, int], int, int]]] = {}

        # Iterate through the benchmark results to extract the shapes
        for name, results in self.benchmark_results.items():
            shapes[name] = []
            for i, _ in enumerate(results):
                config = self.benchmark_sets[name][i][1]
                dyn_prof_shape = self.benchmark_sets[name][i][2]
                shapes[name].append([config.M, config.N, config.K, dyn_prof_shape])

        self._save_json(shapes, path.join(log_commit_path, self.BENCHMARK_SHAPES_FILE))

        # Save device info into JSON
        self._save_json(
            {"device": self.benchmark_target},
            path.join(log_commit_path, self.BENCHMARK_DEVICE_FILE),
        )

    def _save_json(self, data, file_path):
        """Helper function to save JSON data to a file."""
        with open(file_path, "w") as f:
            json.dump(data, f)

    @classmethod
    def deserialize_from_logs(cls, commit_id: str) -> None:
        """Deserialize benchmark results from JSON files."""
        benchmark = cls()
        commit_id_path = f"CommitID_{commit_id}"
        log_commit_path = path.join(benchmark.log_path, commit_id_path)

        benchmark.benchmark_results = cls._load_json(
            path.join(log_commit_path, cls.BENCHMARK_RESULTS_FILE))

        shapes_file = path.join(log_commit_path, cls.BENCHMARK_SHAPES_FILE)

        with open(shapes_file, "r") as f:
            shapes = json.load(f)
            for name, shape_list in shapes.items():
                for shape in shape_list:
                    M, N, K, dyn_prof_shape = shape
                    benchmark.add_benchmark_set(
                        name,
                        [
                            benchmark.generate_op_unit(
                                benchmark.generate_operator_config(name, M, N, K),
                                dynamic_profiling_shape=dyn_prof_shape,
                            )
                        ],
                    )

        benchmark.benchmark_target = cls._load_json(
            path.join(log_commit_path, cls.BENCHMARK_DEVICE_FILE))["device"]

        return benchmark

    @staticmethod
    def _load_json(file_path):
        """Helper function to load JSON data from a file."""
        with open(file_path, "r") as f:
            return json.load(f)

    def report(self):
        """Generate and print a report of the benchmark results."""
        results4compare = {}
        for name, results in self.benchmark_results.items():
            name, strategy = name.split("STRATEGY")
            results4compare.setdefault(name, {})[strategy] = results

        for name, strategy in results4compare.items():
            table_data = [
                ["TAG:", name, "Device:", self.benchmark_target],
                [
                    "Shape (M-N-K / N-K_M)",
                    "Single Batching Time (ms)",
                    "Shape (M-N-K / N-K_M)",
                    "Contiguous Batching Time (ms)",
                    "Tune Time (s)",
                ],
            ]

            def legalize_shape(M, N, K, dyn_prof_shape):
                """Generate a string representation of the operator shape.

                Args:
                    M: The M dimension (can be an int or a tuple).
                    N: The N dimension (must be an int).
                    K: The K dimension (must be an int).
                    dyn_prof_shape: The dynamic profiling shape (dict with "m" key if M is dynamic).

                Returns:
                    A string representing the shape in either 'M-N-K' or 'N-K_M' format.
                """
                if isinstance(M, int):
                    return f"{M}-{N}-{K}"
                elif dyn_prof_shape and "m" in dyn_prof_shape:
                    return f"{M}-{N}-{K}_{dyn_prof_shape['m']}"
                else:
                    # Calculate the average of tuple M
                    str_m = "[" + "-".join(str(m) for m in M) + "]"
                    opt_m = sum(M) / len(M)
                    return f"{N}-{K}_{str_m}_{opt_m}"

            data = []
            for strategy_name, results in strategy.items():
                tmp_data = []
                origin_name = f"{name}STRATEGY{strategy_name}"
                for i, benchmark_set in enumerate(self.benchmark_sets[origin_name]):
                    op_config = benchmark_set[1]
                    sub_results = results[i * len(self.OPT_SHAPES):(i + 1) * len(self.OPT_SHAPES)]
                    for i, result in enumerate(sub_results):
                        latency = result[0]
                        dyn_prof_shape = {"m": self.OPT_SHAPES[i]}
                        shape = legalize_shape("DYN", op_config.N, op_config.K, dyn_prof_shape)
                        latency_str = "N/A" if latency is None else f"{latency:.3f}"
                        tmp_data.append([shape, latency_str])
                if len(data) == 0:
                    data = tmp_data
                else:
                    for i, item in enumerate(tmp_data):
                        data[i].extend(item)

            for i, item in enumerate(data):
                base = item[1]
                head = item[3]

                speedup = float(head) / float(base) - 1
                symbol = "+" if speedup > 0 else "-"
                speedup = abs(speedup)
                data[i][3] = f"{head} ({symbol}{speedup * 100 :.3f}%)"
                table_data.append([*data[i], "N/A"])

            print(tabulate(table_data, headers="firstrow", tablefmt="fancy_grid"))

            for data in table_data:
                print(data)

    def get_operator(self):
        """Return the Matmul operator."""
        return Matmul

    def get_operator_config(self):
        """Return the Matmul operator configuration."""
        return MatmulConfig

    def make_operator(self, operator: Matmul, config: MatmulConfig) -> Matmul:
        """Make an Matmul instance."""
        # Disable default tuning when do benchmark
        return operator(config, target=self.benchmark_target, enable_tuning=False)

    def benchmark(self):
        """Run benchmarks on all benchmark sets."""
        for name, benchmark_set in self.benchmark_sets.items():
            self.benchmark_results[name] = []
            for op, config, _ in benchmark_set:
                for opt in self.OPT_SHAPES:
                    self.benchmark_results[name].extend(
                        [self.run_benchmark(op, config, {"m": opt})])

    def run_compare_strategy(self, report=True, serialize=True, enable_tuning: bool = False):
        """Run the benchmark process."""

        if not path.exists(self.log_path):
            makedirs(self.log_path)

        if enable_tuning:
            self.enable_tuning()

        self.prepare_benchmark_sets()
        self.benchmark()

        if report:
            self.report()

        self.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bitblas Matmul Operator Benchmark")

    parser.add_argument(
        "--enable_tuning",
        action="store_true",
        help="Enable hardware-aware tuning",
    )

    args = parser.parse_args()
    enable_tuning = args.enable_tuning
    BitblasMatmulOpsBenchmarkCompareStategies().run_compare_strategy(
        enable_tuning=args.enable_tuning)
