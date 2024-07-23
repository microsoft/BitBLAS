# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas.benchmark import BitblasOperatorBenchmarkBase
from bitblas import Matmul, MatmulConfig
from bitblas.utils import get_commit_id
from bitblas import set_log_level
from tabulate import tabulate
import json
from os import path, makedirs
from typing import Tuple, Dict, List, Union

set_log_level("DEBUG")

HELPER_MESSAGE = """
**Note**: Bitblas supports dynamic shape tensors as input, resulting in two possible formats for the \
"Shape (M-N-K / N-K_M)" column in the report. The "M-N-K" format indicates a static shape operator, \
while the "N-K_M" format denotes a dynamic shape operator where only the M dimension is dynamic. \
In this context, "_M" represents the specific M shape used for dynamic profiling.
"""


class BitblasMatmulOpsBenchmark(BitblasOperatorBenchmarkBase):

    BENCHMARK_RESULTS_FILE = "benchmark_results.json"
    BENCHMARK_SHAPES_FILE = "benchmark_shapes.json"
    BENCHMARK_DEVICE_FILE = "benchmark_device.json"

    config_map = {
        "FP16xFP16_ACCFP16_NT": {
            "A_dtype": "float16",
            "W_dtype": "float16",
            "accum_dtype": "float16",
        },
        "INT8xINT8_ACCINT32_NT": {
            "A_dtype": "int8",
            "W_dtype": "int8",
            "accum_dtype": "int32",
            "out_dtype": "int8",
        },
        "FP16xINT4_ACCINT32_NT": {
            "A_dtype": "float16",
            "W_dtype": "int4",
            "accum_dtype": "float16",
        },
    }

    CURRENT_COMMIT_ID = get_commit_id()

    def prepare_benchmark_sets(self):
        """Prepare benchmark sets."""
        self.add_benchmark_set(
            "FP16xFP16_ACCFP16_NT",
            [
                self.generate_op_unit(
                    self.generate_operator_config("FP16xFP16_ACCFP16_NT", 16384, 16384, 16384),),
                self.generate_op_unit(
                    self.generate_operator_config("FP16xFP16_ACCFP16_NT", [1, 1024], 16384, 16384),
                    dynamic_profiling_shape={"M": 1024},
                ),
            ],
        )

        self.add_benchmark_set(
            "INT8xINT8_ACCINT32_NT",
            [
                self.generate_op_unit(
                    self.generate_operator_config("INT8xINT8_ACCINT32_NT", 16384, 16384, 16384),),
                self.generate_op_unit(
                    self.generate_operator_config("INT8xINT8_ACCINT32_NT", [1, 1024], 16384, 16384),
                    dynamic_profiling_shape={"M": 1024},
                ),
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
        for name, results in self.benchmark_results.items():
            table_data = [
                ["TAG:", name, "Device:", self.benchmark_target],
                [
                    "Shape (M-N-K / N-K_M)",
                    "Time (ms)",
                    "Throughput (TFLOPS)",
                    "Tune Time (s)",
                ],
            ]

            def legalize_shape(M, N, K, dyn_prof_shape):
                """Generate a string representation of the operator shape.

                Args:
                    M: The M dimension (can be an int or a tuple).
                    N: The N dimension (must be an int).
                    K: The K dimension (must be an int).
                    dyn_prof_shape: The dynamic profiling shape (dict with 'M' key if M is dynamic).

                Returns:
                    A string representing the shape in either 'M-N-K' or 'N-K_M' format.
                """
                if isinstance(M, int):
                    return f"{M}-{N}-{K}"
                elif dyn_prof_shape and "M" in dyn_prof_shape:
                    return f"{N}-{K}_{dyn_prof_shape['M']}"
                else:
                    # Calculate the average of tuple M
                    opt_m = sum(M) / len(M)
                    return f"{N}-{K}_{opt_m}"

            for i, (latency, tuning_time) in enumerate(results):
                op_config = self.benchmark_sets[name][i][1]
                dyn_prof_shape = self.benchmark_sets[name][i][2]
                shape = legalize_shape(op_config.M, op_config.N, op_config.K, dyn_prof_shape)

                benchmark_M = (
                    sum(op_config.M) /
                    len(op_config.M) if isinstance(op_config.M, Tuple) else op_config.M)

                throughput = (
                    f"{(2 * benchmark_M * op_config.N * op_config.K / (latency * 1e-3) / 1e12):.3f}"
                    if latency else "N/A")
                latency_str = "N/A" if latency is None else f"{latency:.3f}"
                tuning_time_str = ("N/A" if tuning_time is None else f"{tuning_time:.3f}")

                table_data.append([shape, latency_str, throughput, tuning_time_str])

            print(tabulate(table_data, headers="firstrow", tablefmt="fancy_grid"))
            print(HELPER_MESSAGE)

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


if __name__ == "__main__":
    BitblasMatmulOpsBenchmark().run(enable_tuning=False)
