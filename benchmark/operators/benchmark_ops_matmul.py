# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas.benchmark import BitblasOperatorBenchmarkBase
from bitblas.ops import Matmul, MatmulConfig
from bitblas.utils import get_commit_id
from bitblas import set_log_level
from tabulate import tabulate
import json
from os import path, makedirs
from typing import Tuple

set_log_level("DEBUG")


class BitblasMatmulOpsBenchmark(BitblasOperatorBenchmarkBase):

    BENCHMARK_RESULTS_FILE = "benchmark_results.json"
    BENCHMARK_SHAPES_FILE = "benchmark_shapes.json"
    BENCHMARK_DEVICE_FILE = "benchmark_device.json"

    config_map = {
        "FP16xFP16_ACCFP16_NT": {
            "in_dtype": "float16",
            "out_dtype": "float16",
            "accum_dtype": "float16",
        }
    }

    CURRENT_COMMIT_ID = get_commit_id()

    def prepare_benchmark_sets(self):
        """Prepare benchmark sets."""
        self.disable_tuning()
        self.add_benchmark_set(
            "FP16xFP16_ACCFP16_NT",
            [
                (
                    Matmul,
                    self.generate_operator_config("FP16xFP16_ACCFP16_NT", [1, 1024], 16384, 16384),
                ),
            ],
        )

    def generate_operator_config(self, name: str, M, N, K) -> MatmulConfig:
        """Generate configuration for the given operator."""
        if name not in self.config_map:
            raise ValueError(f"Operator {name} not found in config map")
        return MatmulConfig(
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
        shapes = [(config.M, config.N, config.K)
                  for name, results in self.benchmark_results.items() for i, _ in enumerate(results)
                  for config in [self.benchmark_sets[name][i][1]]]
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

    def deserialize_results(self, log_path: str) -> None:
        """Deserialize benchmark results from JSON files."""
        self.benchmark_results = self._load_json(path.join(log_path, self.BENCHMARK_RESULTS_FILE))

        shapes_file = path.join(log_path, self.BENCHMARK_SHAPES_FILE)
        with open(shapes_file, "r") as f:
            shapes = json.load(f)
            # TODO: Reconstruction of benchmark_sets from shapes
            del shapes

        self.benchmark_target = self._load_json(path.join(log_path,
                                                          self.BENCHMARK_DEVICE_FILE))["device"]

    def _load_json(self, file_path):
        """Helper function to load JSON data from a file."""
        with open(file_path, "r") as f:
            return json.load(f)

    def report(self):
        """Generate and print a report of the benchmark results."""
        for name, results in self.benchmark_results.items():
            table_data = [
                ["TAG:", name, "Device:", self.benchmark_target],
                [
                    "Shape (M-N-K)",
                    "Time (ms)",
                    "Throughput (TFLOPS)",
                    "Tune Time (s)",
                ],
            ]

            for i, (latency, tuning_time) in enumerate(results):
                op_config = self.benchmark_sets[name][i][1]
                shape = f"{op_config.M}-{op_config.N}-{op_config.K}"

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

    def get_operator(self):
        """Return the Matmul operator."""
        return Matmul

    def get_operator_config(self):
        """Return the Matmul operator configuration."""
        return MatmulConfig


if __name__ == "__main__":
    BitblasMatmulOpsBenchmark().run()
