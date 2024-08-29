# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas.benchmark import BitblasOperatorBenchmarkBase
from bitblas import Matmul, MatmulConfig
from bitblas.utils import get_commit_id
from bitblas import set_log_level
from tabulate import tabulate
from os import path, makedirs
from typing import List
import argparse
from tqdm import tqdm

set_log_level("DEBUG")


class BitblasMatmulOpsBenchmarkCompareStategies(BitblasOperatorBenchmarkBase):

    BENCHMARK_RESULTS_FILE = "benchmark_results.json"
    BENCHMARK_SHAPES_FILE = "benchmark_shapes.json"
    BENCHMARK_DEVICE_FILE = "benchmark_device.json"

    config_map = {
        "FP16xFP16_GEMV": {
            "A_dtype": "float16",
            "W_dtype": "float16",
            "accum_dtype": "float16",
        },
        "FP16xUINT4_GEMV_DECODING_NAIVE": {
            "A_dtype": "float16",
            "W_dtype": "uint4",
            "accum_dtype": "float16",
            "fast_decoding": False,
        },
        "FP16xUINT4_GEMV_DECODING_FAST": {
            "A_dtype": "float16",
            "W_dtype": "uint4",
            "accum_dtype": "float16",
            "fast_decoding": True,
        },
        "FP16xUINT2_GEMV_DECODING_NAIVE": {
            "A_dtype": "float16",
            "W_dtype": "uint2",
            "accum_dtype": "float16",
            "fast_decoding": False,
        },
        "FP16xUINT2_GEMV_DECODING_FAST": {
            "A_dtype": "float16",
            "W_dtype": "uint2",
            "accum_dtype": "float16",
            "fast_decoding": True,
        },
        "INT8xUINT2_GEMV_DECODING_NAIVE": {
            "A_dtype": "int8",
            "W_dtype": "uint2",
            "accum_dtype": "int32",
            "out_dtype": "int32",
            "fast_decoding": False,
        },
        "INT8xUINT2_GEMV_DECODING_FAST": {
            "A_dtype": "int8",
            "W_dtype": "uint2",
            "accum_dtype": "int32",
            "out_dtype": "int32",
            "fast_decoding": True,
        },
    }

    OPT_SHAPES = 1  # our test focuses on GEMV only

    CURRENT_COMMIT_ID = get_commit_id()

    def __init__(self):
        super().__init__()

    def prepare_set_group_4x(self, name: str, N, K) -> List:
        assert name in self.config_map, f"Operator {name} not found in config map"
        return [
            self.generate_op_unit(self.generate_operator_config(name, self.OPT_SHAPES, N, K)),
        ]

    def prepare_benchmark_sets(self):
        """Prepare benchmark sets."""

        self.add_benchmark_set(
            "FP16xUINT4_GEMV_DECODING_NAIVE",
            [
                *self.prepare_set_group_4x("FP16xUINT4_GEMV_DECODING_NAIVE", 16384, 16384),
                *self.prepare_set_group_4x("FP16xUINT4_GEMV_DECODING_NAIVE", 3200, 3200),
                *self.prepare_set_group_4x("FP16xUINT4_GEMV_DECODING_NAIVE", 8640, 3200),
                *self.prepare_set_group_4x("FP16xUINT4_GEMV_DECODING_NAIVE", 3200, 8640),
                *self.prepare_set_group_4x("FP16xUINT4_GEMV_DECODING_NAIVE", 1024, 8192),
                *self.prepare_set_group_4x("FP16xUINT4_GEMV_DECODING_NAIVE", 8192, 8192),
                *self.prepare_set_group_4x("FP16xUINT4_GEMV_DECODING_NAIVE", 28672, 8192),
                *self.prepare_set_group_4x("FP16xUINT4_GEMV_DECODING_NAIVE", 8192, 28672),
            ],
        )

        self.add_benchmark_set(
            "FP16xUINT4_GEMV_DECODING_FAST",
            [
                *self.prepare_set_group_4x(
                    "FP16xUINT4_GEMV_DECODING_FAST",
                    16384,
                    16384,
                ),
                *self.prepare_set_group_4x(
                    "FP16xUINT4_GEMV_DECODING_FAST",
                    3200,
                    3200,
                ),
                *self.prepare_set_group_4x(
                    "FP16xUINT4_GEMV_DECODING_FAST",
                    8640,
                    3200,
                ),
                *self.prepare_set_group_4x(
                    "FP16xUINT4_GEMV_DECODING_FAST",
                    3200,
                    8640,
                ),
                *self.prepare_set_group_4x(
                    "FP16xUINT4_GEMV_DECODING_FAST",
                    1024,
                    8192,
                ),
                *self.prepare_set_group_4x(
                    "FP16xUINT4_GEMV_DECODING_FAST",
                    8192,
                    8192,
                ),
                *self.prepare_set_group_4x(
                    "FP16xUINT4_GEMV_DECODING_FAST",
                    28672,
                    8192,
                ),
                *self.prepare_set_group_4x(
                    "FP16xUINT4_GEMV_DECODING_FAST",
                    8192,
                    28672,
                ),
            ],
        )

        self.add_benchmark_set(
            "FP16xUINT2_GEMV_DECODING_NAIVE",
            [
                *self.prepare_set_group_4x("FP16xUINT2_GEMV_DECODING_NAIVE", 16384, 16384),
                *self.prepare_set_group_4x("FP16xUINT2_GEMV_DECODING_NAIVE", 3200, 3200),
                *self.prepare_set_group_4x("FP16xUINT2_GEMV_DECODING_NAIVE", 8640, 3200),
                *self.prepare_set_group_4x("FP16xUINT2_GEMV_DECODING_NAIVE", 3200, 8640),
                *self.prepare_set_group_4x("FP16xUINT2_GEMV_DECODING_NAIVE", 1024, 8192),
                *self.prepare_set_group_4x("FP16xUINT2_GEMV_DECODING_NAIVE", 8192, 8192),
                *self.prepare_set_group_4x("FP16xUINT2_GEMV_DECODING_NAIVE", 28672, 8192),
                *self.prepare_set_group_4x("FP16xUINT2_GEMV_DECODING_NAIVE", 8192, 28672),
            ],
        )

        self.add_benchmark_set(
            "FP16xUINT2_GEMV_DECODING_FAST",
            [
                *self.prepare_set_group_4x(
                    "FP16xUINT2_GEMV_DECODING_FAST",
                    16384,
                    16384,
                ),
                *self.prepare_set_group_4x(
                    "FP16xUINT2_GEMV_DECODING_FAST",
                    3200,
                    3200,
                ),
                *self.prepare_set_group_4x(
                    "FP16xUINT2_GEMV_DECODING_FAST",
                    8640,
                    3200,
                ),
                *self.prepare_set_group_4x(
                    "FP16xUINT2_GEMV_DECODING_FAST",
                    3200,
                    8640,
                ),
                *self.prepare_set_group_4x(
                    "FP16xUINT2_GEMV_DECODING_FAST",
                    1024,
                    8192,
                ),
                *self.prepare_set_group_4x(
                    "FP16xUINT2_GEMV_DECODING_FAST",
                    8192,
                    8192,
                ),
                *self.prepare_set_group_4x(
                    "FP16xUINT2_GEMV_DECODING_FAST",
                    28672,
                    8192,
                ),
                *self.prepare_set_group_4x(
                    "FP16xUINT2_GEMV_DECODING_FAST",
                    8192,
                    28672,
                ),
            ],
        )

        self.add_benchmark_set(
            "INT8xUINT2_GEMV_DECODING_NAIVE",
            [
                *self.prepare_set_group_4x("INT8xUINT2_GEMV_DECODING_NAIVE", 16384, 16384),
                *self.prepare_set_group_4x("INT8xUINT2_GEMV_DECODING_NAIVE", 3200, 3200),
                *self.prepare_set_group_4x("INT8xUINT2_GEMV_DECODING_NAIVE", 8640, 3200),
                *self.prepare_set_group_4x("INT8xUINT2_GEMV_DECODING_NAIVE", 3200, 8640),
                *self.prepare_set_group_4x("INT8xUINT2_GEMV_DECODING_NAIVE", 1024, 8192),
                *self.prepare_set_group_4x("INT8xUINT2_GEMV_DECODING_NAIVE", 8192, 8192),
                *self.prepare_set_group_4x("INT8xUINT2_GEMV_DECODING_NAIVE", 28672, 8192),
                *self.prepare_set_group_4x("INT8xUINT2_GEMV_DECODING_NAIVE", 8192, 28672),
            ],
        )

        self.add_benchmark_set(
            "INT8xUINT2_GEMV_DECODING_FAST",
            [
                *self.prepare_set_group_4x(
                    "INT8xUINT2_GEMV_DECODING_FAST",
                    16384,
                    16384,
                ),
                *self.prepare_set_group_4x(
                    "INT8xUINT2_GEMV_DECODING_FAST",
                    3200,
                    3200,
                ),
                *self.prepare_set_group_4x(
                    "INT8xUINT2_GEMV_DECODING_FAST",
                    8640,
                    3200,
                ),
                *self.prepare_set_group_4x(
                    "INT8xUINT2_GEMV_DECODING_FAST",
                    3200,
                    8640,
                ),
                *self.prepare_set_group_4x(
                    "INT8xUINT2_GEMV_DECODING_FAST",
                    1024,
                    8192,
                ),
                *self.prepare_set_group_4x(
                    "INT8xUINT2_GEMV_DECODING_FAST",
                    8192,
                    8192,
                ),
                *self.prepare_set_group_4x(
                    "INT8xUINT2_GEMV_DECODING_FAST",
                    28672,
                    8192,
                ),
                *self.prepare_set_group_4x(
                    "INT8xUINT2_GEMV_DECODING_FAST",
                    8192,
                    28672,
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

    def report(self):
        """Generate and print a report of the benchmark results."""
        results4compare = {}
        for name, results in self.benchmark_results.items():
            if "DECODING" not in name:
                name = f"{name}"
                strategy = ""
            else:
                name, strategy = name.split("DECODING")
            results4compare.setdefault(name, {})[strategy] = results

        data = []
        for name, strategy in results4compare.items():
            table_data = [
                ["TAG:", name, "Device:", self.benchmark_target],
                [
                    "Shape (M-N-K / N-K_M)",
                    "Native Decoding Time (ms)",
                    "Shape (M-N-K / N-K_M)",
                    "Fast Decoding Time (ms)",
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

            for strategy_name, results in strategy.items():
                tmp_data = []
                if strategy_name == "":
                    origin_name = name
                else:
                    origin_name = f"{name}DECODING{strategy_name}"
                for i, benchmark_set in enumerate(self.benchmark_sets[origin_name]):
                    op_config = benchmark_set[1]
                    if isinstance(self.OPT_SHAPES, int):
                        sub_results = results[i]
                        latency = sub_results[0]
                        dyn_prof_shape = {"m": self.OPT_SHAPES}
                        shape = legalize_shape("DYN", op_config.N, op_config.K, dyn_prof_shape)
                        latency_str = "N/A" if latency is None else f"{latency:.3f}"
                        tmp_data.append([shape, latency_str])
                    else:
                        sub_results = results[i * len(self.OPT_SHAPES):(i + 1) *
                                              len(self.OPT_SHAPES)]
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
        # Calculate the total number of benchmark runs for the progress bar
        total_runs = sum(
            (len(benchmark_set) *
             (len(self.OPT_SHAPES) if isinstance(self.OPT_SHAPES, list) else self.OPT_SHAPES))
            for benchmark_set in self.benchmark_sets.values())

        with tqdm(total=total_runs, desc="Total Progress", unit="benchmark") as pbar:
            for name, benchmark_set in self.benchmark_sets.items():
                self.benchmark_results[name] = []
                for op, config, _ in benchmark_set:
                    if isinstance(self.OPT_SHAPES, int):
                        print(f"Running benchmark for {name} with shape {self.OPT_SHAPES}")
                        self.benchmark_results[name].extend(
                            [self.run_benchmark(op, config, {"m": self.OPT_SHAPES})])
                        # Update the progress bar after each run
                        pbar.update(1)
                    else:
                        for opt in self.OPT_SHAPES:
                            print(f"Running benchmark for {name} with shape {opt}")
                            self.benchmark_results[name].extend(
                                [self.run_benchmark(op, config, {"m": opt})])
                            # Update the progress bar after each run
                            pbar.update(1)

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

    def serialize_results(self) -> None:
        """Serialize the benchmark results."""
        pass


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
