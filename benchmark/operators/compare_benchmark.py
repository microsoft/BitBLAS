# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from benchmark_ops_matmul import BitblasMatmulOpsBenchmark, HELPER_MESSAGE
from tabulate import tabulate
from typing import Tuple


def compare(base: BitblasMatmulOpsBenchmark, head: BitblasMatmulOpsBenchmark):
    """Generate and print a report of the benchmark results."""
    for name, results in head.benchmark_results.items():
        table_data = [
            ["TAG:", name, "Device:", head.benchmark_target],
            [
                "Shape (M-N-K / N-K_M)",
                "Time (ms)",
                "Throughput (TFLOPS)",
                "Tune Time (s)",
            ],
        ]

        def get_suffix(base, head):
            symbol = "↑" if head > base else "↓" if head < base else "="
            ratio = f"{((head - base) / base) * 100:.2f}%" if base is not None else "N/A"
            return f"{symbol}({ratio})"

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
            op_config = head.benchmark_sets[name][i][1]
            dyn_prof_shape = head.benchmark_sets[name][i][2]
            shape = legalize_shape(op_config.M, op_config.N, op_config.K, dyn_prof_shape)

            benchmark_M = (
                sum(op_config.M) /
                len(op_config.M) if isinstance(op_config.M, Tuple) else op_config.M)

            base_latency = base.benchmark_results[name][i][0]
            if latency is not None:
                throughput = (2 * benchmark_M * op_config.N * op_config.K / (latency * 1e-3) / 1e12)
                base_throughput = (2 * benchmark_M * op_config.N * op_config.K /
                                   (base_latency * 1e-3) / 1e12)
                throughput = f"{throughput:.3f}{get_suffix(base_throughput, throughput)}"
            else:
                throughput = "N/A"

            if base_latency is not None:
                latency_str = f"{latency:.3f}{get_suffix(base_latency, latency)}"
            else:
                latency_str = "N/A"

            base_tuning_time = base.benchmark_results[name][i][1]
            if tuning_time is not None:
                tuning_time_str = f"{tuning_time:.3f}{get_suffix(base_tuning_time, tuning_time)}"
            else:
                tuning_time_str = "N/A"

            table_data.append([shape, latency_str, throughput, tuning_time_str])

        print(tabulate(table_data, headers="firstrow", tablefmt="fancy_grid"))
        print(HELPER_MESSAGE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default="df7e9aa61e3db411ac3f2fd98a1854a36194ef0c",
        type=str,
        help="the base commit id",
    )
    parser.add_argument(
        "--head",
        default="1c033654d15dc98707edeaabfcd8951b3a800734",
        type=str,
        help="the head commit id",
    )
    args = parser.parse_args()

    base_benchmark = BitblasMatmulOpsBenchmark.deserialize_from_logs(args.base)

    head_benchmark = BitblasMatmulOpsBenchmark.deserialize_from_logs(args.head)

    compare(base_benchmark, head_benchmark)
