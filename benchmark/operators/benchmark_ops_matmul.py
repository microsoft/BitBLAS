# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from base import BitblasOperatorBenchmarkBase
from bitblas.ops import Matmul, MatmulConfig
from bitblas import set_log_level

set_log_level("DEBUG")

class BitblasMatmulOpsBenchmark(BitblasOperatorBenchmarkBase):

    config_map = {
        "FP16xFP16_ACCFP16_NT": {
            "in_dtype": "float16",
            "out_dtype": "float16",
            "accum_dtype": "float16",
        }
    }

    def prepare_benchmark_sets(self):
        self.add_benchmark_set(
            "FP16xFP16_ACCFP16_NT",
            [
                (Matmul, self.generate_operator_config("FP16xFP16_ACCFP16_NT", 16384, 16384, 16384)),
            ],
        )

    def generate_operator_config(self, name:str, M, N, K) -> MatmulConfig:
        if name not in self.config_map:
            raise ValueError(f"Operator {name} not found in config map")
        return MatmulConfig(
            M=M,
            N=N,
            K=K,
            **self.config_map[name],
        )

    def get_operator(self):
        return Matmul

    def get_operator_config(self):
        return MatmulConfig

if __name__ == "__main__":
    BitblasMatmulOpsBenchmark().run()
