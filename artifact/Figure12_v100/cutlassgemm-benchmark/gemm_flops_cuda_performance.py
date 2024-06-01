# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Model benchmark example for Cutlass GEMM FLOPs performance.

Commands to run:
  python3 examples/benchmarks/gemm_flops_cuda_performance.py
"""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.common.utils import logger

_gemm_sizes = [
    [2, 2, 2],
    [4, 4, 4],
    [8, 8, 8],
    [16, 16, 16],
    [32, 32, 32],
    [64, 64, 64],
    [128, 128, 128],
    [256, 256, 256],
    [512, 512, 512],
    [1024, 1024, 1024],
    [2048, 2048, 2048],
    [4096, 4096, 4096],
    [8192, 8192, 8192],
    [16384, 16384, 16384]
]

if __name__ == '__main__':
    for m, k, n in _gemm_sizes:
        
        parameters = '--n {0} --k {1} --m {2}'.format(n, k, m)
        context = BenchmarkRegistry.create_benchmark_context('gemm-flops', platform=Platform.CUDA, parameters=parameters)

        benchmark = BenchmarkRegistry.launch_benchmark(context)
        if benchmark:
            logger.info(
                'benchmark: {}, return code: {}, result: {}'.format(
                    benchmark.name, benchmark.return_code, benchmark.result
                )
            )
