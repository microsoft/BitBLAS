import os
import json
import re
from reproduce_result import (
    matmul_providers,
    matmul_times_data,
    conv_providers,
    conv_times_data
)

def parse_runtime(log_path, A_layout='col', B_layout='row'):
    fp32_cudacore_runtime = []
    fp32_tensorcore_runtime = []
    fp16_cudacore_runtime = []
    fp16_tensorcore_runtime = []
    s8_cudacore_runtime = []
    s8_tensorcore_runtime = []
    with open(log_path) as f:
        render = csv.reader(f)
        # get header
        header_row = next(render)

        A_idx = header_row.index('A')
        B_idx = header_row.index('B')
        Op_class_idx = header_row.index('op_class')
        runtime_idx = header_row.index('Runtime')
        for row in render:
            if A_layout not in row[A_idx]:  # Skip rows that don't match the layout
                continue
            if B_layout not in row[B_idx]:  # Skip rows that don't match the layout
                continue
            runtime_data = float(row[runtime_idx])
            if 'cf32' in row[A_idx]:
                continue
            if 'bf32' in row[A_idx]:
                continue
            if 'f32' in row[A_idx]:
                fp32_tensorcore_runtime.append(runtime_data)
                if 'tensor' in row[Op_class_idx]:
                    fp32_tensorcore_runtime.append(runtime_data)
                else:
                    fp32_cudacore_runtime.append(runtime_data)
            elif 'f16' in row[A_idx]:
                if 'tensor' in row[Op_class_idx]:
                    fp16_tensorcore_runtime.append(runtime_data)
                else:
                    fp16_cudacore_runtime.append(runtime_data)
            elif 's8' in row[A_idx]:
                if 'tensor' in row[Op_class_idx]:
                    s8_tensorcore_runtime.append(runtime_data)
                else:
                    s8_cudacore_runtime.append(runtime_data)

    # print(fp32_tensorcore_runtime)
    min_fp32_cudacore_runtime, min_fp32_tensorcore_runtime, min_fp16_cudacore_runtime, min_fp16_tensorcore_runtime, min_s8_cudacore_runtime, min_s8_tensorcore_runtime = [min(runtime_array) if len(
        runtime_array) else 'not support' for runtime_array in (fp32_cudacore_runtime, fp32_tensorcore_runtime, fp16_cudacore_runtime, fp16_tensorcore_runtime, s8_cudacore_runtime, s8_tensorcore_runtime)]

    return min_fp32_cudacore_runtime, min_fp32_tensorcore_runtime, min_fp16_cudacore_runtime, min_fp16_tensorcore_runtime, min_s8_cudacore_runtime, min_s8_tensorcore_runtime


def get_and_print(log_path, log_case, A_layout='col', B_layout='row'):
    min_fp32_cudacore_runtime, min_fp32_tensorcore_runtime, min_fp16_cudacore_runtime, min_fp16_tensorcore_runtime, min_s8_cudacore_runtime, min_s8_tensorcore_runtime = parse_runtime(
        log_path, A_layout, B_layout)
    res = 0.0
    if log_case == 'min_fp32_cudacore_runtime':
        print(min_fp32_cudacore_runtime)
        res = min_fp32_cudacore_runtime
    elif log_case == 'min_fp32_tensorcore_runtime':
        print(min_fp32_tensorcore_runtime)
        res = min_fp32_tensorcore_runtime
    elif log_case == 'min_fp16_cudacore_runtime':
        print(min_fp16_cudacore_runtime)
        res = min_fp16_cudacore_runtime
    elif log_case == 'min_fp16_tensorcore_runtime':
        print(min_fp16_tensorcore_runtime)
        res = min_fp16_tensorcore_runtime
    elif log_case == 'min_s8_cudacore_runtime':
        print(min_s8_cudacore_runtime)
        res = min_s8_cudacore_runtime
    elif log_case == 'min_s8_tensorcore_runtime':
        print(min_s8_tensorcore_runtime)
        res = min_s8_tensorcore_runtime
    return res
    
# parse the results from cutlass
for m, n, k in [
        [1,14336,57344],
        [32,14336,57344],
        [4096,14336,57344],
        [1,8192,28672],
        [32,8192,28672],
        [4096,8192,28672],
    ]:
    log_path = f"./cutlassgemm-benchmark/csv_logs/cutlass_shape_{m}_{k}_{n}_performance.csv"
    if not os.path.exists(log_path):
        continue
    data = get_and_print(log_path, "min_fp16_tensorcore_runtime")
    print(data)

# parse the results from cutlass_fpa_intb
for m, n, k in [
        [1,14336,57344],
        [32,14336,57344],
        [4096,14336,57344],
        [1,8192,28672],
        [32,8192,28672],
        [4096,8192,28672],
    ]:
    log_path = f"./cutlass-dequantize-benchmark/logs/cutlass_fpa_intb.log"
    if not os.path.exists(log_path):
        continue
    else:
        with open(log_path) as f:
            lines = f.readlines()
            for line in lines:
                if f"{m}_{n}_{k}" in line:
                    data = float(re.findall(r"\d+\.\d+", line)[-1])
                    print(data)
                    
# parse the results from cublas
for m, n, k in [
        [1,14336,57344],
        [32,14336,57344],
        [4096,14336,57344],
        [1,8192,28672],
        [32,8192,28672],
        [4096,8192,28672],
    ]:
    log_path = f"./cublasgemm-benchmark/build/cublas_shape_{m}_{k}_{n}_performance.csv"
    if not os.path.exists(log_path):
        continue
    data = get_and_print(log_path, "min_fp16_tensorcore_runtime")
    print(data)

# parse the results from ladder
for m, n, k in [
        [1,14336,57344],
        [32,14336,57344],
        [4096,14336,57344],
        [1,8192,28672],
        [32,8192,28672],
        [4096,8192,28672],
    ]:
    log_path = f"./ladder-benchmark/build/ladder_shape_{m}_{k}_{n}_performance.csv"
    if not os.path.exists(log_path):
        continue
    data = get_and_print(log_path, "min_fp16_tensorcore_runtime")
    print(data)


# parse the results from cudnn
for m, n, k in [
        [1,14336,57344],
        [32,14336,57344],
        [4096,14336,57344],
        [1,8192,28672],
        [32,8192,28672],
        [4096,8192,28672],
    ]:
    log_path = f"./cudnn-benchmark/build/cudnn_shape_{m}_{k}_{n}_performance.csv"
    if not os.path.exists(log_path):
        continue
    data = get_and_print(log_path, "min_fp16_tensorcore_runtime")
    print(data)

# write the results to back
reproduced_results = f"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

matmul_providers = {matmul_providers}
matmul_times_data = {matmul_times_data}

conv_providers = {conv_providers}
conv_times_data = {conv_times_data}
"""

with open("reproduce_result/__init__.py", "w") as f:
    f.write(reproduced_results)
