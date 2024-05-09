import os
import json
import re

_ = '''
# paper result
matmul_providers = ["M0", "M1", "M2", "M3", "M4", "M5"]
matmul_times_data = [
    ("cuBLAS", [1.017856, 1.1241472, 31.240396, 0.2287616, 0.2945024, 8.7457794]),
    (
        "cuTLASS-W$_{INT4}$A$_{FP16}$",
        [0.674009323, 1.186704636, 33.67717266, 0.153660774, 0.259065628, 12.6046657],
    ),
    (
        "vLLM-W$_{INT4}$A$_{FP16}$",
        [0.484972, 0.972840786, 123.6705709, 0.18430233, 0.27905941, 29.32891846],
    ),
    (
        "Bitter",
        [0.935731232, 1.050994396, 26.89023972, 0.270745605, 0.38573581, 7.485508442],
    ),
    (
        "Bitter-W$_{INT4}$A$_{FP16}$",
        [0.258867204, 0.99830687, 24.94899178, 0.079725713, 0.36928492, 6.955895424],
    ),
    (
        "Bitter-W$_{NF4}$A$_{FP16}$",
        [0.418611199, 1.114526272, 30.12454414, 0.125337601, 0.415465951, 8.341504097],
    ),
    (
        "Bitter-W$_{FP8}$A$_{FP16}$",
        [0.485785574, 0.944679379, 25.32633591, 0.143359989, 0.38356927, 7.078502178],
    ),
    (
        "Bitter-W$_{INT1}$A$_{INT8}$",
        [0.083967999, 0.530721366, 16.49015427, 0.0305152, 0.208530769, 4.851322174],
    ),
    (
        "Bitter-W$_{MXFP8}$A$_{MXFP8}$",
        [0.702719986, 1.678665757, 48.04956055, 0.214783996, 0.617724717, 15.08615303],
    ),
]


conv_providers = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
conv_times_data = [
    (
        "Bitter",
        [
            0.175513595,
            0.065536,
            0.143359989,
            0.090112001,
            0.017695315,
            0.00489919,
            0.026812863,
            0.010390706,
        ],
    ),
    (
        "Bitter-W$_{FP8}$A$_{FP16}$",
        [
            0.195993602,
            0.065536,
            0.164659202,
            0.090521596,
            0.01959509,
            0.004884709,
            0.034379646,
            0.013394201,
        ],
    ),
    (
        "Bitter-W$_{MXFP8}$A$_{MXFP8}$",
        [
            0.208281592,
            0.066150397,
            0.200294405,
            0.094617598,
            0.035001777,
            0.006058795,
            0.043367449,
            0.020062251,
        ],
    ),
    (
        "Bitter-W$_{INT4}$A$_{INT4}$",
        [
            0.0978944,
            0.074137598,
            0.082124799,
            0.125337601,
            0.055220462,
            0.009339727,
            0.171313092,
            0.061613843,
        ],
    ),
    (
        "cuDNN",
        [
            0.343142402,
            0.131583999,
            0.248934399,
            0.119091203,
            0.060006401,
            0.058163201,
            0.058572801,
            0.0657408,
        ],
    ),
    (
        "AMOS",
        [
            1.764688,
            0.302502,
            0.857058,
            0.3248,
            0.073223653,
            0.020407328,
            0.106228661,
            0.046321647,
        ],
    ),
    (
        "TensorIR",
        [
            0.2430228,
            0.0932777,
            0.216,
            0.0802,
            0.018251371,
            0.00476218,
            0.026268746,
            0.010746771,
        ],
    ),
]
'''
exec(_)

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
