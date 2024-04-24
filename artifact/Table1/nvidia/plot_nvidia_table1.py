# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import re
import argparse
from prettytable import PrettyTable
from paper_result import (
    mi250_res,
    v100_res,
    a100_res,
)
# add device argument

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="A100", choices=["A100", "V100", "A6000"])
parser.add_argument("--reproduce", action="store_true")
args = parser.parse_args()
device = args.device
reproduce = args.reproduce

PEAK_TFLOPS = {
    "A100": {
        "W$_{FP16}$A$_{FP16}$": 312,
        "W$_{INT8}$A$_{INT8}$": 624,
    },
    "V100": {
        "W$_{FP16}$A$_{FP16}$": 112,
    },
    "MI250": {
        "W$_{FP16}$A$_{FP16}$": 181,
        "W$_{INT8}$A$_{INT8}$": 181,
    },
}

nvidia_res = eval(f"{device.lower()}_res")

def latency_to_tflops(latency, M, N, K):
    return 2*M*N*K/latency/1000/1000/1000

def extract_cublas_perf(log_path="./cublas-benchmark/build/cublas_benchmark.log"):
    lines = open(log_path).read()
    pattern = r"\d+,\d+,\d+,\d+,\d+,(\d+\.\d+),(\d+\.\d+)"
    matches = re.search(pattern, lines)
    if matches:
        fp16_latency = matches.group(1)  # fp16 latency
        int8_latency = matches.group(2)  # int8 latency
        return fp16_latency, int8_latency
    else:
        raise ValueError("No match found in the cublas log file")

if reproduce:
    cublas_fp16_latency, cublas_int8_latency = extract_cublas_perf()
    cublas_fp16_tflops = latency_to_tflops(float(cublas_fp16_latency), 16384, 16384, 16384)
    cublas_int8_tflops = latency_to_tflops(float(cublas_int8_latency), 16384, 16384, 16384)
    cublas_fp16_percent = cublas_fp16_tflops / PEAK_TFLOPS[device]["W$_{FP16}$A$_{FP16}$"] * 100
    cublas_int8_percent = cublas_int8_tflops / PEAK_TFLOPS[device]["W$_{INT8}$A$_{INT8}$"] * 100

def extract_amos_fp16_perf(log_path="./amos-benchmark/gemm_nt_16384_float16.log"):
    with open(log_path, "r") as log_file:  
        log_data = log_file.read()  
  
    # Extract cost of gemm-float16-float16-layer  
    cost_pattern = r"Cost of gemm-nt-float16-float16-layer-\((\d+),\s(\d+),\s(\d+)\) is (\d+\.\d+) ms"  
    cost_matches = re.findall(cost_pattern, log_data)
    cost_match = cost_matches[-1]
    M, N, K = int(cost_match[0]), int(cost_match[1]), int(cost_match[2])
    cost = float(cost_match[3])
    return cost

if reproduce:
    amos_fp16_latency = extract_amos_fp16_perf()
    amos_fp16_tflops = latency_to_tflops(amos_fp16_latency, 16384, 16384, 16384)

    if device == "V100":
        nvidia_res["W$_{FP16}$A$_{FP16}$"]["cuBLAS"] = f"{cublas_fp16_percent:.0f}%"
    else:
        nvidia_res["W$_{FP16}$A$_{FP16}$"]["cuBLAS"] = f"{cublas_fp16_percent:.0f}%"
        nvidia_res["W$_{INT8}$A$_{INT8}$"]["cuBLAS"] = f"{cublas_int8_percent:.0f}%"

# initialize the figures
table = PrettyTable()
table.title = f"Performance Overview - {device}"
table.field_names = ["Library", "W$_{FP16}$A$_{FP16}$", "W$_{INT8}$A$_{INT8}$", "W$_{FP8}$A$_{FP8}$", "W$_{NF4}$A$_{FP16}$"]

# collect the transposed data
transposed_data = {key: [] for key in table.field_names[1:]}  # 初始化库对应的列表
libraries = ["cuBLAS", "rocBLAS", "AMOS", "TensorIR", "Roller"]

for lib in libraries:
    row = [lib]
    for precision in table.field_names[1:]:
        row.append(nvidia_res[precision].get(lib, 'x'))
    table.add_row(row)

print(table)