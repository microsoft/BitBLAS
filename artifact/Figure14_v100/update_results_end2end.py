import os
import json
import re

from reproduce_result import (
    b1s1_providers,
    b1s1_times_data,
    b1s4096_providers,
    b1s4096_times_data,
    b1s1_matmul_providers,
    b1s1_matmul_times_data,
    b1s4096_matmul_providers,
    b1s4096_matmul_times_data,
)

## update ladder results
def parse_ladder_logs(log):
    pattern = r"[\d]+\.[\d]+"
    data = None
    if not os.path.exists(log):
        raise FileNotFoundError(f"File {log} not found")
    with open(log, 'r') as f:
        lines = f.readlines()
        is_next_line=False
        for line in lines:
            if 'mean (ms)' in line:
                is_next_line = True
            if is_next_line:
                matches = re.findall(pattern, line)
                if matches:
                    data = float(matches[0])
                    is_next_line = False
    if data is not None:
        print(f"Ladder data from {log} is {data}")

    return data

"""
llama2-70b_b1_s1_fp16.log
llama2-70b_b1_s1_fp16xint1.log
llama2-70b_b1_s1_fp16xint2.log
llama2-70b_b1_s1_fp16xint4.log
llama2-70b_b1_s1_fp16xint8.log
llama2-70b_b1_s1_int4xint1_int.log
llama2-70b_b1_s1_int4xint2_int.log
llama2-70b_b1_s1_int4xint4_int.log
llama2-70b_b1_s1_int8xint2_int.log
llama2-70b_b1_s1_int8xint4_int.log
llama2-70b_b1_s1_int8xint8_int.log
llama2-70b_b1_s4096_fp16.log
llama2-70b_b1_s4096_fp16xint1.log
llama2-70b_b1_s4096_fp16xint2.log
llama2-70b_b1_s4096_fp16xint4.log
llama2-70b_b1_s4096_fp16xint8.log
llama2-70b_b1_s4096_int8xint1_int.log
llama2-70b_b1_s4096_int8xint2_int.log
llama2-70b_b1_s4096_int8xint4_int.log
llama2-70b_b1_s4096_int8xint8_int.log
"""
ladder_llama_b1s1_fp16xfp16_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s1_fp16.log")
ladder_llama_b1s1_fp16xint8_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s1_fp16xint8.log")
ladder_llama_b1s1_fp16xint4_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s1_fp16xint4.log")
ladder_llama_b1s1_fp16xint2_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s1_fp16xint2.log")
ladder_llama_b1s1_fp16xint1_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s1_fp16xint1.log")
ladder_llama_b1s1_int8xint8_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s1_int8xint8_int.log")
ladder_llama_b1s1_int8xint4_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s1_int8xint4_int.log")
ladder_llama_b1s1_int8xint2_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s1_int8xint2_int.log")
ladder_llama_b1s1_int8xint1_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s1_int8xint1_int.log")
ladder_llama_b1s1_int4xint4_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s1_int4xint4_int.log")
ladder_llama_b1s1_int4xint2_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s1_int4xint2_int.log")
ladder_llama_b1s1_int4xint1_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s1_int4xint1_int.log")

ladder_llama_b1s4096_fp16xfp16_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s4096_fp16.log")
ladder_llama_b1s4096_fp16xint8_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s4096_fp16xint8.log")
ladder_llama_b1s4096_fp16xint4_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s4096_fp16xint4.log")
ladder_llama_b1s4096_fp16xint2_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s4096_fp16xint2.log")
ladder_llama_b1s4096_fp16xint1_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s4096_fp16xint1.log")
ladder_llama_b1s4096_int8xint8_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s4096_int8xint8_int.log")
ladder_llama_b1s4096_int8xint4_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s4096_int8xint4_int.log")
ladder_llama_b1s4096_int8xint2_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s4096_int8xint2_int.log")
ladder_llama_b1s4096_int8xint1_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s4096_int8xint1_int.log")
ladder_llama_b1s4096_int4xint4_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s4096_int4xint4_int.log")
ladder_llama_b1s4096_int4xint2_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s4096_int4xint2_int.log")
ladder_llama_b1s4096_int4xint1_latency = parse_ladder_logs("ladder-benchmark/llama2-70b_b1_s4096_int4xint1_int.log")

b1s1_times_data = [
    ("Bitter", [ladder_llama_b1s1_fp16xfp16_latency]),
    ("Bitter-W$_{INT8}$A$_{FP16}$", [ladder_llama_b1s1_fp16xint8_latency]),
    ("Bitter-W$_{INT4}$A$_{FP16}$", [ladder_llama_b1s1_fp16xint4_latency]),
    ("Bitter-W$_{INT2}$A$_{FP16}$", [ladder_llama_b1s1_fp16xint2_latency]),
    ("Bitter-W$_{INT1}$A$_{FP16}$", [ladder_llama_b1s1_fp16xint1_latency]),
    ("Bitter-W$_{INT8}$A$_{INT8}$", [ladder_llama_b1s1_int8xint8_latency]),
    ("Bitter-W$_{INT4}$A$_{INT8}$", [ladder_llama_b1s1_int8xint4_latency]),
    ("Bitter-W$_{INT2}$A$_{INT8}$", [ladder_llama_b1s1_int8xint2_latency]),
    ("Bitter-W$_{INT1}$A$_{INT8}$", [ladder_llama_b1s1_int8xint1_latency]),
    ("Bitter-W$_{INT4}$A$_{INT4}$", [ladder_llama_b1s1_int4xint4_latency]),
    ("Bitter-W$_{INT2}$A$_{INT4}$", [ladder_llama_b1s1_int4xint2_latency]),
    ("Bitter-W$_{INT1}$A$_{INT4}$", [ladder_llama_b1s1_int4xint1_latency]),
]

b1s4096_times_data = [
    ("Bitter", [ladder_llama_b1s4096_fp16xfp16_latency]),
    ("Bitter-W$_{INT8}$A$_{FP16}$", [ladder_llama_b1s4096_fp16xint8_latency]),
    ("Bitter-W$_{INT4}$A$_{FP16}$", [ladder_llama_b1s4096_fp16xint4_latency]),
    ("Bitter-W$_{INT2}$A$_{FP16}$", [ladder_llama_b1s4096_fp16xint2_latency]),
    ("Bitter-W$_{INT1}$A$_{FP16}$", [ladder_llama_b1s4096_fp16xint1_latency]),
    ("Bitter-W$_{INT8}$A$_{INT8}$", [ladder_llama_b1s4096_int8xint8_latency]),
    ("Bitter-W$_{INT4}$A$_{INT8}$", [ladder_llama_b1s4096_int8xint4_latency]),
    ("Bitter-W$_{INT2}$A$_{INT8}$", [ladder_llama_b1s4096_int8xint2_latency]),
    ("Bitter-W$_{INT1}$A$_{INT8}$", [ladder_llama_b1s4096_int8xint1_latency]),
    ("Bitter-W$_{INT4}$A$_{INT4}$", [ladder_llama_b1s4096_int4xint4_latency]),
    ("Bitter-W$_{INT2}$A$_{INT4}$", [ladder_llama_b1s4096_int4xint2_latency]),
    ("Bitter-W$_{INT1}$A$_{INT4}$", [ladder_llama_b1s4096_int4xint1_latency]),
]

reproduced_results = f"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
b1s1_providers = {b1s1_providers}
b1s1_times_data = {b1s1_times_data}

b1s4096_providers = {b1s4096_providers}
b1s4096_times_data = {b1s4096_times_data}

b1s1_matmul_providers = {b1s1_matmul_providers}
b1s1_matmul_times_data = {b1s1_matmul_times_data}

b1s4096_matmul_providers = {b1s4096_matmul_providers}
b1s4096_matmul_times_data = {b1s4096_matmul_times_data}

"""

with open("reproduce_result/__init__.py", "w") as f:
    f.write(reproduced_results)
