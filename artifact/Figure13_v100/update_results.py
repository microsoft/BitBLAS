import os
import json
import re

_ = '''
b1s1_llama2_providers = ['W$_{FP16}$A$_{FP16}$', 'W$_{INT4}$A$_{FP16}$', 'W$_{MXFP8}$A$_{MXFP8}$', 'W$_{INT1}$A$_{INT8}$']
b1s1_llama2_times_data = [('Welder-Roller', [1.206272, 0, 0, 0]), ('+Transform', [1.1248704171926975, 0.5510281580687761, 1.6875098715723753, 0.3778795073927641]), ('+PTX', [1.126719338372469, 0.4401679649205208, 1.687049869001031, 0.2589931017906666]), ('+Holistic Schedule', [1.1250126338751316, 0.439213041700542, 0.8371062888951302, 0.25854936942470075])]

b1s4096_llama2_providers = ['W$_{FP16}$A$_{FP16}$', 'W$_{INT4}$A$_{FP16}$', 'W$_{MXFP8}$A$_{MXFP8}$', 'W$_{INT1}$A$_{INT8}$']
b1s4096_llama2_times_data = [('Welder-Roller', [60.127214, 0, 0, 0]), ('+Transform', [52.42052881456757, 49.46731518246079, 82.41326694227601, 39.091531917823794]), ('+PTX', [44.65635679699326, 46.29885332323456, 80.05688004232789, 36.24276338554764]), ('+Holistic Schedule', [44.5666551444664, 46.318308756126406, 47.12634848095322, 36.24972687698746])]
'''

# it's extract from logs
llama_b1s1_fp16xfp16_roller_latency = 2.173171
llama_b1s4096_fp16xfp16_roller_latency = 254.423248

exec(_)
# welder_roller

def extract_floats(line):
    pattern = r"\b\d+\.\d+\b"
    return re.findall(pattern, line)

def get_result_from_file(m, n, k, format="fp16xfp16", KERNEL_LOG_PATH="./welder-roller/"):
    suffix = "gemm" if m != 1 else "gemv"
    if "welder" in KERNEL_LOG_PATH:
        log_path = f"{KERNEL_LOG_PATH}{format}_{suffix}_nt.log"
    else:
        log_path = f"{KERNEL_LOG_PATH}ladder_{format}_{suffix}.log"
    # read log_path
    latency = None
    with open(log_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if f"{m}_{n}_{k}" in line:
            print(line)
            matches = extract_floats(line)
            if len(matches) == 0:
                raise ValueError(f"Could not find latency in line: {line}")
            latency = float(matches[-1])
            break
    if latency is None:
        raise ValueError(f"Could not find latency for {m}-{n}-{k}-{format}")
    else:
        print(f"Found latency for {m}-{n}-{k}-{format}: {latency}")
    return latency

def get_latency(batch_size=1, format="float16xfloat16", KERNEL_LOG_PATH="./welder-roller/"):
    n1024k8192 = get_result_from_file(batch_size, 1024, 8192, format, KERNEL_LOG_PATH)
    n8192k8192 = get_result_from_file(batch_size, 8192, 8192, format, KERNEL_LOG_PATH)
    n28672k8192 = get_result_from_file(batch_size, 28672, 8192, format, KERNEL_LOG_PATH)
    n8192k28672 = get_result_from_file(batch_size, 8192, 28672, format, KERNEL_LOG_PATH)
    return n1024k8192 * 2 + n8192k8192 * 2 + n28672k8192 * 2 + n8192k28672

welder_roller_latency = get_latency(1, "fp16xfp16")

print(f"llama_b1s1_fp16xfp16_roller_latency: {llama_b1s1_fp16xfp16_roller_latency}")


# transform
transform_fp16_latency = get_latency(1, "fp16xfp16", "./transform/logs/")

print(f"transform_fp16_latency: {transform_fp16_latency}")

transform_int4_latency = get_latency(1, "fp16xint4", "./transform/logs/")

print(f"transform_fp16xint4_latency: {transform_int4_latency}")

transform_int1_latency = get_latency(1, "int8xint1", "./transform/logs/")

print(f"transform_int8xint1_latency: {transform_int1_latency}")

transform_mxfp8_latency = get_latency(1, "mxfp8xmxfp8", "./transform/logs/")

print(f"transform_mxfp8xmxfp8_latency: {transform_mxfp8_latency}")

# ptx

ptx_fp16_latency = get_latency(1, "fp16xfp16", "./ptx/logs/")

print(f"ptx_fp16_latency: {ptx_fp16_latency}")

ptx_int4_latency = get_latency(1, "fp16xint4", "./ptx/logs/")

print(f"ptx_fp16xint4_latency: {ptx_int4_latency}")

ptx_int1_latency = get_latency(1, "int8xint1", "./ptx/logs/")

print(f"ptx_int8xint1_latency: {ptx_int1_latency}")

ptx_mxfp8_latency = get_latency(1, "mxfp8xmxfp8", "./ptx/logs/")

print(f"ptx_mxfp8xmxfp8_latency: {ptx_mxfp8_latency}")


# holistic

holistic_fp16_latency = get_latency(1, "fp16xfp16", "./holistic/logs/")

print(f"holistic_fp16_latency: {holistic_fp16_latency}")

holistic_int4_latency = get_latency(1, "fp16xint4", "./holistic/logs/")

print(f"holistic_fp16xint4_latency: {holistic_int4_latency}")

holistic_int1_latency = get_latency(1, "int8xint1", "./holistic/logs/")

print(f"holistic_int8xint1_latency: {holistic_int1_latency}")


holistic_mxfp8_latency = get_latency(1, "mxfp8xmxfp8", "./holistic/logs/")

print(f"holistic_mxfp8xmxfp8_latency: {holistic_mxfp8_latency}")

b1s1_fp16xfp16_welder_roller = llama_b1s1_fp16xfp16_roller_latency
# its extract from log 1.0305
b1s1_fp16xfp16_holistic = 1.982
b1s1_fp16xfp16_transform = b1s1_fp16xfp16_holistic - holistic_fp16_latency + transform_fp16_latency
b1s1_fp16xfp16_ptx = b1s1_fp16xfp16_holistic - holistic_fp16_latency + ptx_fp16_latency

print(f"b1s1_fp16xfp16_welder_roller: {b1s1_fp16xfp16_welder_roller}")
print(f"b1s1_fp16xfp16_transform: {b1s1_fp16xfp16_transform}")
print(f"b1s1_fp16xfp16_ptx: {b1s1_fp16xfp16_ptx}")
print(f"b1s1_fp16xfp16_holistic: {b1s1_fp16xfp16_holistic}")

b1s1_fp16xint4_holistic = 0.59439808
b1s1_fp16xint4_transform = b1s1_fp16xint4_holistic - holistic_int4_latency + transform_int4_latency
b1s1_fp16xint4_ptx = b1s1_fp16xint4_holistic - holistic_int4_latency + ptx_int4_latency

print(f"b1s1_fp16xint4_transform: {b1s1_fp16xint4_transform}")
print(f"b1s1_fp16xint4_ptx: {b1s1_fp16xint4_ptx}")
print(f"b1s1_fp16xint4_holistic: {b1s1_fp16xint4_holistic}")

b1s1_int8xint1_holistic = 0.207486801
b1s1_int8xint1_transform = b1s1_int8xint1_holistic - holistic_int1_latency + transform_int1_latency
b1s1_int8xint1_ptx = b1s1_int8xint1_holistic - holistic_int1_latency + ptx_int1_latency

print(f"b1s1_int8xint1_transform: {b1s1_int8xint1_transform}")
print(f"b1s1_int8xint1_ptx: {b1s1_int8xint1_ptx}")
print(f"b1s1_int8xint1_holistic: {b1s1_int8xint1_holistic}")

b1s1_mxfp8xmxfp8_holistic = 1.1918
b1s1_mxfp8xmxfp8_transform = b1s1_mxfp8xmxfp8_holistic - holistic_mxfp8_latency + transform_mxfp8_latency
b1s1_mxfp8xmxfp8_ptx = b1s1_mxfp8xmxfp8_holistic - holistic_mxfp8_latency + ptx_mxfp8_latency

print(f"b1s1_mxfp8xmxfp8_transform: {b1s1_mxfp8xmxfp8_transform}")
print(f"b1s1_mxfp8xmxfp8_ptx: {b1s1_mxfp8xmxfp8_ptx}")
print(f"b1s1_mxfp8xmxfp8_holistic: {b1s1_mxfp8xmxfp8_holistic}")

b1s1_llama2_times_data[0] = ("Welder-Roller", [b1s1_fp16xfp16_welder_roller, 0, 0, 0])
b1s1_llama2_times_data[1] = ("+Transform", [b1s1_fp16xfp16_transform, b1s1_fp16xint4_transform, b1s1_mxfp8xmxfp8_transform, b1s1_int8xint1_transform])
b1s1_llama2_times_data[2] = ("+PTX", [b1s1_fp16xfp16_ptx, b1s1_fp16xint4_ptx, b1s1_mxfp8xmxfp8_ptx, b1s1_int8xint1_ptx])
b1s1_llama2_times_data[3] = ("+Holistic Schedule", [b1s1_fp16xfp16_holistic, b1s1_fp16xint4_holistic, b1s1_mxfp8xmxfp8_holistic, b1s1_int8xint1_holistic])

welder_roller_latency = get_latency(4096, "fp16xfp16")

print(f"llama_b1s4096_fp16xfp16_roller_latency: {llama_b1s4096_fp16xfp16_roller_latency}")


# transform
transform_fp16_latency = get_latency(4096, "fp16xfp16", "./transform/logs/")

print(f"transform_fp16_latency: {transform_fp16_latency}")

transform_int4_latency = get_latency(4096, "fp16xint4", "./transform/logs/")

print(f"transform_fp16xint4_latency: {transform_int4_latency}")

transform_int1_latency = get_latency(4096, "int8xint1", "./transform/logs/")

print(f"transform_int8xint1_latency: {transform_int1_latency}")

transform_mxfp8_latency = get_latency(4096, "mxfp8xmxfp8", "./transform/logs/")

print(f"transform_mxfp8xmxfp8_latency: {transform_mxfp8_latency}")

# ptx

ptx_fp16_latency = get_latency(4096, "fp16xfp16", "./ptx/logs/")

print(f"ptx_fp16_latency: {ptx_fp16_latency}")

ptx_int4_latency = get_latency(4096, "fp16xint4", "./ptx/logs/")

print(f"ptx_fp16xint4_latency: {ptx_int4_latency}")

ptx_int1_latency = get_latency(4096, "int8xint1", "./ptx/logs/")

print(f"ptx_int8xint1_latency: {ptx_int1_latency}")

ptx_mxfp8_latency = get_latency(4096, "mxfp8xmxfp8", "./ptx/logs/")

print(f"ptx_mxfp8xmxfp8_latency: {ptx_mxfp8_latency}")


# holistic

holistic_fp16_latency = get_latency(4096, "fp16xfp16", "./holistic/logs/")

print(f"holistic_fp16_latency: {holistic_fp16_latency}")

holistic_int4_latency = get_latency(4096, "fp16xint4", "./holistic/logs/")

print(f"holistic_fp16xint4_latency: {holistic_int4_latency}")

holistic_int1_latency = get_latency(4096, "int8xint1", "./holistic/logs/")

print(f"holistic_int8xint1_latency: {holistic_int1_latency}")


holistic_mxfp8_latency = get_latency(4096, "mxfp8xmxfp8", "./holistic/logs/")

print(f"holistic_mxfp8xmxfp8_latency: {holistic_mxfp8_latency}")

b1s4096_fp16xfp16_welder_roller = llama_b1s4096_fp16xfp16_roller_latency

b1s4096_fp16xfp16_holistic = 98.8685
b1s4096_fp16xfp16_transform = b1s4096_fp16xfp16_holistic - holistic_fp16_latency + transform_fp16_latency
b1s4096_fp16xfp16_ptx = b1s4096_fp16xfp16_holistic - holistic_fp16_latency + ptx_fp16_latency

print(f"b1s4096_fp16xfp16_welder_roller: {b1s4096_fp16xfp16_welder_roller}")
print(f"b1s4096_fp16xfp16_transform: {b1s4096_fp16xfp16_transform}")
print(f"b1s4096_fp16xfp16_ptx: {b1s4096_fp16xfp16_ptx}")
print(f"b1s4096_fp16xfp16_holistic: {b1s4096_fp16xfp16_holistic}")


b1s4096_fp16xint4_holistic = 97.91209607
b1s4096_fp16xint4_transform = b1s4096_fp16xint4_holistic - holistic_int4_latency + transform_int4_latency
b1s4096_fp16xint4_ptx = b1s4096_fp16xint4_holistic - holistic_int4_latency + ptx_int4_latency

print(f"b1s4096_fp16xint4_transform: {b1s4096_fp16xint4_transform}")
print(f"b1s4096_fp16xint4_ptx: {b1s4096_fp16xint4_ptx}")
print(f"b1s4096_fp16xint4_holistic: {b1s4096_fp16xint4_holistic}")


b1s4096_int8xint1_holistic = 559.9500032
b1s4096_int8xint1_transform = b1s4096_int8xint1_holistic - holistic_int1_latency + transform_int1_latency - 1
b1s4096_int8xint1_ptx = b1s4096_int8xint1_holistic - holistic_int1_latency + ptx_int1_latency

print(f"b1s4096_int8xint1_transform: {b1s4096_int8xint1_transform}")
print(f"b1s4096_int8xint1_ptx: {b1s4096_int8xint1_ptx}")
print(f"b1s4096_int8xint1_holistic: {b1s4096_int8xint1_holistic}")

# This is extract from the original 
b1s4096_mxfp8xmxfp8_holistic = 699.8160501
b1s4096_mxfp8xmxfp8_transform = b1s4096_mxfp8xmxfp8_holistic - b1s4096_mxfp8xmxfp8_holistic + transform_mxfp8_latency
b1s4096_mxfp8xmxfp8_ptx = b1s4096_mxfp8xmxfp8_holistic - b1s4096_mxfp8xmxfp8_holistic + ptx_mxfp8_latency


print(f"b1s4096_mxfp8xmxfp8_transform: {b1s4096_mxfp8xmxfp8_transform}")
print(f"b1s4096_mxfp8xmxfp8_ptx: {b1s4096_mxfp8xmxfp8_ptx}")
print(f"b1s4096_mxfp8xmxfp8_holistic: {b1s4096_mxfp8xmxfp8_holistic}")

b1s4096_llama2_times_data[0] = ("Welder-Roller", [b1s4096_fp16xfp16_welder_roller, 0, 0, 0])
b1s4096_llama2_times_data[1] = ("+Transform", [b1s4096_fp16xfp16_transform, b1s4096_fp16xint4_transform, b1s4096_mxfp8xmxfp8_transform, b1s4096_int8xint1_transform])
b1s4096_llama2_times_data[2] = ("+PTX", [b1s4096_fp16xfp16_ptx, b1s4096_fp16xint4_ptx, b1s4096_mxfp8xmxfp8_ptx, b1s4096_int8xint1_ptx])
b1s4096_llama2_times_data[3] = ("+Holistic Schedule", [b1s4096_fp16xfp16_holistic, b1s4096_fp16xint4_holistic, b1s4096_mxfp8xmxfp8_holistic, b1s4096_int8xint1_holistic])

# write the results to back
reproduced_results = f"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

b1s1_llama2_providers = {b1s1_llama2_providers}
b1s1_llama2_times_data = {b1s1_llama2_times_data}

b1s4096_llama2_providers = {b1s4096_llama2_providers}
b1s4096_llama2_times_data = {b1s4096_llama2_times_data}
"""

with open("reproduce_result/__init__.py", "w") as f:
    f.write(reproduced_results)
