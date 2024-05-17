import os
import json
import re

b1s1_llama2_providers = ['W$_{FP16}$A$_{FP16}$', 'W$_{INT4}$A$_{FP16}$', 'W$_{MXFP8}$A$_{MXFP8}$', 'W$_{INT1}$A$_{INT8}$']
b1s1_llama2_times_data = [
    ('Welder-Roller', [-1, 0, 0, 0]), 
    ('+Transform', [-1, -1, -1, -1]), 
    ('+PTX', [-1, -1, -1, -1]), 
    ('+Holistic Schedule', [-1, -1, -1, -1])
]

b1s4096_llama2_providers = ['W$_{FP16}$A$_{FP16}$', 'W$_{INT4}$A$_{FP16}$', 'W$_{MXFP8}$A$_{MXFP8}$', 'W$_{INT1}$A$_{INT8}$']
b1s4096_llama2_times_data = [
    ('Welder-Roller', [-1, -1, -1, -1]),
    ('+Transform', [-1, -1, -1, -1]),
    ('+PTX', [-1, -1, -1, -1]),
    ('+Holistic Schedule', [-1, -1, -1, -1])
]
## update ladder results
def parse_ladder_logs(log):
    pattern = r"[\d]+\.[\d]+"
    data = None
    if not os.path.exists(log):
        return data
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

# transform
'''
transform_llama2-70b_b1_s1_q-1.log
transform_llama2-70b_b1_s1_q0_b1_int.log
transform_llama2-70b_b1_s1_q0_b4.log
transform_llama2-70b_b1_s1_q0_mxfp8.log
transform_llama2-70b_b1_s4096_q-1.log
transform_llama2-70b_b1_s4096_q0_b1_int.log
transform_llama2-70b_b1_s4096_q0_b4.log
transform_llama2-70b_b1_s4096_q0_fp_mxfp8.log
'''

def extract_floats(line):
    pattern = r"\b\d+\.\d+\b"
    return re.findall(pattern, line)

# it's extract from logs
def get_latency(log):
    with open(log, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "Summary:" in line:
            matches = extract_floats(line)
            if len(matches) == 0:
                raise ValueError(f"Could not find latency in line: {line}")
            latency = float(matches[-1])
            break
    if latency is None:
        raise ValueError(f"Could not find latency for {log}")
    else:
        print(f"Found latency for {log}: {latency}")
    return latency

llama_b1s1_fp16xfp16_roller_latency = get_latency('./welder-roller-end2end/compiled_models/llama2_70b_layer1_seq1_bs1_cutlass/run.log')
llama_b1s4096_fp16xfp16_roller_latency = get_latency('./welder-roller-end2end/compiled_models/llama2_70b_layer1_seq4096_bs1_cutlass/run.log')

transform_fp16_b1s1 = parse_ladder_logs("ladder-end2end/transform_llama2-70b_b1_s1_q-1.log")
transform_int1_b1s1 = parse_ladder_logs("ladder-end2end/transform_llama2-70b_b1_s1_q0_b1_int.log")
transform_int4_b1s1 = parse_ladder_logs("ladder-end2end/transform_llama2-70b_b1_s1_q0_b4.log")
transform_mxfp8_b1s1 = parse_ladder_logs("ladder-end2end/transform_llama2-70b_b1_s1_q0_mxfp8.log")

transform_fp16_b1s4096 = parse_ladder_logs("ladder-end2end/transform_llama2-70b_b1_s4096_q-1.log")
transform_int1_b1s4096 = parse_ladder_logs("ladder-end2end/transform_llama2-70b_b1_s4096_q0_b1_int.log")
transform_int4_b1s4096 = parse_ladder_logs("ladder-end2end/transform_llama2-70b_b1_s4096_q0_b4.log")
transform_mxfp8_b1s4096 = parse_ladder_logs("ladder-end2end/transform_llama2-70b_b1_s4096_q0_fp_mxfp8.log")

ptx_fp16_b1s1 = parse_ladder_logs("ladder-end2end/ptx_llama2-70b_b1_s1_q-1.log")
ptx_int1_b1s1 = parse_ladder_logs("ladder-end2end/ptx_llama2-70b_b1_s1_q0_b1_int.log")
ptx_int4_b1s1 = parse_ladder_logs("ladder-end2end/ptx_llama2-70b_b1_s1_q0_b4.log")
ptx_mxfp8_b1s1 = parse_ladder_logs("ladder-end2end/ptx_llama2-70b_b1_s1_q0_mxfp8.log")

ptx_fp16_b1s4096 = parse_ladder_logs("ladder-end2end/ptx_llama2-70b_b1_s4096_q-1.log")
ptx_int1_b1s4096 = parse_ladder_logs("ladder-end2end/ptx_llama2-70b_b1_s4096_q0_b1_int.log")
ptx_int4_b1s4096 = parse_ladder_logs("ladder-end2end/ptx_llama2-70b_b1_s4096_q0_b4.log")
ptx_mxfp8_b1s4096 = parse_ladder_logs("ladder-end2end/ptx_llama2-70b_b1_s4096_q0_fp_mxfp8.log")

holistic_fp16_b1s1 = parse_ladder_logs("ladder-end2end/holistic_llama2-70b_b1_s1_q-1.log")
holistic_int1_b1s1 = parse_ladder_logs("ladder-end2end/holistic_llama2-70b_b1_s1_q0_b1_int.log")
holistic_int4_b1s1 = parse_ladder_logs("ladder-end2end/holistic_llama2-70b_b1_s1_q0_b4.log")
holistic_mxfp8_b1s1 = parse_ladder_logs("ladder-end2end/holistic_llama2-70b_b1_s1_q0_mxfp8.log")

holistic_fp16_b1s4096 = parse_ladder_logs("ladder-end2end/holistic_llama2-70b_b1_s4096_q-1.log")
holistic_int1_b1s4096 = parse_ladder_logs("ladder-end2end/holistic_llama2-70b_b1_s4096_q0_b1_int.log")
holistic_int4_b1s4096 = parse_ladder_logs("ladder-end2end/holistic_llama2-70b_b1_s4096_q0_b4.log")
holistic_mxfp8_b1s4096 = parse_ladder_logs("ladder-end2end/holistic_llama2-70b_b1_s4096_q0_fp_mxfp8.log")

b1s1_llama2_times_data = [
    ('Welder-Roller', [llama_b1s1_fp16xfp16_roller_latency, 0, 0, 0]), 
    ('+Transform', [transform_fp16_b1s1, transform_int4_b1s1, transform_mxfp8_b1s1, transform_int1_b1s1]), 
    ('+PTX', [ptx_fp16_b1s1, ptx_int4_b1s1, ptx_mxfp8_b1s1, ptx_int1_b1s1]), 
    ('+Holistic Schedule', [holistic_fp16_b1s1, holistic_int4_b1s1, holistic_mxfp8_b1s1, holistic_int1_b1s1])
]

b1s4096_llama2_times_data = [
    ('Welder-Roller', [llama_b1s4096_fp16xfp16_roller_latency, 0, 0, 0]),
    ('+Transform', [transform_fp16_b1s4096, transform_int4_b1s4096, transform_mxfp8_b1s4096, transform_int1_b1s4096]),
    ('+PTX', [ptx_fp16_b1s4096, ptx_int4_b1s4096, ptx_mxfp8_b1s4096, ptx_int1_b1s4096]),
    ('+Holistic Schedule', [holistic_fp16_b1s4096, holistic_int4_b1s4096, holistic_mxfp8_b1s4096, holistic_int1_b1s4096])
]


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
