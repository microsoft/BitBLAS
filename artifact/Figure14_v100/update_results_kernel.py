import os
import json
import re

_ = '''
b1s1_providers = ['End2End LLAMA']
b1s1_times_data = [('Bitter', [1.0305]), ('Bitter-W$_{INT8}$A$_{FP16}$', [0.5924999999999999]), ('Bitter-W$_{INT4}$A$_{FP16}$', [0.32849999999999996]), ('Bitter-W$_{INT2}$A$_{FP16}$', [0.2895]), ('Bitter-W$_{INT1}$A$_{FP16}$', [0.2815]), ('Bitter-W$_{INT8}$A$_{INT8}$', [0.6004999999999999]), ('Bitter-W$_{INT4}$A$_{INT8}$', [0.3175]), ('Bitter-W$_{INT2}$A$_{INT8}$', [0.19149999999999998]), ('Bitter-W$_{INT1}$A$_{INT8}$', [0.15849999999999997]), ('Bitter-W$_{INT4}$A$_{INT4}$', [0.3145]), ('Bitter-W$_{INT2}$A$_{INT4}$', [0.19249999999999998]), ('Bitter-W$_{INT1}$A$_{INT4}$', [0.15849999999999997])]

b1s4096_providers = ['End2End LLAMA']
b1s4096_times_data = [('Bitter', [33.7857]), ('Bitter-W$_{INT8}$A$_{FP16}$', [34.7037]), ('Bitter-W$_{INT4}$A$_{FP16}$', [32.58669999999999]), ('Bitter-W$_{INT2}$A$_{FP16}$', [33.5087]), ('Bitter-W$_{INT1}$A$_{FP16}$', [33.33369999999999]), ('Bitter-W$_{INT8}$A$_{INT8}$', [22.998699999999996]), ('Bitter-W$_{INT4}$A$_{INT8}$', [22.5307]), ('Bitter-W$_{INT2}$A$_{INT8}$', [22.005699999999997]), ('Bitter-W$_{INT1}$A$_{INT8}$', [22.2537]), ('Bitter-W$_{INT4}$A$_{INT4}$', [14.267433072996138]), ('Bitter-W$_{INT2}$A$_{INT4}$', [14.268457080793379]), ('Bitter-W$_{INT1}$A$_{INT4}$', [14.270300294828413])]

b1s1_matmul_providers = ['M0', 'M1', 'M2', 'M3']
b1s1_matmul_times_data = [('Bitter', [0.01, 0.081, 0.267, 0.271]), ('Bitter-W$_{INT8}$A$_{FP16}$', [0.01, 0.045, 0.143, 0.153]), ('Bitter-W$_{INT4}$A$_{FP16}$', [0.007, 0.024, 0.072, 0.079]), ('Bitter-W$_{INT2}$A$_{FP16}$', [0.007, 0.021, 0.062, 0.066]), ('Bitter-W$_{INT1}$A$_{FP16}$', [0.007, 0.021, 0.06, 0.062]), ('Bitter-W$_{INT8}$A$_{INT8}$', [0.01, 0.048, 0.148, 0.145]), ('Bitter-W$_{INT4}$A$_{INT8}$', [0.006, 0.021, 0.073, 0.074]), ('Bitter-W$_{INT2}$A$_{INT8}$', [0.006, 0.011, 0.038, 0.038]), ('Bitter-W$_{INT1}$A$_{INT8}$', [0.005, 0.011, 0.028, 0.027]), ('Bitter-W$_{INT4}$A$_{INT4}$', [0.006, 0.02, 0.073, 0.073]), ('Bitter-W$_{INT2}$A$_{INT4}$', [0.006, 0.011, 0.038, 0.039]), ('Bitter-W$_{INT1}$A$_{INT4}$', [0.005, 0.011, 0.028, 0.027])]

b1s4096_matmul_providers = ['M0', 'M1', 'M2', 'M3']
b1s4096_matmul_times_data = [('Bitter', [0.34, 2.104, 7.13, 8.457]), ('Bitter-W$_{INT8}$A$_{FP16}$', [0.35, 2.184, 7.785, 7.885]), ('Bitter-W$_{INT4}$A$_{FP16}$', [0.342, 2.07, 7.12, 7.342]), ('Bitter-W$_{INT2}$A$_{FP16}$', [0.343, 2.172, 7.396, 7.506]), ('Bitter-W$_{INT1}$A$_{FP16}$', [0.346, 2.165, 7.319, 7.493]), ('Bitter-W$_{INT8}$A$_{INT8}$', [0.204, 1.329, 4.544, 4.664]), ('Bitter-W$_{INT4}$A$_{INT8}$', [0.198, 1.297, 4.471, 4.418]), ('Bitter-W$_{INT2}$A$_{INT8}$', [0.194, 1.253, 4.331, 4.269]), ('Bitter-W$_{INT1}$A$_{INT8}$', [0.198, 1.278, 4.394, 4.333]), ('Bitter-W$_{INT4}$A$_{INT4}$', [0.11120639741420746, 0.6909952163696289, 2.0856833457946777, 2.3109631538391113]), ('Bitter-W$_{INT2}$A$_{INT4}$', [0.1114111989736557, 0.6901760101318359, 2.0865025520324707, 2.311577558517456]), ('Bitter-W$_{INT1}$A$_{INT4}$', [0.1114111989736557, 0.6901760101318359, 2.0873217582702637, 2.3117823600769043])]
'''
# get float16xfloat16_results
# analysis from the end2end process
bs1_fp16_time = 1.982
bs4096_fp16_time = 98.8685
KERNEL_LOG_PATH = "./kernel-benchmark/logs/"
exec(_)
def extract_floats(line):
    pattern = r"\b\d+\.\d+\b"
    return re.findall(pattern, line)

def get_result_from_file(m, n, k, format="float16xfloat16"):
    suffix = "gemm" if m != 1 else "gemv"
    log_path = f"{KERNEL_LOG_PATH}{format}_{suffix}.log"
    # read log_path
    latency = None
    with open(log_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if f"{m}-{n}-{k}" in line:
            matches = extract_floats(line)
            if len(matches) == 0:
                raise ValueError(f"Could not find latency in line: {line}")
            latency = float(matches[-2])
            break
    if latency is None:
        raise ValueError(f"Could not find latency for {m}-{n}-{k}-{format}")
    else:
        print(f"Found latency for {m}-{n}-{k}-{format}: {latency}")
    return latency

def get_latency(batch_size=1, format="float16xfloat16"):
    n1024k8192 = get_result_from_file(batch_size, 1024, 8192, format)
    n8192k8192 = get_result_from_file(batch_size, 8192, 8192, format)
    n28672k8192 = get_result_from_file(batch_size, 28672, 8192, format)
    n8192k28672 = get_result_from_file(batch_size, 8192, 28672, format)
    return n1024k8192 * 2 + n8192k8192 * 2 + n28672k8192 * 2 + n8192k28672

fp16xfp16_bs1_overall_latency = get_latency(1, "float16xfloat16")
fp16xint8_bs1_overall_latency = get_latency(1, "float16xint8")
fp16xint4_bs1_overall_latency = get_latency(1, "float16xint4")
fp16xint2_bs1_overall_latency = get_latency(1, "float16xint2")
fp16xint1_bs1_overall_latency = get_latency(1, "float16xint1")
int8xint8_bs1_overall_latency = get_latency(1, "int8xint8")
int8xint4_bs1_overall_latency = get_latency(1, "int8xint4")
int8xint2_bs1_overall_latency = get_latency(1, "int8xint2")
int8xint1_bs1_overall_latency = get_latency(1, "int8xint1")
int4xint4_bs1_overall_latency = get_latency(1, "int4xint4")
int4xint2_bs1_overall_latency = get_latency(1, "int4xint2")
int4xint1_bs1_overall_latency = get_latency(1, "int4xint1")

fp16xint8_bs1_end2end_latency = bs1_fp16_time - fp16xfp16_bs1_overall_latency + fp16xint8_bs1_overall_latency
fp16xint4_bs1_end2end_latency = bs1_fp16_time - fp16xfp16_bs1_overall_latency + fp16xint4_bs1_overall_latency
fp16xint2_bs1_end2end_latency = bs1_fp16_time - fp16xfp16_bs1_overall_latency + fp16xint2_bs1_overall_latency
fp16xint1_bs1_end2end_latency = bs1_fp16_time - fp16xfp16_bs1_overall_latency + fp16xint1_bs1_overall_latency
int8xint8_bs1_end2end_latency = bs1_fp16_time - fp16xfp16_bs1_overall_latency + int8xint8_bs1_overall_latency
int8xint4_bs1_end2end_latency = bs1_fp16_time - fp16xfp16_bs1_overall_latency + int8xint4_bs1_overall_latency
int8xint2_bs1_end2end_latency = bs1_fp16_time - fp16xfp16_bs1_overall_latency + int8xint2_bs1_overall_latency
int8xint1_bs1_end2end_latency = bs1_fp16_time - fp16xfp16_bs1_overall_latency + int8xint1_bs1_overall_latency
int4xint4_bs1_end2end_latency = bs1_fp16_time - fp16xfp16_bs1_overall_latency + int4xint4_bs1_overall_latency
int4xint2_bs1_end2end_latency = bs1_fp16_time - fp16xfp16_bs1_overall_latency + int4xint2_bs1_overall_latency
int4xint1_bs1_end2end_latency = bs1_fp16_time - fp16xfp16_bs1_overall_latency + int4xint1_bs1_overall_latency

b1s1_times_data[0] = ("Bitter", [bs1_fp16_time])
b1s1_times_data[1] = ("Bitter-W$_{INT8}$A$_{FP16}$", [fp16xint8_bs1_end2end_latency])
b1s1_times_data[2] = ("Bitter-W$_{INT4}$A$_{FP16}$", [fp16xint4_bs1_end2end_latency])
b1s1_times_data[3] = ("Bitter-W$_{INT2}$A$_{FP16}$", [fp16xint2_bs1_end2end_latency])
b1s1_times_data[4] = ("Bitter-W$_{INT1}$A$_{FP16}$", [fp16xint1_bs1_end2end_latency])
b1s1_times_data[5] = ("Bitter-W$_{INT8}$A$_{INT8}$", [int8xint8_bs1_end2end_latency])
b1s1_times_data[6] = ("Bitter-W$_{INT4}$A$_{INT8}$", [int8xint4_bs1_end2end_latency])
b1s1_times_data[7] = ("Bitter-W$_{INT2}$A$_{INT8}$", [int8xint2_bs1_end2end_latency])
b1s1_times_data[8] = ("Bitter-W$_{INT1}$A$_{INT8}$", [int8xint1_bs1_end2end_latency])
b1s1_times_data[9] = ("Bitter-W$_{INT4}$A$_{INT4}$", [int4xint4_bs1_end2end_latency])
b1s1_times_data[10] = ("Bitter-W$_{INT2}$A$_{INT4}$", [int4xint2_bs1_end2end_latency])
b1s1_times_data[11] = ("Bitter-W$_{INT1}$A$_{INT4}$", [int4xint1_bs1_end2end_latency])

def get_result_from_file_ladder(m, n, k, format="float16xfloat16"):
    suffix = "gemm" if m != 1 else "gemv"
    log_path = f"{KERNEL_LOG_PATH}{format}_{suffix}.log"
    # read log_path
    latency = None
    with open(log_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if f"{m}_{n}_{k}" in line:
            matches = extract_floats(line)
            if len(matches) == 0:
                raise ValueError(f"Could not find latency in line: {line}")
            latency = float(matches[0])
            break
    if latency is None:
        raise ValueError(f"Could not find latency for {m}-{n}-{k}-{format}")
    else:
        print(f"Found latency for {m}-{n}-{k}-{format}: {latency}")
    return latency

def get_latency_ladder(batch_size=1, format="float16xfloat16"):
    n1024k8192 = get_result_from_file_ladder(batch_size, 1024, 8192, format)
    n8192k8192 = get_result_from_file_ladder(batch_size, 8192, 8192, format)
    n28672k8192 = get_result_from_file_ladder(batch_size, 28672, 8192, format)
    n8192k28672 = get_result_from_file_ladder(batch_size, 8192, 28672, format)
    return n1024k8192 * 2 + n8192k8192 * 2 + n28672k8192 * 2 + n8192k28672

fp16xfp16_bs4096_overall_latency = get_latency_ladder(4096, "float16xfloat16")
fp16xint8_bs4096_overall_latency = get_latency_ladder(4096, "float16xint8")
fp16xint4_bs4096_overall_latency = get_latency_ladder(4096, "float16xint4")
fp16xint2_bs4096_overall_latency = get_latency_ladder(4096, "float16xint2")
fp16xint1_bs4096_overall_latency = get_latency_ladder(4096, "float16xint1")
int8xint8_bs4096_overall_latency = get_latency(4096, "int8xint8")
int8xint4_bs4096_overall_latency = get_latency(4096, "int8xint4")
int8xint2_bs4096_overall_latency = get_latency(4096, "int8xint2")
int8xint1_bs4096_overall_latency = get_latency(4096, "int8xint1")
int4xint4_bs4096_overall_latency = get_latency(4096, "int4xint4")
int4xint2_bs4096_overall_latency = get_latency(4096, "int4xint2")
int4xint1_bs4096_overall_latency = get_latency(4096, "int4xint1")

fp16xint8_bs4096_end2end_latency = bs4096_fp16_time - fp16xfp16_bs4096_overall_latency + fp16xint8_bs4096_overall_latency
fp16xint4_bs4096_end2end_latency = bs4096_fp16_time - fp16xfp16_bs4096_overall_latency + fp16xint4_bs4096_overall_latency
fp16xint2_bs4096_end2end_latency = bs4096_fp16_time - fp16xfp16_bs4096_overall_latency + fp16xint2_bs4096_overall_latency
fp16xint1_bs4096_end2end_latency = bs4096_fp16_time - fp16xfp16_bs4096_overall_latency + fp16xint1_bs4096_overall_latency
int8xint8_bs4096_end2end_latency = bs4096_fp16_time - fp16xfp16_bs4096_overall_latency + int8xint8_bs4096_overall_latency
int8xint4_bs4096_end2end_latency = bs4096_fp16_time - fp16xfp16_bs4096_overall_latency + int8xint4_bs4096_overall_latency
int8xint2_bs4096_end2end_latency = bs4096_fp16_time - fp16xfp16_bs4096_overall_latency + int8xint2_bs4096_overall_latency
int8xint1_bs4096_end2end_latency = bs4096_fp16_time - fp16xfp16_bs4096_overall_latency + int8xint1_bs4096_overall_latency
int4xint4_bs4096_end2end_latency = bs4096_fp16_time - fp16xfp16_bs4096_overall_latency + int4xint4_bs4096_overall_latency
int4xint2_bs4096_end2end_latency = bs4096_fp16_time - fp16xfp16_bs4096_overall_latency + int4xint2_bs4096_overall_latency
int4xint1_bs4096_end2end_latency = bs4096_fp16_time - fp16xfp16_bs4096_overall_latency + int4xint1_bs4096_overall_latency

b1s4096_times_data[0] = ("Bitter", [bs4096_fp16_time])
b1s4096_times_data[1] = ("Bitter-W$_{INT8}$A$_{FP16}$", [fp16xint8_bs4096_end2end_latency])
b1s4096_times_data[2] = ("Bitter-W$_{INT4}$A$_{FP16}$", [fp16xint4_bs4096_end2end_latency])
b1s4096_times_data[3] = ("Bitter-W$_{INT2}$A$_{FP16}$", [fp16xint2_bs4096_end2end_latency])
b1s4096_times_data[4] = ("Bitter-W$_{INT1}$A$_{FP16}$", [fp16xint1_bs4096_end2end_latency])
b1s4096_times_data[5] = ("Bitter-W$_{INT8}$A$_{INT8}$", [int8xint8_bs4096_end2end_latency])
b1s4096_times_data[6] = ("Bitter-W$_{INT4}$A$_{INT8}$", [int8xint4_bs4096_end2end_latency])
b1s4096_times_data[7] = ("Bitter-W$_{INT2}$A$_{INT8}$", [int8xint2_bs4096_end2end_latency])
b1s4096_times_data[8] = ("Bitter-W$_{INT1}$A$_{INT8}$", [int8xint1_bs4096_end2end_latency])
b1s4096_times_data[9] = ("Bitter-W$_{INT4}$A$_{INT4}$", [int4xint4_bs4096_end2end_latency])
b1s4096_times_data[10] = ("Bitter-W$_{INT2}$A$_{INT4}$", [int4xint2_bs4096_end2end_latency])
b1s4096_times_data[11] = ("Bitter-W$_{INT1}$A$_{INT4}$", [int4xint1_bs4096_end2end_latency])

# write the results to back

b1s1_matmul_times_data[0] = ("Bitter", [
    get_result_from_file(1, 1024, 8192, "float16xfloat16"),
    get_result_from_file(1, 8192, 8192, "float16xfloat16"),
    get_result_from_file(1, 28672, 8192, "float16xfloat16"),
    get_result_from_file(1, 8192, 28672, "float16xfloat16"),
])

b1s1_matmul_times_data[1] = ("Bitter-W$_{INT8}$A$_{FP16}$", [
    get_result_from_file(1, 1024, 8192, "float16xint8"),
    get_result_from_file(1, 8192, 8192, "float16xint8"),
    get_result_from_file(1, 28672, 8192, "float16xint8"),
    get_result_from_file(1, 8192, 28672, "float16xint8"),
])

b1s1_matmul_times_data[2] = ("Bitter-W$_{INT4}$A$_{FP16}$", [
    get_result_from_file(1, 1024, 8192, "float16xint4"),
    get_result_from_file(1, 8192, 8192, "float16xint4"),
    get_result_from_file(1, 28672, 8192, "float16xint4"),
    get_result_from_file(1, 8192, 28672, "float16xint4"),
])

b1s1_matmul_times_data[3] = ("Bitter-W$_{INT2}$A$_{FP16}$", [
    get_result_from_file(1, 1024, 8192, "float16xint2"),
    get_result_from_file(1, 8192, 8192, "float16xint2"),
    get_result_from_file(1, 28672, 8192, "float16xint2"),
    get_result_from_file(1, 8192, 28672, "float16xint2"),
])

b1s1_matmul_times_data[4] = ("Bitter-W$_{INT1}$A$_{FP16}$", [
    get_result_from_file(1, 1024, 8192, "float16xint1"),
    get_result_from_file(1, 8192, 8192, "float16xint1"),
    get_result_from_file(1, 28672, 8192, "float16xint1"),
    get_result_from_file(1, 8192, 28672, "float16xint1"),
])

b1s1_matmul_times_data[5] = ("Bitter-W$_{INT8}$A$_{INT8}$", [
    get_result_from_file(1, 1024, 8192, "int8xint8"),
    get_result_from_file(1, 8192, 8192, "int8xint8"),
    get_result_from_file(1, 28672, 8192, "int8xint8"),
    get_result_from_file(1, 8192, 28672, "int8xint8"),
])

b1s1_matmul_times_data[6] = ("Bitter-W$_{INT4}$A$_{INT8}$", [
    get_result_from_file(1, 1024, 8192, "int8xint4"),
    get_result_from_file(1, 8192, 8192, "int8xint4"),
    get_result_from_file(1, 28672, 8192, "int8xint4"),
    get_result_from_file(1, 8192, 28672, "int8xint4"),
])

b1s1_matmul_times_data[7] = ("Bitter-W$_{INT2}$A$_{INT8}$", [
    get_result_from_file(1, 1024, 8192, "int8xint2"),
    get_result_from_file(1, 8192, 8192, "int8xint2"),
    get_result_from_file(1, 28672, 8192, "int8xint2"),
    get_result_from_file(1, 8192, 28672, "int8xint2"),
])

b1s1_matmul_times_data[8] = ("Bitter-W$_{INT1}$A$_{INT8}$", [
    get_result_from_file(1, 1024, 8192, "int8xint1"),
    get_result_from_file(1, 8192, 8192, "int8xint1"),
    get_result_from_file(1, 28672, 8192, "int8xint1"),
    get_result_from_file(1, 8192, 28672, "int8xint1"),
])

b1s1_matmul_times_data[9] = ("Bitter-W$_{INT4}$A$_{INT4}$", [
    get_result_from_file(1, 1024, 8192, "int4xint4"),
    get_result_from_file(1, 8192, 8192, "int4xint4"),
    get_result_from_file(1, 28672, 8192, "int4xint4"),
    get_result_from_file(1, 8192, 28672, "int4xint4"),
])

b1s1_matmul_times_data[10] = ("Bitter-W$_{INT2}$A$_{INT4}$", [
    get_result_from_file(1, 1024, 8192, "int4xint2"),
    get_result_from_file(1, 8192, 8192, "int4xint2"),
    get_result_from_file(1, 28672, 8192, "int4xint2"),
    get_result_from_file(1, 8192, 28672, "int4xint2"),
])

b1s1_matmul_times_data[11] = ("Bitter-W$_{INT1}$A$_{INT4}$", [
    get_result_from_file(1, 1024, 8192, "int4xint1"),
    get_result_from_file(1, 8192, 8192, "int4xint1"),
    get_result_from_file(1, 28672, 8192, "int4xint1"),
    get_result_from_file(1, 8192, 28672, "int4xint1"),
])

b1s4096_matmul_times_data[0] = ("Bitter", [
    get_result_from_file_ladder(4096, 1024, 8192, "float16xfloat16"),
    get_result_from_file_ladder(4096, 8192, 8192, "float16xfloat16"),
    get_result_from_file_ladder(4096, 28672, 8192, "float16xfloat16"),
    get_result_from_file_ladder(4096, 8192, 28672, "float16xfloat16"),
])

b1s4096_matmul_times_data[1] = ("Bitter-W$_{INT8}$A$_{FP16}$", [
    get_result_from_file_ladder(4096, 1024, 8192, "float16xint8"),
    get_result_from_file_ladder(4096, 8192, 8192, "float16xint8"),
    get_result_from_file_ladder(4096, 28672, 8192, "float16xint8"),
    get_result_from_file_ladder(4096, 8192, 28672, "float16xint8"),
])

b1s4096_matmul_times_data[2] = ("Bitter-W$_{INT4}$A$_{FP16}$", [
    get_result_from_file_ladder(4096, 1024, 8192, "float16xint4"),
    get_result_from_file_ladder(4096, 8192, 8192, "float16xint4"),
    get_result_from_file_ladder(4096, 28672, 8192, "float16xint4"),
    get_result_from_file_ladder(4096, 8192, 28672, "float16xint4"),
])

b1s4096_matmul_times_data[3] = ("Bitter-W$_{INT2}$A$_{FP16}$", [
    get_result_from_file_ladder(4096, 1024, 8192, "float16xint2"),
    get_result_from_file_ladder(4096, 8192, 8192, "float16xint2"),
    get_result_from_file_ladder(4096, 28672, 8192, "float16xint2"),
    get_result_from_file_ladder(4096, 8192, 28672, "float16xint2"),
])

b1s4096_matmul_times_data[4] = ("Bitter-W$_{INT1}$A$_{FP16}$", [
    get_result_from_file_ladder(4096, 1024, 8192, "float16xint1"),
    get_result_from_file_ladder(4096, 8192, 8192, "float16xint1"),
    get_result_from_file_ladder(4096, 28672, 8192, "float16xint1"),
    get_result_from_file_ladder(4096, 8192, 28672, "float16xint1"),
])

b1s4096_matmul_times_data[5] = ("Bitter-W$_{INT8}$A$_{INT8}$", [
    get_result_from_file(4096, 1024, 8192, "int8xint8"),
    get_result_from_file(4096, 8192, 8192, "int8xint8"),
    get_result_from_file(4096, 28672, 8192, "int8xint8"),
    get_result_from_file(4096, 8192, 28672, "int8xint8"),
])

b1s4096_matmul_times_data[6] = ("Bitter-W$_{INT4}$A$_{INT8}$", [
    get_result_from_file(4096, 1024, 8192, "int8xint4"),
    get_result_from_file(4096, 8192, 8192, "int8xint4"),
    get_result_from_file(4096, 28672, 8192, "int8xint4"),
    get_result_from_file(4096, 8192, 28672, "int8xint4"),
])

b1s4096_matmul_times_data[7] = ("Bitter-W$_{INT2}$A$_{INT8}$", [
    get_result_from_file(4096, 1024, 8192, "int8xint2"),
    get_result_from_file(4096, 8192, 8192, "int8xint2"),
    get_result_from_file(4096, 28672, 8192, "int8xint2"),
    get_result_from_file(4096, 8192, 28672, "int8xint2"),
])

b1s4096_matmul_times_data[8] = ("Bitter-W$_{INT1}$A$_{INT8}$", [
    get_result_from_file(4096, 1024, 8192, "int8xint1"),
    get_result_from_file(4096, 8192, 8192, "int8xint1"),
    get_result_from_file(4096, 28672, 8192, "int8xint1"),
    get_result_from_file(4096, 8192, 28672, "int8xint1"),
])

b1s4096_matmul_times_data[9] = ("Bitter-W$_{INT4}$A$_{INT4}$", [
    get_result_from_file(4096, 1024, 8192, "int4xint4"),
    get_result_from_file(4096, 8192, 8192, "int4xint4"),
    get_result_from_file(4096, 28672, 8192, "int4xint4"),
    get_result_from_file(4096, 8192, 28672, "int4xint4"),
])

b1s4096_matmul_times_data[10] = ("Bitter-W$_{INT2}$A$_{INT4}$", [
    get_result_from_file(4096, 1024, 8192, "int4xint2"),
    get_result_from_file(4096, 8192, 8192, "int4xint2"),
    get_result_from_file(4096, 28672, 8192, "int4xint2"),
    get_result_from_file(4096, 8192, 28672, "int4xint2"),
])

b1s4096_matmul_times_data[11] = ("Bitter-W$_{INT1}$A$_{INT4}$", [
    get_result_from_file(4096, 1024, 8192, "int4xint1"),
    get_result_from_file(4096, 8192, 8192, "int4xint1"),
    get_result_from_file(4096, 28672, 8192, "int4xint1"),
    get_result_from_file(4096, 8192, 28672, "int4xint1"),
])




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
