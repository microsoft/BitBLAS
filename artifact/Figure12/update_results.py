import os
import json
import re

_ = '''
matmul_providers = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5']
matmul_times_data = [
    ('cuBLAS', [1.0265737319667645, 1.090096979579753, 32.19012415077543, 0.23601211538704653, 0.3087000088622984, 8.437653349759566]),
    (
        "cuTLASS-W$_{INT4}$A$_{FP16}$",
        [0.674009323, 1.186704636, 33.67717266, 0.153660774, 0.259065628, 12.6046657],
    ),
    (
        "vLLM-W$_{INT4}$A$_{FP16}$",
        [0.484972, 0.972840786, 123.6705709, 0.18430233, 0.27905941, 29.32891846],
    ),
    ('Bitter', [0.9405568846412276, 1.0551276457642045, 27.429847626938027, 0.2771763573698525, 0.37289717979611964, 7.4194252736607105]),
    ('Bitter-W$_{INT4}$A$_{FP16}$', [0.25010346697898184, 1.0387403257520407, 24.396201991737335, 0.08182817299037373, 0.3599448964549887, 6.908026137726486]),
    ('Bitter-W$_{NF4}$A$_{FP16}$', [0.4349415730022948, 1.076610324314873, 29.317079643679847, 0.1302926846604392, 0.4114702888693471, 8.07674323812536]),
    ('Bitter-W$_{FP8}$A$_{FP16}$', [0.5063097258488188, 0.9116776456324998, 24.726286922592116, 0.1496626511701893, 0.40203168771938597, 7.377204819112391]),
    ('Bitter-W$_{INT1}$A$_{INT8}$', [0.08608339837700796, 0.5491120564385428, 16.392895179202853, 0.031667271184114415, 0.21352213323045818, 4.696140613337209]),
    ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [0.6882330597220419, 1.7458009555693463, 49.64125827473772, 0.2086473571409937, 0.6039546796566252, 14.876466315323867]),
]

conv_providers = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
conv_times_data = [
    ('Bitter', [0.17222472205030204, 0.0627413114266695, 0.1444130641442279, 0.08958541672377299, 0.017040223486371315, 0.00475818323917974, 0.027083107486657255, 0.01007713817509755]),
    ('Bitter-W$_{FP8}$A$_{FP16}$', [0.1920176582196141, 0.0649479272122272, 0.16466570875166223, 0.09424366220429246, 0.019521395359481428, 0.004846908101216902, 0.033601077131209904, 0.013450713721221136]),
    ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [0.20945456879787666, 0.06390084338885563, 0.19512846820997637, 0.09007950653989266, 0.03574100723027465, 0.006131866719143937, 0.04281825508486072, 0.02079322744118922]),
    ('Bitter-W$_{INT4}$A$_{INT4}$', [0.09983259941458962, 0.07319274742701586, 0.08117026180627498, 0.1258184631771389, 0.054007608357730066, 0.009136191443331875, 0.1652795893521068, 0.06247456887300485]),
    ('cuDNN', [0.3454588013021706, 0.12968900304905515, 0.2582418254785964, 0.12204221896644463, 0.0609764208856802, 0.059271116948255334, 0.05592442013986263, 0.06370989806214339]),
    ('AMOS', [1.764688, 0.302502, 0.857058, 0.3248, 0.073223653, 0.020407328, 0.106228661, 0.046321647]), ('TensorIR', [0.2430228, 0.0932777, 0.216, 0.0802, 0.018251371, 0.00476218, 0.026268746, 0.010746771])
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


# parse the results from cublas
cublas_data = matmul_times_data[0][1]
def get_and_print_cublas(m, n, k, log):
    data = None
    with open(log) as f:
        lines = f.readlines()
        for line in lines:
            if f"{m},{n},{k}" in line:
                data = float(re.findall(r"\d+\.\d+", line)[-2])
                print(data)
    return data
for i, (m, n, k) in enumerate([
        [1,14336,57344],
        [32,14336,57344],
        [4096,14336,57344],
        [1,8192,28672],
        [32,8192,28672],
        [4096,8192,28672],
    ]):
    log_path = f"./cublas-benchmark/build/cublas_benchmark.log"
    if not os.path.exists(log_path):
        continue
    data = get_and_print_cublas(m, n, k, log_path)
    cublas_data[i] = data
matmul_times_data[0] = ("cuBLAS", cublas_data)

cutlass_data = matmul_times_data[1][1]
# parse the results from cutlass_fpa_intb
for i, (m, n, k) in enumerate([
        [1,14336,57344],
        [32,14336,57344],
        [4096,14336,57344],
        [1,8192,28672],
        [32,8192,28672],
        [4096,8192,28672],
    ]):
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
                    cutlass_data[i] = data
matmul_times_data[1] = ("cuTLASS-W$_{INT4}$A$_{FP16}$", cutlass_data)

# parse the results from vllm
vllm_data = matmul_times_data[2][1]
# parse the results from cutlass_fpa_intb
for i, (m, n, k) in enumerate([
        [1,14336,57344],
        [32,14336,57344],
        [4096,14336,57344],
        [1,8192,28672],
        [32,8192,28672],
        [4096,8192,28672],
    ]):
    log_path = f"./vllm-benchmark/logs/vllm_benchmark_kernel.log"
    if not os.path.exists(log_path):
        continue
    else:
        with open(log_path) as f:
            lines = f.readlines()
            for line in lines:
                if f"{m} {n} {k}" in line:
                    data = float(re.findall(r"\d+\.\d+", line)[-1])
                    print(data)
                    vllm_data[i] = data
matmul_times_data[2] = ("vLLM-W$_{INT4}$A$_{FP16}$", vllm_data)

# parse the result from bitter
def extract_floats(line):
    pattern = r"\b\d+\.\d+\b"
    return re.findall(pattern, line)
def get_result_from_file(m, n, k, format="fp16xfp16", KERNEL_LOG_PATH="./ladder-benchmark/logs/"):
    suffix = "gemm" if m != 1 else "gemv"
    if "welder" in KERNEL_LOG_PATH:
        log_path = f"{KERNEL_LOG_PATH}{format}_{suffix}_nt.log"
    else:
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
            latency = float(matches[-1])
            break
    return latency

bitter_data = matmul_times_data[3][1]
for i, (m, n, k) in enumerate([
        [1,14336,57344],
        [32,14336,57344],
        [4096,14336,57344],
        [1,8192,28672],
        [32,8192,28672],
        [4096,8192,28672],
    ]):
    data = get_result_from_file(m, n, k, format="fp16xfp16", KERNEL_LOG_PATH="./ladder-benchmark/logs/")
    if data is None:
        print(f"Could not find Bitter latency for {m}_{n}_{k}")
    else:
        print(f"Bitter latency for {m}_{n}_{k}: {data}, the paper results is {bitter_data[i]}")
    bitter_data[i] = data
matmul_times_data[3] = ("Bitter", bitter_data)

bitter_int4_fp16_data = matmul_times_data[4][1]
for i, (m, n, k) in enumerate([
        [1,14336,57344],
        [32,14336,57344],
        [4096,14336,57344],
        [1,8192,28672],
        [32,8192,28672],
        [4096,8192,28672],
    ]):
    data = get_result_from_file(m, n, k, format="fp16xint4", KERNEL_LOG_PATH="./ladder-benchmark/logs/")
    if data is None:
        print(f"Could not find Bitter-W_INT4_A_FP16 latency for {m}_{n}_{k}")
    else:
        print(f"Bitter-W_INT4_A_FP16 latency for {m}_{n}_{k}: {data}, the paper results is {bitter_int4_fp16_data[i]}")
    bitter_int4_fp16_data[i] = data
matmul_times_data[4] = ("Bitter-W$_{INT4}$A$_{FP16}$", bitter_int4_fp16_data)

bitter_nf4_fp16_data = matmul_times_data[5][1]
for i, (m, n, k) in enumerate([
        [1,14336,57344],
        [32,14336,57344],
        [4096,14336,57344],
        [1,8192,28672],
        [32,8192,28672],
        [4096,8192,28672],
    ]):
    data = get_result_from_file(m, n, k, format="fp16xnf4", KERNEL_LOG_PATH="./ladder-benchmark/logs/")
    if data is None:
        print(f"Could not find Bitter-W_NF4_A_FP16 latency for {m}_{n}_{k}")
    else:
        print(f"Bitter-W_NF4_A_FP16 latency for {m}_{n}_{k}: {data}, the paper results is {bitter_nf4_fp16_data[i]}")
    bitter_nf4_fp16_data[i] = data
matmul_times_data[5] = ("Bitter-W$_{NF4}$A$_{FP16}$", bitter_nf4_fp16_data)

bitter_fp8_fp16_data = matmul_times_data[6][1]
for i, (m, n, k) in enumerate([
        [1,14336,57344],
        [32,14336,57344],
        [4096,14336,57344],
        [1,8192,28672],
        [32,8192,28672],
        [4096,8192,28672],
    ]):
    data = get_result_from_file(m, n, k, format="fp16xfp8", KERNEL_LOG_PATH="./ladder-benchmark/logs/")
    if data is None:
        print(f"Could not find Bitter-W_FP8_A_FP16 latency for {m}_{n}_{k}")
    else:
        print(f"Bitter-W_FP8_A_FP16 latency for {m}_{n}_{k}: {data}, the paper results is {bitter_fp8_fp16_data[i]}")
    bitter_fp8_fp16_data[i] = data

matmul_times_data[6] = ("Bitter-W$_{FP8}$A$_{FP16}$", bitter_fp8_fp16_data)

bitter_int1_int8_data = matmul_times_data[7][1]
for i, (m, n, k) in enumerate([
        [1,14336,57344],
        [32,14336,57344],
        [4096,14336,57344],
        [1,8192,28672],
        [32,8192,28672],
        [4096,8192,28672],
    ]):
    data = get_result_from_file(m, n, k, format="int8xint1", KERNEL_LOG_PATH="./ladder-benchmark/logs/")
    if data is None:
        print(f"Could not find Bitter-W_INT1_A_INT8 latency for {m}_{n}_{k}")
    else:
        print(f"Bitter-W_INT1_A_INT8 latency for {m}_{n}_{k}: {data}, the paper results is {bitter_int1_int8_data[i]}")
    bitter_int1_int8_data[i] = data

matmul_times_data[7] = ("Bitter-W$_{INT1}$A$_{INT8}$", bitter_int1_int8_data)

bitter_mxfp8_mxfp8_data = matmul_times_data[8][1]
for i, (m, n, k) in enumerate([
        [1,14336,57344],
        [32,14336,57344],
        [4096,14336,57344],
        [1,8192,28672],
        [32,8192,28672],
        [4096,8192,28672],
    ]):
    if m == 1:
        data = get_result_from_file(m, n, k, format="fp32xmxfp8", KERNEL_LOG_PATH="./ladder-benchmark/logs/")
    else:
        data = get_result_from_file(m, n, k, format="bf16xmxfp8", KERNEL_LOG_PATH="./ladder-benchmark/logs/")
    if data is None:
        print(f"Could not find Bitter-W_MXFP8_A_MXFP8 latency for {m}_{n}_{k}")
    else:
        print(f"Bitter-W_MXFP8_A_MXFP8 latency for {m}_{n}_{k}: {data}, the paper results is {bitter_mxfp8_mxfp8_data[i]}")
    bitter_mxfp8_mxfp8_data[i] = data

matmul_times_data[8] = ("Bitter-W$_{MXFP8}$A$_{MXFP8}$", bitter_mxfp8_mxfp8_data)

# update the results for cudnn
cudnn_data = conv_times_data[4][1]
for i, (n, c, h, w, f, k, s, p) in enumerate(
    [
        (128, 64, 56, 56, 64, 3, 1, 1,),
        (128, 64, 56, 56, 64, 1, 1, 0,),
        (128, 128, 28, 28, 128, 3, 1, 1,),
        (128, 512, 28, 28, 128, 1, 1, 0,),
        (1, 64, 56, 56, 64, 3, 1, 1,),
        (1, 64, 56, 56, 64, 1, 1, 0,),
        (1, 128, 28, 28, 128, 3, 1, 1,),
        (1, 512, 28, 28, 128, 1, 1, 0,),
    ]
):
    with open(f"./cudnn-benchmark/logs/cudnn_fp16.log") as file:
        lines = file.readlines()
        for line in lines:
            if f"{n},{c},{h},{w},{f},{k},{s},{p}" in line:
                data = float(re.findall(r"\d+\.\d+", line)[-1])
                print("find data for cudnn: ", data, "the paper results is ", cudnn_data[i])
                cudnn_data[i] = data
conv_times_data[4] = ("cuDNN", cudnn_data)
            
# update the results for Bitter
bitter_data = conv_times_data[0][1]
for i, (n, c, h, w, f, k, s, p) in enumerate(
    [
        (128, 64, 56, 56, 64, 3, 1, 1,),
        (128, 64, 56, 56, 64, 1, 1, 0,),
        (128, 128, 28, 28, 128, 3, 1, 1,),
        (128, 512, 28, 28, 128, 1, 1, 0,),
        (1, 64, 56, 56, 64, 3, 1, 1,),
        (1, 64, 56, 56, 64, 1, 1, 0,),
        (1, 128, 28, 28, 128, 3, 1, 1,),
        (1, 512, 28, 28, 128, 1, 1, 0,),
    ]
):
    if n == 128:
        with open(f"./ladder-benchmark/logs/conv_nhwc_nhwc_fp16xfp16.log") as file:
            lines = file.readlines()
            for line in lines:
                if f"{n}_{c}_{h}_{w}_{f}_{k}_{k}_{s}_1_{p}" in line:
                    data = float(re.findall(r"\d+\.\d+", line)[-1])
                    print("find data for bitter: ", data, "the paper results is ", bitter_data[i])
                    bitter_data[i] = data
    elif n == 1:
        with open(f"./ladder-benchmark/logs/direct_conv_nhwc_nhwc_fp16xfp16.log") as file:
            lines = file.readlines()
            for line in lines:
                if f"{n}_{c}_{h}_{w}_{f}_{k}_{k}_{s}_1_{p}" in line:
                    data = float(re.findall(r"\d+\.\d+", line)[-1])
                    print("find data for bitter: ", data, "the paper results is ", bitter_data[i])
                    bitter_data[i] = data

# update the results for Bitter-W_FP8_A_FP16
bitter_fp8_fp16_data = conv_times_data[1][1]
# update the results for Bitter
bitter_data = conv_times_data[0][1]
for i, (n, c, h, w, f, k, s, p) in enumerate(
    [
        (128, 64, 56, 56, 64, 3, 1, 1,),
        (128, 64, 56, 56, 64, 1, 1, 0,),
        (128, 128, 28, 28, 128, 3, 1, 1,),
        (128, 512, 28, 28, 128, 1, 1, 0,),
        (1, 64, 56, 56, 64, 3, 1, 1,),
        (1, 64, 56, 56, 64, 1, 1, 0,),
        (1, 128, 28, 28, 128, 3, 1, 1,),
        (1, 512, 28, 28, 128, 1, 1, 0,),
    ]
):
    if n == 128:
        with open(f"./ladder-benchmark/logs/conv_nhwc_nhwc_fp16xfp8_e5m2.log") as file:
            lines = file.readlines()
            for line in lines:
                if f"{n}_{c}_{h}_{w}_{f}_{k}_{k}_{s}_1_{p}" in line:
                    data = float(re.findall(r"\d+\.\d+", line)[-1])
                    print("find data for bitter: ", data, "the paper results is ", bitter_data[i])
                    bitter_data[i] = data
    elif n == 1:
        with open(f"./ladder-benchmark/logs/direct_conv_nhwc_nhwc_fp16xfp8_e5m2.log") as file:
            lines = file.readlines()
            for line in lines:
                if f"{n}_{c}_{h}_{w}_{f}_{k}_{k}_{s}_1_{p}" in line:
                    data = float(re.findall(r"\d+\.\d+", line)[-1])
                    print("find data for bitter: ", data, "the paper results is ", bitter_data[i])
                    bitter_data[i] = data

conv_times_data[1] = ("Bitter-W$_{FP8}$A$_{FP16}$", bitter_fp8_fp16_data)

# update the results for Bitter
bitter_data = conv_times_data[2][1]
for i, (n, c, h, w, f, k, s, p) in enumerate(
    [
        (128, 64, 56, 56, 64, 3, 1, 1,),
        (128, 64, 56, 56, 64, 1, 1, 0,),
        (128, 128, 28, 28, 128, 3, 1, 1,),
        (128, 512, 28, 28, 128, 1, 1, 0,),
        (1, 64, 56, 56, 64, 3, 1, 1,),
        (1, 64, 56, 56, 64, 1, 1, 0,),
        (1, 128, 28, 28, 128, 3, 1, 1,),
        (1, 512, 28, 28, 128, 1, 1, 0,),
    ]
):
    if n == 128:
        with open(f"./ladder-benchmark/logs/conv_nhwc_nhwc_bfp16xmxfp8_e5m2.log") as file:
            lines = file.readlines()
            for line in lines:
                if f"{n}_{c}_{h}_{w}_{f}_{k}_{k}_{s}_1_{p}" in line:
                    data = float(re.findall(r"\d+\.\d+", line)[-1])
                    print("find data for bitter mxfp n128: ", data, "the paper results is ", bitter_data[i])
                    bitter_data[i] = data
    elif n == 1:
        with open(f"./ladder-benchmark/logs/direct_conv_nhwc_nhwc_fp32xmxfp8_e5m2.log") as file:
            lines = file.readlines()
            for line in lines:
                if f"{n}_{c}_{h}_{w}_{f}_{k}_{k}_{s}_1_{p}" in line:
                    data = float(re.findall(r"\d+\.\d+", line)[-1])
                    print("find data for bitter mxfp n1: ", data, "the paper results is ", bitter_data[i])
                    bitter_data[i] = data

conv_times_data[2] = ("Bitter-W$_{MXFP8}$A$_{MXFP8}$", bitter_data)

# update the results for Bitter-W_INT4_A_INT4
bitter_data = conv_times_data[3][1]
for i, (n, c, h, w, f, k, s, p) in enumerate(
    [
        (128, 64, 56, 56, 64, 3, 1, 1,),
        (128, 64, 56, 56, 64, 1, 1, 0,),
        (128, 128, 28, 28, 128, 3, 1, 1,),
        (128, 512, 28, 28, 128, 1, 1, 0,),
        (1, 64, 56, 56, 64, 3, 1, 1,),
        (1, 64, 56, 56, 64, 1, 1, 0,),
        (1, 128, 28, 28, 128, 3, 1, 1,),
        (1, 512, 28, 28, 128, 1, 1, 0,),
    ]
):
    if n == 128:
        with open(f"./ladder-benchmark/logs/conv_nhwc_nhwc_int4xint4.log") as file:
            lines = file.readlines()
            for line in lines:
                if f"{n}_{c}_{h}_{w}_{f}_{k}_{k}_{s}_1_{p}" in line:
                    data = float(re.findall(r"\d+\.\d+", line)[-1])
                    print("find data for bitter int4 n128: ", data, "the paper results is ", bitter_data[i])
                    bitter_data[i] = data
    elif n == 1:
        with open(f"./ladder-benchmark/logs/direct_conv_nhwc_nhwc_int8xint4.log") as file:
            lines = file.readlines()
            for line in lines:
                if f"{n}_{c}_{h}_{w}_{f}_{k}_{k}_{s}_1_{p}" in line:
                    data = float(re.findall(r"\d+\.\d+", line)[-1])
                    print("find data for bitter int4 n1: ", data, "the paper results is ", bitter_data[i])
                    bitter_data[i] = data

conv_times_data[3] = ("Bitter-W$_{INT4}$A$_{INT4}$", bitter_data)

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
