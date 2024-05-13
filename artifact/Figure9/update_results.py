# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import json
import re


llama2_providers = ["BS1 SEQ1", "BS32 SEQ1", "BS1 SEQ4096"]

bloom_providers = ["BS1 SEQ1", "BS32 SEQ1", "BS1 SEQ4096"]

resnet_providers = ["BS1", "BS128"]

shufflenet_providers = ["BS1", "BS128"]

conformer_providers = ["BS1", "BS128"]

vit_providers = ["BS1", "BS128"]

## update pytorch Inductor results
resnet_50_b1_logs = './pytorch-inductor-benchmark/logs/resnet-50-b1.log'
pattern = r"avg: [\d]+\.[\d]+ ms"
with open(resnet_50_b1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            pytorch_time_b1 = float(matches[0].split(' ')[-2])
print(pytorch_time_b1)

resnet_50_b128_logs = './pytorch-inductor-benchmark/logs/resnet-50-b128.log'
pattern = r"avg: [\d]+\.[\d]+ ms"
with open(resnet_50_b128_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            pytorch_time_b128 = float(matches[0].split(' ')[-2])
print(pytorch_time_b128)

_ = """
llama2_providers = ['BS1 SEQ1', 'BS32 SEQ1', 'BS1 SEQ4096']
llama2_times_data = [
    ('PyTorch-Inductor', [3.162500858, 3.11050415, 100.0415564]),
    ('ONNXRuntime', [3.6252, 4.4371, 144.3875]),
    ('TensorRT', [2.11548, 2.35596, 121.227]),
    ('Welder', [2.144288, 2.480128, 94.676994]),
    ('vLLM', [2.348845005, 2.505731583, 90.14932156]),
    ('vLLM-W$_{INT4}$A$_{FP16}$', [0, 0, 0]),
    ('Bitter', [1.982, 2.447, 98.8685]),
    ('Bitter-W$_{INT4}$A$_{FP16}$', [0.59439808, 2.158567894, 97.91209607]),
    ('Bitter-W$_{NF4}$A$_{FP16}$', [0.830255887, 2.324443739, 107.4976185]),
    ('Bitter-W$_{FP8}$A$_{FP8}$', [1.089282821, 3.466108892, 228.1671086]),
    ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [0.204822669, 10.80787646, 699.8160501]),
    ('Bitter-W$_{INT1}$A$_{INT8}$', [0.207486801, 5.313923571, 559.9500032]),
]

bloom_providers = ['BS1 SEQ1', 'BS32 SEQ1', 'BS1 SEQ4096']
bloom_times_data = [
    ('PyTorch-Inductor', [8.333725929, 8.624098301, 0]),
    ('ONNXRuntime', [0, 0, 0]),
    ('TensorRT', [5.7852, 6.15668, 0]),
    ('Welder', [5.849088, 6.606816, 0]),
    ('vLLM', [0, 0, 0]),
    ('vLLM-W$_{INT4}$A$_{FP16}$', [0, 0, 0]),
    ('Bitter', [5.5886, 6.1652, 0]),
    ('Bitter-W$_{INT4}$A$_{FP16}$', [1.53901594, 3.901990775, 0]),
    ('Bitter-W$_{NF4}$A$_{FP16}$', [1.69907736, 5.181669567, 0]),
    ('Bitter-W$_{FP8}$A$_{FP8}$', [2.83026982, 5.502423499, 0]),
    ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [3.310397363, 3.310397363, 0]),
    ('Bitter-W$_{INT1}$A$_{INT8}$', [0.43558272, 6.1652, 0]),
]

resnet_providers = ['BS1', 'BS128']
resnet_times_data = [
    ('PyTorch-Inductor', [4.82632637, 25.97796679]),
    ('ONNXRuntime', [3.7384, 97.6342]),
    ('TensorRT', [1.85937, 18.8322]),
    ('AMOS', [19.77248, 144.9735]),
    ('TensorIR', [1.609463602, 26.15646306]),
    ('Welder', [2.656288, 42.615776]),
    ('Bitter', [1.4638, 21.0237]),
    ('Bitter_W$_{FP8}$A$_{FP8}$', [1.4638, 21.0237]),
    ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [2.48846, 94.60665]),
    ('Bitter_W$_{INT1}$A$_{INT4}$', [1.4638, 21.0237]),
]

shufflenet_providers = ['BS1', 'BS128']
shufflenet_times_data = [
    ('PyTorch-Inductor', [6.236689091, 6.174676418]),
    ('ONNXRuntime', [2.8359, 14.0666]),
    ('TensorRT', [1.33392, 5.26163]),
    ('AMOS', [3.954496, 35.31771]),
    ('TensorIR', [0.411856816, 7.081917907]),
    ('Welder', [0.40752, 6.562784]),
    ('Bitter', [0.4042, 5.2663]),
    ('Bitter_W$_{FP8}$A$_{FP8}$', [0.4042, 5.2663]),
]

conformer_providers = ['BS1', 'BS128']
conformer_times_data = [
    ('PyTorch-Inductor', [13.62011671, 168.6849737]),
    ('ONNXRuntime', [10.1335, 408.1039]),
    ('TensorRT', [3.53897, 162.431]),
    ('AMOS', [0, 0]),
    ('TensorIR', [0, 0]),
    ('Welder', [4.04784, 172.965851]),
    ('Bitter', [3.5447, 193.1576]),
    ('Bitter-W$_{INT4}$A$_{INT8}$', [3.5447, 193.1576]),
    ('Bitter-W$_{INT4}$A$_{INT4}$', [3.5447, 193.1576]),
]

vit_providers = ['BS1', 'BS128']
vit_times_data = [
    ('PyTorch-Inductor', [5.180325508, 8.272943497]),
    ('ONNXRuntime', [3.5002, 23.8669]),
    ('TensorRT', [1.17185, 8.76167]),
    ('AMOS', [0, 0]),
    ('TensorIR', [1.179153433, 14.82752]),
    ('Welder', [1.31072, 8.150656]),
    ('Bitter', [1.32948, 9.2983]),
    ('Bitter-W$_{FP8}$A$_{FP8}$', [1.32948, 9.2983]),
    ('Bitter-W$_{INT4}$A$_{INT4}$', [1.32948, 9.2983]),
]
"""


exec(_)
resnet_times_data[0] = ('Pytorch Inductor', [pytorch_time_b1, pytorch_time_b128])

shufflenet_b1_logs = './pytorch-inductor-benchmark/logs/shufflenet-b1.log'
pattern = r"avg: [\d]+\.[\d]+ ms"
with open(shufflenet_b1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            pytorch_time_b1 = float(matches[0].split(' ')[-2])
print(pytorch_time_b1)

shufflenet_b128_logs = './pytorch-inductor-benchmark/logs/shufflenet-b128.log'
pattern = r"avg: [\d]+\.[\d]+ ms"
with open(shufflenet_b128_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            pytorch_time_b128 = float(matches[0].split(' ')[-2])
print(pytorch_time_b128)
shufflenet_times_data[0] = ('Pytorch Inductor', [pytorch_time_b1, pytorch_time_b128])

Conformer_b1_logs = './pytorch-inductor-benchmark/logs/Conformer-b1.log'
pattern = r"avg: [\d]+\.[\d]+ ms"
with open(Conformer_b1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            pytorch_time_b1 = float(matches[0].split(' ')[-2])
print(pytorch_time_b1)

Conformer_b128_logs = './pytorch-inductor-benchmark/logs/Conformer-b128.log'
pattern = r"avg: [\d]+\.[\d]+ ms"
with open(Conformer_b128_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            pytorch_time_b128 = float(matches[0].split(' ')[-2])
print(pytorch_time_b128)

conformer_times_data[0] = ('Pytorch Inductor', [pytorch_time_b1, pytorch_time_b128])

vit_b1_logs = './pytorch-inductor-benchmark/logs/vit-b1.log'
pattern = r"avg: [\d]+\.[\d]+ ms"
with open(vit_b1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            pytorch_time_b1 = float(matches[0].split(' ')[-2])
print(pytorch_time_b1)

vit_b128_logs = './pytorch-inductor-benchmark/logs/vit-b128.log'

pattern = r"avg: [\d]+\.[\d]+ ms"
with open(vit_b128_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            pytorch_time_b128 = float(matches[0].split(' ')[-2])
print(pytorch_time_b128)

vit_times_data[0] = ('Pytorch Inductor', [pytorch_time_b1, pytorch_time_b128])

llama_70b_b1s1_logs = './pytorch-inductor-benchmark/logs/llama-70b-layer1-seq1-bs1.log'
pattern = r"avg: [\d]+\.[\d]+ ms"
with open(llama_70b_b1s1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            pytorch_time_b1 = float(matches[0].split(' ')[-2])
print(pytorch_time_b1)

llama_70b_b32s1_logs = './pytorch-inductor-benchmark/logs/llama-70b-layer1-seq1-bs32.log'
pattern = r"avg: [\d]+\.[\d]+ ms"
with open(llama_70b_b32s1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            pytorch_time_b32 = float(matches[0].split(' ')[-2])
print(pytorch_time_b32)

llama_70b_b1s4096_logs = './pytorch-inductor-benchmark/logs/llama-70b-layer1-seq4096-bs1.log'

pattern = r"avg: [\d]+\.[\d]+ ms"
with open(llama_70b_b1s4096_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            pytorch_time_b4096 = float(matches[0].split(' ')[-2])
print(pytorch_time_b4096)

llama2_times_data[0] = ('Pytorch Inductor', [pytorch_time_b1, pytorch_time_b32, pytorch_time_b4096])

bloom_176b_b1s1_logs = './pytorch-inductor-benchmark/logs/bloom-176b-layer1-seq1-bs1.log'
pattern = r"avg: [\d]+\.[\d]+ ms"
with open(bloom_176b_b1s1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            pytorch_time_b1 = float(matches[0].split(' ')[-2])
print(pytorch_time_b1)

bloom_176b_b32s1_logs = './pytorch-inductor-benchmark/logs/bloom-176b-layer1-seq1-bs32.log'
pattern = r"avg: [\d]+\.[\d]+ ms"
with open(bloom_176b_b32s1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            pytorch_time_b32 = float(matches[0].split(' ')[-2])
print(pytorch_time_b32)

bloom_176b_b1s4096_logs = './pytorch-inductor-benchmark/logs/bloom-176b-layer1-seq4096-bs1.log'
pattern = r"avg: [\d]+\.[\d]+ ms"
with open(bloom_176b_b1s4096_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            pytorch_time_b4096 = float(matches[0].split(' ')[-2])
print(pytorch_time_b4096)
bloom_times_data[0] = ('Pytorch Inductor', [pytorch_time_b1, pytorch_time_b32, pytorch_time_b4096])

## update onnxruntime results
resnet_50_b1_logs = './onnxruntime-benchmark/logs/resnet-50-b1.log'
### match x(float) from Average time for each run: x ms 
pattern = r"Average time for each run: [\d]+\.[\d]+ ms"
with open(resnet_50_b1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'Average time for each run' in line:
            matches = re.findall(pattern, line)
            if matches:
                onnx_time_b1 = float(matches[0].split(' ')[-2])
print(onnx_time_b1)

resnet_50_b128_logs = './onnxruntime-benchmark/logs/resnet-50-b128.log'
### match x(float) from Average time for each run: x ms
pattern = r"Average time for each run: [\d]+\.[\d]+ ms"
with open(resnet_50_b128_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'Average time for each run' in line:
            matches = re.findall(pattern, line)
            if matches:
                onnx_time_b128 = float(matches[0].split(' ')[-2])
print(onnx_time_b128)

resnet_times_data[1] = ('ONNXRuntime', [onnx_time_b1, onnx_time_b128])

shufflenet_b1_logs = './onnxruntime-benchmark/logs/shufflenet-b1.log'
### match x(float) from Average time for each run: x ms
pattern = r"Average time for each run: [\d]+\.[\d]+ ms"
with open(shufflenet_b1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'Average time for each run' in line:
            matches = re.findall(pattern, line)
            if matches:
                onnx_time_b1 = float(matches[0].split(' ')[-2])
print(onnx_time_b1)

shufflenet_b128_logs = './onnxruntime-benchmark/logs/shufflenet-b128.log'
### match x(float) from Average time for each run: x ms
pattern = r"Average time for each run: [\d]+\.[\d]+ ms"
with open(shufflenet_b128_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'Average time for each run' in line:
            matches = re.findall(pattern, line)
            if matches:
                onnx_time_b128 = float(matches[0].split(' ')[-2])
print(onnx_time_b128)

shufflenet_times_data[1] = ('ONNXRuntime', [onnx_time_b1, onnx_time_b128])

Conformer_b1_logs = './onnxruntime-benchmark/logs/Conformer-b1.log'
### match x(float) from Average time for each run: x ms
pattern = r"Average time for each run: [\d]+\.[\d]+ ms"
with open(Conformer_b1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'Average time for each run' in line:
            matches = re.findall(pattern, line)
            if matches:
                onnx_time_b1 = float(matches[0].split(' ')[-2])
print(onnx_time_b1)

Conformer_b128_logs = './onnxruntime-benchmark/logs/Conformer-b128.log'
### match x(float) from Average time for each run: x ms
pattern = r"Average time for each run: [\d]+\.[\d]+ ms"
with open(Conformer_b128_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'Average time for each run' in line:
            matches = re.findall(pattern, line)
            if matches:
                onnx_time_b128 = float(matches[0].split(' ')[-2])
print(onnx_time_b128)
conformer_times_data[1] = ('ONNXRuntime', [onnx_time_b1, onnx_time_b128])

vit_b1_logs = './onnxruntime-benchmark/logs/vit-b1.log'
### match x(float) from Average time for each run: x ms
pattern = r"Average time for each run: [\d]+\.[\d]+ ms"
with open(vit_b1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'Average time for each run' in line:
            matches = re.findall(pattern, line)
            if matches:
                onnx_time_b1 = float(matches[0].split(' ')[-2])
print(onnx_time_b1)

vit_b128_logs = './onnxruntime-benchmark/logs/vit-b128.log'
### match x(float) from Average time for each run: x ms
pattern = r"Average time for each run: [\d]+\.[\d]+ ms"
with open(vit_b128_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'Average time for each run' in line:
            matches = re.findall(pattern, line)
            if matches:
                onnx_time_b128 = float(matches[0].split(' ')[-2])
print(onnx_time_b128)

vit_times_data[1] = ('ONNXRuntime', [onnx_time_b1, onnx_time_b128])

llama_70b_b1s1_logs = './onnxruntime-benchmark/logs/llama-70b-layer1-seq1-bs1.log'
pattern = r"Average time for each run: [\d]+\.[\d]+ ms"
with open(llama_70b_b1s1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            onnx_time_b1 = float(matches[0].split(' ')[-2])
print(onnx_time_b1)

llama_70b_b32s1_logs = './onnxruntime-benchmark/logs/llama-70b-layer1-seq1-bs32.log'
pattern = r"Average time for each run: [\d]+\.[\d]+ ms"
with open(llama_70b_b32s1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            onnx_time_b32 = float(matches[0].split(' ')[-2])
print(onnx_time_b32)

llama_70b_b1s4096_logs = './onnxruntime-benchmark/logs/llama-70b-layer1-seq4096-bs1.log'
pattern = r"Average time for each run: [\d]+\.[\d]+ ms"
with open(llama_70b_b1s4096_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            onnx_time_b4096 = float(matches[0].split(' ')[-2])
print(onnx_time_b4096)

llama2_times_data[1] = ('ONNXRuntime', [onnx_time_b1, onnx_time_b32, onnx_time_b4096])

bloom_176b_b1s1_logs = './onnxruntime-benchmark/logs/bloom-176b-layer1-seq1-bs1.log'
pattern = r"Average time for each run: [\d]+\.[\d]+ ms"
with open(bloom_176b_b1s1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            onnx_time_b1 = float(matches[0].split(' ')[-2])
print(onnx_time_b1)

bloom_176b_b32s1_logs = './onnxruntime-benchmark/logs/bloom-176b-layer1-seq1-bs32.log'
pattern = r"Average time for each run: [\d]+\.[\d]+ ms"
with open(bloom_176b_b32s1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            onnx_time_b32 = float(matches[0].split(' ')[-2])
print(onnx_time_b32)

bloom_176b_b1s4096_logs = './onnxruntime-benchmark/logs/bloom-176b-layer1-seq4096-bs1.log'
pattern = r"Average time for each run: [\d]+\.[\d]+ ms"
with open(bloom_176b_b1s4096_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            onnx_time_b4096 = float(matches[0].split(' ')[-2])
print(onnx_time_b4096)

bloom_times_data[1] = ('ONNXRuntime', [onnx_time_b1, onnx_time_b32, onnx_time_b4096])

## update tensorrt results
resnet_50_b1_logs = './tensorrt-benchmark/logs/resnet-50-b1.log'
### parse x(float) mean = x ms
pattern = r"mean = [\d]+\.[\d]+ ms"
with open(resnet_50_b1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if "GPU Compute Time" in line:
            matches = re.findall(pattern, line)
            if matches:
                tensorrt_time_b1 = float(matches[0].split(' ')[-2])
print(tensorrt_time_b1)

resnet_50_b128_logs = './tensorrt-benchmark/logs/resnet-50-b128.log'
### parse x(float) mean = x ms
pattern = r"mean = [\d]+\.[\d]+ ms"
with open(resnet_50_b128_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if "GPU Compute Time" in line:
            matches = re.findall(pattern, line)
            if matches:
                tensorrt_time_b128 = float(matches[0].split(' ')[-2])
print(tensorrt_time_b128)

resnet_times_data[2] = ('TensorRT', [tensorrt_time_b1, tensorrt_time_b128])

shufflenet_b1_logs = './tensorrt-benchmark/logs/shufflenet-b1.log'
pattern = r"mean = [\d]+\.[\d]+ ms"
with open(shufflenet_b1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if "GPU Compute Time" in line:
            matches = re.findall(pattern, line)
            if matches:
                tensorrt_time_b1 = float(matches[0].split(' ')[-2])
print(tensorrt_time_b1)

shufflenet_b128_logs = './tensorrt-benchmark/logs/shufflenet-b128.log'
pattern = r"mean = [\d]+\.[\d]+ ms"
with open(shufflenet_b128_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if "GPU Compute Time" in line:
            matches = re.findall(pattern, line)
            if matches:
                tensorrt_time_b128 = float(matches[0].split(' ')[-2])
print(tensorrt_time_b128)

shufflenet_times_data[2] = ('TensorRT', [tensorrt_time_b1, tensorrt_time_b128])

Conformer_b1_logs = './tensorrt-benchmark/logs/Conformer-b1.log'
pattern = r"mean = [\d]+\.[\d]+ ms"
with open(Conformer_b1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if "GPU Compute Time" in line:
            matches = re.findall(pattern, line)
            if matches:
                tensorrt_time_b1 = float(matches[0].split(' ')[-2])
print(tensorrt_time_b1)

Conformer_b128_logs = './tensorrt-benchmark/logs/Conformer-b128.log'
pattern = r"mean = [\d]+\.[\d]+ ms"
with open(Conformer_b128_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if "GPU Compute Time" in line:
            matches = re.findall(pattern, line)
            if matches:
                tensorrt_time_b128 = float(matches[0].split(' ')[-2])
print(tensorrt_time_b128)

conformer_times_data[2] = ('TensorRT', [tensorrt_time_b1, tensorrt_time_b128])

vit_b1_logs = './tensorrt-benchmark/logs/vit-b1.log'
pattern = r"mean = [\d]+\.[\d]+ ms"
with open(vit_b1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if "GPU Compute Time" in line:
            matches = re.findall(pattern, line)
            if matches:
                tensorrt_time_b1 = float(matches[0].split(' ')[-2])
print(tensorrt_time_b1)

vit_b128_logs = './tensorrt-benchmark/logs/vit-b128.log'
pattern = r"mean = [\d]+\.[\d]+ ms"
with open(vit_b128_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if "GPU Compute Time" in line:
            matches = re.findall(pattern, line)
            if matches:
                tensorrt_time_b128 = float(matches[0].split(' ')[-2])
print(tensorrt_time_b128)

vit_times_data[2] = ('TensorRT', [tensorrt_time_b1, tensorrt_time_b128])

llama_70b_b1s1_logs = './tensorrt-benchmark/logs/llama-70b-layer1-seq1-bs1.log'
pattern = r"mean = [\d]+\.[\d]+ ms"
with open(llama_70b_b1s1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if "GPU Compute Time" in line:
            matches = re.findall(pattern, line)
            if matches:
                tensorrt_time_b1 = float(matches[0].split(' ')[-2])
print(tensorrt_time_b1)

llama_70b_b32s1_logs = './tensorrt-benchmark/logs/llama-70b-layer1-seq1-bs32.log'
pattern = r"mean = [\d]+\.[\d]+ ms"
with open(llama_70b_b32s1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if "GPU Compute Time" in line:
            matches = re.findall(pattern, line)
            if matches:
                tensorrt_time_b32 = float(matches[0].split(' ')[-2])
print(tensorrt_time_b32)

llama_70b_b1s4096_logs = './tensorrt-benchmark/logs/llama-70b-layer1-seq4096-bs1.log'
pattern = r"mean = [\d]+\.[\d]+ ms"
with open(llama_70b_b1s4096_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if "GPU Compute Time" in line:
            matches = re.findall(pattern, line)
            if matches:
                tensorrt_time_b4096 = float(matches[0].split(' ')[-2])
print(tensorrt_time_b4096)

llama2_times_data[2] = ('TensorRT', [tensorrt_time_b1, tensorrt_time_b32, tensorrt_time_b4096])

bloom_176b_b1s1_logs = './tensorrt-benchmark/logs/bloom-176b-layer1-seq1-bs1.log'
pattern = r"mean = [\d]+\.[\d]+ ms"
with open(bloom_176b_b1s1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if "GPU Compute Time" in line:
            matches = re.findall(pattern, line)
            if matches:
                tensorrt_time_b1 = float(matches[0].split(' ')[-2])
print(tensorrt_time_b1)

bloom_176b_b32s1_logs = './tensorrt-benchmark/logs/bloom-176b-layer1-seq1-bs32.log'
pattern = r"mean = [\d]+\.[\d]+ ms"
with open(bloom_176b_b32s1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if "GPU Compute Time" in line:
            matches = re.findall(pattern, line)
            if matches:
                tensorrt_time_b32 = float(matches[0].split(' ')[-2])
print(tensorrt_time_b32)

bloom_176b_b1s4096_logs = './tensorrt-benchmark/logs/bloom-176b-layer1-seq4096-bs1.log'
pattern = r"mean = [\d]+\.[\d]+ ms"
with open(bloom_176b_b1s4096_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if "GPU Compute Time" in line:
            matches = re.findall(pattern, line)
            if matches:
                tensorrt_time_b4096 = float(matches[0].split(' ')[-2])
print(tensorrt_time_b4096)

bloom_times_data[2] = ('TensorRT', [tensorrt_time_b1, tensorrt_time_b32, tensorrt_time_b4096])

## update welder results
## not tuned yet.

## update ladder results
## not tuned yet


## update amos results
resnet_50_b1_logs = './amos-benchmark/logs/resnet50_b1.log'
### match x(float) from Whole graph cost is x ms 
pattern = r"Whole graph cost is [\d]+\.[\d]+ ms"
with open(resnet_50_b1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'Whole graph cost is' in line:
            matches = re.findall(pattern, line)
            if matches:
                amos_time_b1 = float(matches[0].split(' ')[-2])
print(amos_time_b1)

resnet_50_b128_logs = './amos-benchmark/logs/resnet50_b128.log'
### match x(float) from Whole graph cost is x ms
pattern = r"Whole graph cost is [\d]+\.[\d]+ ms"
with open(resnet_50_b128_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'Whole graph cost is' in line:
            matches = re.findall(pattern, line)
            if matches:
                amos_time_b128 = float(matches[0].split(' ')[-2])
print(amos_time_b128)
resnet_times_data[3] = ('AMOS', [amos_time_b1, amos_time_b128])

shufflenet_b1_logs = './amos-benchmark/logs/shufflenet_v2_b1.log'
### match x(float) from Whole graph cost is x ms
pattern = r"Whole graph cost is [\d]+\.[\d]+ ms"
with open(shufflenet_b1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'Whole graph cost is' in line:
            matches = re.findall(pattern, line)
            if matches:
                amos_time_b1 = float(matches[0].split(' ')[-2])
print(amos_time_b1)

shufflenet_b128_logs = './amos-benchmark/logs/shufflenet_v2_b128.log'
### match x(float) from Whole graph cost is x ms
pattern = r"Whole graph cost is [\d]+\.[\d]+ ms"
with open(shufflenet_b128_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'Whole graph cost is' in line:
            matches = re.findall(pattern, line)
            if matches:
                amos_time_b128 = float(matches[0].split(' ')[-2])
print(amos_time_b128)

# analysis the ladder results
# bitter fp16xfp16
llama_b1s1_fp16_logs = './ladder-benchmark/logs/llama2/llama2-70b_b1_s1_q-1.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_llama_b1s1_fp16 = None
with open(llama_b1s1_fp16_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_llama_b1s1_fp16 = float(matches[0])
                is_next_line = False

llama_b32s1_fp16_logs = './ladder-benchmark/logs/llama2/llama2-70b_b32_s1_q-1.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_llama_b32s1_fp16 = None
with open(llama_b32s1_fp16_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_llama_b32s1_fp16 = float(matches[0])
                is_next_line = False

llama_b1s4096_fp16_logs = './ladder-benchmark/logs/llama2/llama2-70b_b1_s4096_q-1.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_llama_b1s4096_fp16 = None
with open(llama_b1s4096_fp16_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_llama_b1s4096_fp16 = float(matches[0])
                is_next_line = False

if ladder_llama_b1s1_fp16 and ladder_llama_b32s1_fp16 and ladder_llama_b1s4096_fp16:
    print(ladder_llama_b1s1_fp16, ladder_llama_b32s1_fp16, ladder_llama_b1s4096_fp16)
    llama2_times_data[6] = ('Bitter', [ladder_llama_b1s1_fp16, ladder_llama_b32s1_fp16, ladder_llama_b1s4096_fp16])


# bitter fp16xint4
llama_b1s1_int4_logs = './ladder-benchmark/logs/llama2/llama2-70b_b1_s1_q0_b4.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_llama_b1s1_int4 = None
with open(llama_b1s1_int4_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_llama_b1s1_int4 = float(matches[0])
                is_next_line = False

llama_b32s1_int4_logs = './ladder-benchmark/logs/llama2/llama2-70b_b32_s1_q0_b4.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_llama_b32s1_int4 = None
with open(llama_b32s1_int4_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_llama_b32s1_int4 = float(matches[0])
                is_next_line = False

llama_b1s4096_int4_logs = './ladder-benchmark/logs/llama2/llama2-70b_b1_s4096_q0_b4.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_llama_b1s4096_int4 = None
with open(llama_b1s4096_int4_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_llama_b1s4096_int4 = float(matches[0])
                is_next_line = False

if ladder_llama_b1s1_int4 and ladder_llama_b32s1_int4 and ladder_llama_b1s4096_int4:
    print(ladder_llama_b1s1_int4, ladder_llama_b32s1_int4, ladder_llama_b1s4096_int4)
    llama2_times_data[7] = ('Bitter-W$_{INT4}$A$_{FP16}$', [ladder_llama_b1s1_int4, ladder_llama_b32s1_int4, ladder_llama_b1s4096_int4])

# bitter fp16xint4
llama_b1s1_nf4_logs = './ladder-benchmark/logs/llama2/llama2-70b_b1_s1_q0_nf4.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_llama_b1s1_nf4 = None
with open(llama_b1s1_nf4_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_llama_b1s1_nf4 = float(matches[0])
                is_next_line = False

llama_b32s1_nf4_logs = './ladder-benchmark/logs/llama2/llama2-70b_b32_s1_q0_nf4.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_llama_b32s1_nf4 = None
with open(llama_b32s1_nf4_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_llama_b32s1_nf4 = float(matches[0])
                is_next_line = False

llama_b1s4096_nf4_logs = './ladder-benchmark/logs/llama2/llama2-70b_b1_s4096_q0_nf4.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_llama_b1s4096_nf4 = None
with open(llama_b1s4096_nf4_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_llama_b1s4096_nf4 = float(matches[0])
                is_next_line = False

if ladder_llama_b1s1_nf4 and ladder_llama_b32s1_nf4 and ladder_llama_b1s4096_nf4:
    print(ladder_llama_b1s1_nf4, ladder_llama_b32s1_nf4, ladder_llama_b1s4096_nf4)
    llama2_times_data[8] = ('Bitter-W$_{NF4}$A$_{FP16}$', [ladder_llama_b1s1_nf4, ladder_llama_b32s1_nf4, ladder_llama_b1s4096_nf4])

## fp8
# bitter fp16xint4
llama_b1s1_fp8_logs = './ladder-benchmark/logs/llama2/llama2-70b_b1_s1_q0_fp_e5m2.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_llama_b1s1_fp8 = None
with open(llama_b1s1_fp8_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_llama_b1s1_fp8 = float(matches[0])
                is_next_line = False

llama_b32s1_fp8_logs = './ladder-benchmark/logs/llama2/llama2-70b_b32_s1_q0_fp_e5m2.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_llama_b32s1_fp8 = None
with open(llama_b32s1_fp8_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_llama_b32s1_fp8 = float(matches[0])
                is_next_line = False

llama_b1s4096_fp8_logs = './ladder-benchmark/logs/llama2/llama2-70b_b1_s4096_q0_fp_e5m2.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_llama_b1s4096_fp8 = None
with open(llama_b1s4096_fp8_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_llama_b1s4096_fp8 = float(matches[0])
                is_next_line = False

if ladder_llama_b1s1_fp8 and ladder_llama_b32s1_fp8 and ladder_llama_b1s4096_fp8:
    print(ladder_llama_b1s1_fp8, ladder_llama_b32s1_fp8, ladder_llama_b1s4096_fp8)
    llama2_times_data[9] = ('Bitter-W$_{FP8}$A$_{FP8}$', [ladder_llama_b1s1_fp8, ladder_llama_b32s1_fp8, ladder_llama_b1s4096_fp8])
else:
    print(ladder_llama_b1s1_fp8, ladder_llama_b32s1_fp8, ladder_llama_b1s4096_fp8)



# bitter fp16xint4
llama_b1s1_mxfp8_logs = './ladder-benchmark/logs/llama2/llama2-70b_b1_s1_q0_mxfp8.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_llama_b1s1_mxfp8 = None
with open(llama_b1s1_mxfp8_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_llama_b1s1_mxfp8 = float(matches[0])
                is_next_line = False

llama_b32s1_mxfp8_logs = './ladder-benchmark/logs/llama2/llama2-70b_b32_s1_q0_mxfp8.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_llama_b32s1_mxfp8 = None
with open(llama_b32s1_mxfp8_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_llama_b32s1_mxfp8 = float(matches[0])
                is_next_line = False

llama_b1s4096_mxfp8_logs = './ladder-benchmark/logs/llama2/llama2-70b_b1_s4096_q0_mxfp8.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_llama_b1s4096_mxfp8 = None
with open(llama_b1s4096_mxfp8_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_llama_b1s4096_mxfp8 = float(matches[0])
                is_next_line = False

if ladder_llama_b1s1_mxfp8 and ladder_llama_b32s1_mxfp8 and ladder_llama_b1s4096_mxfp8:
    print(ladder_llama_b1s1_mxfp8, ladder_llama_b32s1_mxfp8, ladder_llama_b1s4096_mxfp8)
    llama2_times_data[10] = ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [ladder_llama_b1s1_mxfp8, ladder_llama_b32s1_mxfp8, ladder_llama_b1s4096_mxfp8])

# bitter fp16xint4
llama_b1s1_int1_logs = './ladder-benchmark/logs/llama2/llama2-70b_b1_s1_q0_b1_int.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_llama_b1s1_int1 = None
with open(llama_b1s1_int1_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_llama_b1s1_int1 = float(matches[0])
                is_next_line = False

llama_b32s1_int1_logs = './ladder-benchmark/logs/llama2/llama2-70b_b32_s1_q0_b1_int.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_llama_b32s1_int1 = None
with open(llama_b32s1_int1_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_llama_b32s1_int1 = float(matches[0])
                is_next_line = False

llama_b1s4096_int1_logs = './ladder-benchmark/logs/llama2/llama2-70b_b1_s4096_q0_b1_int.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_llama_b1s4096_int1 = None
with open(llama_b1s4096_int1_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_llama_b1s4096_int1 = float(matches[0])
                is_next_line = False

# bitter fp16xfp16
bloom_b1s1_fp16_logs = './ladder-benchmark/logs/bloom/bloom-176b_b1_s1_q-1.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_bloom_b1s1_fp16 = None
with open(bloom_b1s1_fp16_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_bloom_b1s1_fp16 = float(matches[0])
                is_next_line = False

bloom_b32s1_fp16_logs = './ladder-benchmark/logs/bloom/bloom-176b_b32_s1_q-1.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_bloom_b32s1_fp16 = None
with open(bloom_b32s1_fp16_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_bloom_b32s1_fp16 = float(matches[0])
                is_next_line = False

bloom_b1s4096_fp16_logs = './ladder-benchmark/logs/bloom/bloom-176b_b1_s4096_q-1.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_bloom_b1s4096_fp16 = None
with open(bloom_b1s4096_fp16_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_bloom_b1s4096_fp16 = float(matches[0])
                is_next_line = False

if ladder_bloom_b1s1_fp16 and ladder_bloom_b32s1_fp16:
    print(ladder_bloom_b1s1_fp16, ladder_bloom_b32s1_fp16)
    bloom_times_data[6] = ('Bitter', [ladder_bloom_b1s1_fp16, ladder_bloom_b32s1_fp16, 0])


# bitter fp16xint4
bloom_b1s1_int4_logs = './ladder-benchmark/logs/bloom/bloom-176b_b1_s1_q0_b4.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_bloom_b1s1_int4 = None
with open(bloom_b1s1_int4_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_bloom_b1s1_int4 = float(matches[0])
                is_next_line = False

bloom_b32s1_int4_logs = './ladder-benchmark/logs/bloom/bloom-176b_b32_s1_q0_b4.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_bloom_b32s1_int4 = None
with open(bloom_b32s1_int4_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_bloom_b32s1_int4 = float(matches[0])
                is_next_line = False

if ladder_bloom_b1s1_int4 and ladder_bloom_b32s1_int4:
    print(ladder_bloom_b1s1_int4, ladder_bloom_b32s1_int4)
    bloom_times_data[7] = ('Bitter-W$_{INT4}$A$_{FP16}$', [ladder_bloom_b1s1_int4, ladder_bloom_b32s1_int4, 0])

# bitter fp16xint4
bloom_b1s1_nf4_logs = './ladder-benchmark/logs/bloom/bloom-176b_b1_s1_q0_nf4.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_bloom_b1s1_nf4 = None
with open(bloom_b1s1_nf4_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_bloom_b1s1_nf4 = float(matches[0])
                is_next_line = False

bloom_b32s1_nf4_logs = './ladder-benchmark/logs/bloom/bloom-176b_b32_s1_q0_nf4.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_bloom_b32s1_nf4 = None
with open(bloom_b32s1_nf4_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_bloom_b32s1_nf4 = float(matches[0])
                is_next_line = False

if ladder_bloom_b1s1_nf4 and ladder_bloom_b32s1_nf4:
    print(ladder_bloom_b1s1_nf4, ladder_bloom_b32s1_nf4)
    bloom_times_data[8] = ('Bitter-W$_{NF4}$A$_{FP16}$', [ladder_bloom_b1s1_nf4, ladder_bloom_b32s1_nf4, 0])

## fp8
# bitter fp16xint4
bloom_b1s1_fp8_logs = './ladder-benchmark/logs/bloom/bloom-176b_b1_s1_q0_fp_e5m2.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_bloom_b1s1_fp8 = None
with open(bloom_b1s1_fp8_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_bloom_b1s1_fp8 = float(matches[0])
                is_next_line = False

bloom_b32s1_fp8_logs = './ladder-benchmark/logs/bloom/bloom-176b_b32_s1_q0_fp_e5m2.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_bloom_b32s1_fp8 = None
with open(bloom_b32s1_fp8_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_bloom_b32s1_fp8 = float(matches[0])
                is_next_line = False

if ladder_bloom_b1s1_fp8 and ladder_bloom_b32s1_fp8:
    print(ladder_bloom_b1s1_fp8, ladder_bloom_b32s1_fp8)
    bloom_times_data[9] = ('Bitter-W$_{FP8}$A$_{FP8}$', [ladder_bloom_b1s1_fp8, ladder_bloom_b32s1_fp8, 0])
else:
    print(ladder_bloom_b1s1_fp8, ladder_bloom_b32s1_fp8, ladder_bloom_b1s4096_fp8)



# bitter fp16xint4
bloom_b1s1_mxfp8_logs = './ladder-benchmark/logs/bloom/bloom-176b_b1_s1_q0_mxfp8.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_bloom_b1s1_mxfp8 = None
with open(bloom_b1s1_mxfp8_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_bloom_b1s1_mxfp8 = float(matches[0])
                is_next_line = False

bloom_b32s1_mxfp8_logs = './ladder-benchmark/logs/bloom/bloom-176b_b32_s1_q0_mxfp8.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_bloom_b32s1_mxfp8 = None
with open(bloom_b32s1_mxfp8_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_bloom_b32s1_mxfp8 = float(matches[0])
                is_next_line = False

if ladder_bloom_b1s1_mxfp8 and ladder_bloom_b32s1_mxfp8:
    print(ladder_bloom_b1s1_mxfp8, ladder_bloom_b32s1_mxfp8)
    bloom_times_data[10] = ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [ladder_bloom_b1s1_mxfp8, ladder_bloom_b32s1_mxfp8, 0])

# bitter fp16xint4
bloom_b1s1_int1_logs = './ladder-benchmark/logs/bloom/bloom-176b_b1_s1_q0_b1_int.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_bloom_b1s1_int1 = None
with open(bloom_b1s1_int1_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_bloom_b1s1_int1 = float(matches[0])
                is_next_line = False

bloom_b32s1_int1_logs = './ladder-benchmark/logs/bloom/bloom-176b_b32_s1_q0_b1_int.log'
### 
pattern = r"[\d]+\.[\d]+"
ladder_bloom_b32s1_int1 = None
with open(bloom_b32s1_int1_logs, 'r') as f:
    lines = f.readlines()
    is_next_line=False
    for line in lines:
        if 'mean (ms)' in line:
            is_next_line = True
        if is_next_line:
            matches = re.findall(pattern, line)
            if matches:
                ladder_bloom_b32s1_int1 = float(matches[0])
                is_next_line = False

# write the results to back

reproduced_results = f"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
llama2_providers = {llama2_providers}
llama2_times_data = {llama2_times_data}

bloom_providers = {bloom_providers}
bloom_times_data = {bloom_times_data}

resnet_providers = {resnet_providers}
resnet_times_data = {resnet_times_data}

shufflenet_providers = {shufflenet_providers}
shufflenet_times_data = {shufflenet_times_data}

conformer_providers = {conformer_providers}
conformer_times_data = {conformer_times_data}

vit_providers = {vit_providers}
vit_times_data = {vit_times_data}
"""

with open("reproduce_result/data_v100.py", "w") as f:
    f.write(reproduced_results)
