# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import json
import re
# for comparation with paper results
from paper_result.data_a100 import(
    llama2_times_data as paper_llama2_times_data,
    bloom_times_data as paper_bloom_times_data,
    resnet_times_data as paper_resnet_times_data,
    shufflenet_times_data as paper_shufflenet_times_data,
    conformer_times_data as paper_conformer_times_data,
    vit_times_data as paper_vit_times_data
)

llama2_providers = ['BS1 SEQ1', 'BS32 SEQ1', 'BS1 SEQ4096']
bloom_providers = ['BS1 SEQ1', 'BS32 SEQ1', 'BS1 SEQ4096']
resnet_providers = ['BS1', 'BS128']
shufflenet_providers = ['BS1', 'BS128']
conformer_providers = ['BS1', 'BS128']
vit_providers = ['BS1', 'BS128']

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

_ = """llama2_times_data = [
    ('PyTorch-Inductor', [-1, -1, -1]),
    ('ONNXRuntime', [-1, -1, -1]),
    ('TensorRT', [-1, -1, -1]),
    ('Welder', [-1, -1, -1]),
    ('vLLM', [-1, -1, -1]),
    ('vLLM-W$_{INT4}$A$_{FP16}$', [-1, -1, -1]),
    ('Bitter', [-1, -1, -1]),
    ('Bitter-W$_{INT4}$A$_{FP16}$', [-1, -1, -1]),
    ('Bitter-W$_{NF4}$A$_{FP16}$', [-1, -1, -1]),
    ('Bitter-W$_{FP8}$A$_{FP8}$', [-1, -1, -1]),
    ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [-1, -1, -1]),
    ('Bitter-W$_{INT1}$A$_{INT8}$', [-1, -1, -1])
]

bloom_times_data = [
    ('PyTorch-Inductor', [-1, -1, -1]),
    ('ONNXRuntime', [-1, -1, -1]),
    ('TensorRT', [-1, -1, -1]),
    ('Welder', [-1, -1, -1]),
    ('vLLM', [-1, -1, -1]),
    ('vLLM-W$_{INT4}$A$_{FP16}$', [-1, -1, -1]),
    ('Bitter', [-1, -1, -1]),
    ('Bitter-W$_{INT4}$A$_{FP16}$', [-1, -1, -1]),
    ('Bitter-W$_{NF4}$A$_{FP16}$', [-1, -1, -1]),
    ('Bitter-W$_{FP8}$A$_{FP8}$', [-1, -1, -1]),
    ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [-1, -1, -1]),
    ('Bitter-W$_{INT1}$A$_{INT8}$', [-1, -1, -1])
]

resnet_times_data = [
    ('PyTorch-Inductor', [-1, -1]),
    ('ONNXRuntime', [-1, -1]),
    ('TensorRT', [-1, -1]),
    ('AMOS', [-1, -1]),
    ('TensorIR', [-1, -1]),
    ('Welder', [-1, -1]),
    ('Bitter', [-1, -1]),
    ('Bitter_W$_{FP8}$A$_{FP8}$', [-1, -1]),
    ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [-1, -1]),
    ('Bitter_W$_{INT1}$A$_{INT4}$', [-1, -1])
]

shufflenet_times_data = [
    ('PyTorch-Inductor', [-1, -1]),
    ('ONNXRuntime', [-1, -1]),
    ('TensorRT', [-1, -1]),
    ('AMOS', [-1, -1]),
    ('TensorIR', [-1, -1]),
    ('Welder', [-1, -1]),
    ('Bitter', [-1, -1]),
    ('Bitter_W$_{FP8}$A$_{FP8}$', [-1, -1])
]

conformer_times_data = [
    ('PyTorch-Inductor', [-1, -1]),
    ('ONNXRuntime', [-1, -1]),
    ('TensorRT', [-1, -1]),
    ('AMOS', [0, 0]),
    ('TensorIR', [0, 0]),
    ('Welder', [-1, -1]),
    ('Bitter', [-1, -1]),
    ('Bitter-W$_{INT4}$A$_{INT8}$', [-1, -1]),
    ('Bitter-W$_{INT4}$A$_{INT4}$', [-1, -1])
]

vit_times_data = [
    ('PyTorch-Inductor', [-1, -1]),
    ('ONNXRuntime', [-1, -1]),
    ('TensorRT', [-1, -1]),
    ('AMOS', [0, 0]),
    ('TensorIR', [-1, -1]),
    ('Welder', [-1, -1]),
    ('Bitter', [-1, -1]),
    ('Bitter-W$_{FP8}$A$_{FP8}$', [-1, -1]),
    ('Bitter-W$_{INT4}$A$_{INT4}$', [-1, -1])
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
def get_welder_results(log_file):
    # Summary: [min, max, mean] = [3.961856, 3.982336, 3.973309] ms
    # parse x(float) from Summary: [min, max, mean] = [x, x, x] ms
    data = None
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Summary' in line:
                    # get the last float
                    pattern = r"[\d]+\.[\d]+"
                    matches = re.findall(pattern, line)
                    if matches:
                        data = float(matches[-1])
                        
    if data is not None:
        print(f"Welder data from {log_file} is {data}")
    return data   

# welder_times_data = [1.2515, 1.3723, 35.1400]
welder_times_data = [-1, -1, -1]  
latency_llama_b1s1 = get_welder_results('./welder-benchmark/compiled_models/llama2_70b_layer1_seq1_bs1_cutlass/run.log')
latency_llama_b1s32 = get_welder_results('./welder-benchmark/compiled_models/llama2_70b_layer1_seq1_bs32_cutlass/run.log')
latency_llama_b1s4096 = get_welder_results('./welder-benchmark/compiled_models/llama2_70b_layer1_seq4096_bs1_cutlass/run.log')
if latency_llama_b1s1 is not None:
    welder_times_data[0] = latency_llama_b1s1
if latency_llama_b1s32 is not None:
    welder_times_data[1] = latency_llama_b1s32
if latency_llama_b1s4096 is not None:
    welder_times_data[2] = latency_llama_b1s4096

llama2_times_data[3] = ('Welder', welder_times_data)

# welder_times_data = [3.0718, 3.4384, 115.7473]
welder_times_data = [-1, -1, -1]  
latency_bloom_b1s1 = get_welder_results('./welder-benchmark/compiled_models/bloom-176b_layer1_seq1_bs1_cutlass/run.log')
latency_bloom_b1s32 = get_welder_results('./welder-benchmark/compiled_models/bloom-176b_layer1_seq1_bs32_cutlass/run.log')
latency_bloom_b1s4096 = get_welder_results('./welder-benchmark/compiled_models/bloom-176b_layer1_seq4096_bs1_cutlass/run.log')

if latency_bloom_b1s1 is not None:
    welder_times_data[0] = latency_bloom_b1s1
if latency_bloom_b1s32 is not None:
    welder_times_data[1] = latency_bloom_b1s32
if latency_bloom_b1s4096 is not None:
    welder_times_data[2] = latency_bloom_b1s4096

bloom_times_data[3] = ('Welder', welder_times_data)

# welder_times_data = [1.8076, 16.7814]
welder_times_data = [-1, -1]  
layency_resnet_b1 = get_welder_results('./welder-benchmark/compiled_models/resnet-50-b1_cutlass/run.log')
layency_resnet_b128 = get_welder_results('./welder-benchmark/compiled_models/resnet-50-b128_cutlass/run.log')

if layency_resnet_b1 is not None:
    welder_times_data[0] = layency_resnet_b1
if layency_resnet_b128 is not None:
    welder_times_data[1] = layency_resnet_b128

resnet_times_data[5] = ('Welder', welder_times_data)

# welder_times_data = [0.3597, 3.9318]
welder_times_data = [-1, -1]
layency_shufflenet_b1 = get_welder_results('./welder-benchmark/compiled_models/shufflenet-b1_cutlass/run.log')
layency_shufflenet_b128 = get_welder_results('./welder-benchmark/compiled_models/shufflenet-b128_cutlass/run.log')
if layency_shufflenet_b1 is not None:
    welder_times_data[0] = layency_shufflenet_b1
if layency_shufflenet_b128 is not None:
    welder_times_data[1] = layency_shufflenet_b128

shufflenet_times_data[5] = ('Welder', welder_times_data)

# welder_times_data = [1.9198, 88.3134]
welder_times_data = [-1, -1]
layency_conformer_b1 = get_welder_results('./welder-benchmark/compiled_models/Conformer-b1_cutlass/run.log')
layency_conformer_b128 = get_welder_results('./welder-benchmark/compiled_models/Conformer-b128_cutlass/run.log')

if layency_conformer_b1 is not None:
    welder_times_data[0] = layency_conformer_b1
if layency_conformer_b128 is not None:
    welder_times_data[1] = layency_conformer_b128

conformer_times_data[5] = ('Welder', welder_times_data)

# welder_times_data = [1.1366, 5.2987]
welder_times_data = [-1, -1]
layency_vit_b1 = get_welder_results('./welder-benchmark/compiled_models/vit-b1_cutlass/run.log')
layency_vit_b128 = get_welder_results('./welder-benchmark/compiled_models/vit-b128_cutlass/run.log')
if layency_vit_b1 is not None:
    welder_times_data[0] = layency_vit_b1
if layency_vit_b128 is not None:
    welder_times_data[1] = layency_vit_b128

vit_times_data[5] = ('Welder', welder_times_data)

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

## llama fp16
# llama2-70b_b1_s1_q-1.log
# llama2-70b_b1_s1_q0_b1_int.log
# llama2-70b_b1_s1_q0_b4.log
# llama2-70b_b1_s1_q0_fp_e5m2.log
# llama2-70b_b1_s1_q0_mxfp8.log
# llama2-70b_b1_s1_q0_nf4.log
# llama2-70b_b1_s4096_q-1.log
# llama2-70b_b1_s4096_q0_b1_int.log
# llama2-70b_b1_s4096_q0_b4.log
# llama2-70b_b1_s4096_q0_fp_e5m2.log
# llama2-70b_b1_s4096_q0_fp_mxfp8.log
# llama2-70b_b1_s4096_q0_nf4.log
# llama2-70b_b32_s1_q0_b1_int.log
# llama2-70b_b32_s1_q0_b4.log
# llama2-70b_b32_s1_q0_fp_e5m2.log
# llama2-70b_b32_s1_q0_mxfp8.log
# llama2-70b_b32_s1_q0_nf4.log
# ladder_data = [1.0248, 1.3557, 34.7507]
ladder_data = [-1, -1, -1]
ladder_llama_fp16_b1s1_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b1_s1_q-1.log')
ladder_llama_fp16_b32s1_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b32_s1_q-1.log')
ladder_llama_fp16_b1s4096_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b1_s4096_q-1.log')

if ladder_llama_fp16_b1s1_latency is not None:
    ladder_data[0] = ladder_llama_fp16_b1s1_latency

if ladder_llama_fp16_b32s1_latency is not None:
    ladder_data[1] = ladder_llama_fp16_b32s1_latency

if ladder_llama_fp16_b1s4096_latency is not None:
    ladder_data[2] = ladder_llama_fp16_b1s4096_latency

llama2_times_data[6] = ('Bitter', ladder_data)

# ladder_data = [0.3563, 1.1973, 29.5409]
ladder_data = [-1, -1, -1]
ladder_llama_int4_b1s1_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b1_s1_q0_b4.log')
ladder_llama_int4_b32s1_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b32_s1_q0_b4.log')
ladder_llama_int4_b1s4096_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b1_s4096_q0_b4.log')

if ladder_llama_int4_b1s1_latency is not None:
    ladder_data[0] = ladder_llama_int4_b1s1_latency

if ladder_llama_int4_b32s1_latency is not None:
    ladder_data[1] = ladder_llama_int4_b32s1_latency

if ladder_llama_int4_b1s4096_latency is not None:
    ladder_data[2] = ladder_llama_int4_b1s4096_latency

llama2_times_data[7] = ('Bitter-W$_{INT4}$A$_{FP16}$', ladder_data)


# ladder_data = [0.5382, 1.3303, 30.6802]
ladder_data = [-1, -1, -1]
# nf4
ladder_llama_nf4_b1s1_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b1_s1_q0_nf4.log')
ladder_llama_nf4_b32s1_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b32_s1_q0_nf4.log')
ladder_llama_nf4_b1s4096_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b1_s4096_q0_nf4.log')

if ladder_llama_nf4_b1s1_latency is not None:
    ladder_data[0] = ladder_llama_nf4_b1s1_latency

if ladder_llama_nf4_b32s1_latency is not None:
    ladder_data[1] = ladder_llama_nf4_b32s1_latency

if ladder_llama_nf4_b1s4096_latency is not None:
    ladder_data[2] = ladder_llama_nf4_b1s4096_latency
    
llama2_times_data[8] = ('Bitter-W$_{NF4}$A$_{FP16}$', ladder_data)


# fp8
# ladder_data = [0.5758, 1.1959, 29.3180]
ladder_data = [-1, -1, -1]
ladder_llama_fp8_b1s1_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b1_s1_q0_fp_e5m2.log')
ladder_llama_fp8_b32s1_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b32_s1_q0_fp_e5m2.log')
ladder_llama_fp8_b1s4096_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b1_s4096_q0_fp_e5m2.log')

if ladder_llama_fp8_b1s1_latency is not None:
    ladder_data[0] = ladder_llama_fp8_b1s1_latency
if ladder_llama_fp8_b32s1_latency is not None:
    ladder_data[1] = ladder_llama_fp8_b32s1_latency
if ladder_llama_fp8_b1s4096_latency is not None:
    ladder_data[2] = ladder_llama_fp8_b1s4096_latency


llama2_times_data[9] = ('Bitter-W$_{FP8}$A$_{FP8}$', ladder_data)

# mxfp8
# ladder_data = [0.8369, 1.4239, 35.8447]
ladder_data = [-1, -1, -1]
ladder_llama_mxfp8_b1s1_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b1_s1_q0_mxfp8.log')
ladder_llama_mxfp8_b32s1_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b32_s1_q0_mxfp8.log')
ladder_llama_mxfp8_b1s4096_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b1_s4096_q0_fp_mxfp8.log')

if ladder_llama_mxfp8_b1s1_latency is not None:
    ladder_data[0] = ladder_llama_mxfp8_b1s1_latency
if ladder_llama_mxfp8_b32s1_latency is not None:
    ladder_data[1] = ladder_llama_mxfp8_b32s1_latency
if ladder_llama_mxfp8_b1s4096_latency is not None:
    ladder_data[2] = ladder_llama_mxfp8_b1s4096_latency

llama2_times_data[10] = ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', ladder_data)

# int8xint1
# ladder_data = [0.1629, 0.7379, 24.8855]
ladder_data = [-1, -1, -1]
ladder_llama_int4_b1s1_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b1_s1_q0_b1_int.log')
ladder_llama_int4_b32s1_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b32_s1_q0_b1_int.log')
ladder_llama_int4_b1s4096_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b1_s4096_q0_b1_int.log')

if ladder_llama_int4_b1s1_latency is not None:
    ladder_data[0] = ladder_llama_int4_b1s1_latency
if ladder_llama_int4_b32s1_latency is not None:
    ladder_data[1] = ladder_llama_int4_b32s1_latency
if ladder_llama_int4_b1s4096_latency is not None:
    ladder_data[2] = ladder_llama_int4_b1s4096_latency

llama2_times_data[11] = ('Bitter-W$_{INT1}$A$_{INT8}$', ladder_data)

## Bloom fp16
'''
bloom-176b_b1_s1_q-1.log
bloom-176b_b1_s1_q0_b1_int.log
bloom-176b_b1_s1_q0_b4.log
bloom-176b_b1_s1_q0_b8_int.log
bloom-176b_b1_s1_q0_fp_e5m2.log
bloom-176b_b1_s1_q0_mxfp8.log
bloom-176b_b1_s1_q0_nf4.log
bloom-176b_b1_s4096_q-1.log
bloom-176b_b1_s4096_q0_b1_int.log
bloom-176b_b1_s4096_q0_b4.log
bloom-176b_b1_s4096_q0_b8_int.log
bloom-176b_b1_s4096_q0_fp_e5m2.log
bloom-176b_b1_s4096_q0_fp_mxfp8.log
bloom-176b_b1_s4096_q0_nf4.log
bloom-176b_b32_s1_q0_b1_int.log
bloom-176b_b32_s1_q0_b4.log
bloom-176b_b32_s1_q0_fp_e5m2.log
bloom-176b_b32_s1_q0_mxfp8.log
bloom-176b_b32_s1_q0_nf4.log
'''

# ladder_data = [2.7872, 3.0271, 96.1634]
ladder_data = [-1, -1, -1]
ladder_bloom_fp16_b1s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s1_q-1.log')
ladder_bloom_fp16_b32s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b32_s1_q-1.log')
ladder_bloom_fp16_b1s4096_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s4096_q-1.log')

if ladder_bloom_fp16_b1s1_latency is not None:
    print(f"ladder_bloom_fp16_b1s1_latency: {ladder_bloom_fp16_b1s1_latency}, the paper value is {paper_bloom_times_data[6][1][0]}")
    ladder_data[0] = ladder_bloom_fp16_b1s1_latency
if ladder_bloom_fp16_b32s1_latency is not None:
    print(f"ladder_bloom_fp16_b32s1_latency: {ladder_bloom_fp16_b32s1_latency}, the paper value is {paper_bloom_times_data[6][1][1]}")
    ladder_data[1] = ladder_bloom_fp16_b32s1_latency
if ladder_bloom_fp16_b1s4096_latency is not None:
    print(f"ladder_bloom_fp16_b1s4096_latency: {ladder_bloom_fp16_b1s4096_latency}, the paper value is {paper_bloom_times_data[6][1][2]}")
    ladder_data[2] = ladder_bloom_fp16_b1s4096_latency

bloom_times_data[6] = ('Bitter', ladder_data)

# ladder_data = [0.8449, 2.2279, 91.9331]
ladder_data = [-1, -1, -1]
ladder_bloom_int4_b1s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s1_q0_b4.log')
ladder_bloom_int4_b32s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b32_s1_q0_b4.log')
ladder_bloom_int4_b1s4096_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s4096_q0_b4.log')

if ladder_bloom_int4_b1s1_latency is not None:
    ladder_data[0] = ladder_bloom_int4_b1s1_latency
if ladder_bloom_int4_b32s1_latency is not None:
    ladder_data[1] = ladder_bloom_int4_b32s1_latency
if ladder_bloom_int4_b1s4096_latency is not None:
    ladder_data[2] = ladder_bloom_int4_b1s4096_latency
    
bloom_times_data[7] = ('Bitter-W$_{INT4}$A$_{FP16}$', ladder_data)

# ladder_data = [1.3007, 2.6248, 101.0426]
ladder_data = [-1, -1, -1]
# nf4
ladder_bloom_nf4_b1s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s1_q0_nf4.log')
ladder_bloom_nf4_b32s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b32_s1_q0_nf4.log')
ladder_bloom_nf4_b1s4096_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s4096_q0_nf4.log')

if ladder_bloom_nf4_b1s1_latency is not None:
    ladder_data[0] = ladder_bloom_nf4_b1s1_latency
if ladder_bloom_nf4_b32s1_latency is not None:
    ladder_data[1] = ladder_bloom_nf4_b32s1_latency
if ladder_bloom_nf4_b1s4096_latency is not None:
    ladder_data[2] = ladder_bloom_nf4_b1s4096_latency

bloom_times_data[8] = ('Bitter-W$_{NF4}$A$_{FP16}$', ladder_data)

# fp8
# ladder_data = [1.5856, 2.1796, 88.7062]
ladder_data = [-1, -1, -1]
ladder_bloom_fp8_b1s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s1_q0_fp_e5m2.log')
ladder_bloom_fp8_b32s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b32_s1_q0_fp_e5m2.log')
ladder_bloom_fp8_b1s4096_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s4096_q0_fp_e5m2.log')

if ladder_bloom_fp8_b1s1_latency is not None:
    ladder_data[0] = ladder_bloom_fp8_b1s1_latency
if ladder_bloom_fp8_b32s1_latency is not None:
    ladder_data[1] = ladder_bloom_fp8_b32s1_latency
if ladder_bloom_fp8_b1s4096_latency is not None:
    ladder_data[2] = ladder_bloom_fp8_b1s4096_latency

bloom_times_data[9] = ('Bitter-W$_{FP8}$A$_{FP8}$', ladder_data)

# mxfp8
# ladder_data = [2.0269, 3.1147, 104.8811]
ladder_data = [-1, -1, -1]
ladder_bloom_mxfp8_b1s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s1_q0_mxfp8.log')
ladder_bloom_mxfp8_b32s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b32_s1_q0_mxfp8.log')
ladder_bloom_mxfp8_b1s4096_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s4096_q0_fp_mxfp8.log')

if ladder_bloom_mxfp8_b1s1_latency is not None:
    ladder_data[0] = ladder_bloom_mxfp8_b1s1_latency
if ladder_bloom_mxfp8_b32s1_latency is not None:
    ladder_data[1] = ladder_bloom_mxfp8_b32s1_latency
if ladder_bloom_mxfp8_b1s4096_latency is not None:
    ladder_data[2] = ladder_bloom_mxfp8_b1s4096_latency


bloom_times_data[10] = ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', ladder_data)

# int8xint1
# ladder_data = [0.3245, 1.2000, 70.5538]
ladder_data = [-1, -1, -1]
ladder_bloom_int4_b1s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s1_q0_b1_int.log')
ladder_bloom_int4_b32s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b32_s1_q0_b1_int.log')
ladder_bloom_int4_b1s4096_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s4096_q0_b1_int.log')

if ladder_bloom_int4_b1s1_latency is not None:
    ladder_data[0] = ladder_bloom_int4_b1s1_latency
if ladder_bloom_int4_b32s1_latency is not None:
    ladder_data[1] = ladder_bloom_int4_b32s1_latency
if ladder_bloom_int4_b1s4096_latency is not None:
    ladder_data[2] = ladder_bloom_int4_b1s4096_latency
    
bloom_times_data[11] = ('Bitter-W$_{INT1}$A$_{INT8}$', ladder_data)

# resnet
## fp16
ladder_data = resnet_times_data[6][1]
ladder_resnet_fp16_b1_latency = parse_ladder_logs('./ladder-benchmark/logs/resnet-50-b1.log')
ladder_resnet_fp16_b128_latency = parse_ladder_logs('./ladder-benchmark/logs/resnet-50-b128.log')

if ladder_resnet_fp16_b1_latency is not None:
    print(f"Ladder data from resnet-50-b1.log is {ladder_resnet_fp16_b1_latency}, the paper value is {paper_resnet_times_data[6][1][0]}")
    ladder_data[0] = ladder_resnet_fp16_b1_latency

if ladder_resnet_fp16_b128_latency is not None:
    print(f"Ladder data from resnet-50-b128.log is {ladder_resnet_fp16_b128_latency}, the paper value is {paper_resnet_times_data[6][1][1]}")
    ladder_data[1] = ladder_resnet_fp16_b128_latency

resnet_times_data[6] = ('Bitter', ladder_data)

## fp8
ladder_data = resnet_times_data[7][1]
ladder_resnet_fp8_b1_latency = parse_ladder_logs('./ladder-benchmark/logs/resnet-50-b1_fp8_e5m2.log')
ladder_resnet_fp16_b128_latency = parse_ladder_logs('./ladder-benchmark/logs/resnet-50-b128_fp8_e5m2.log')

if ladder_resnet_fp8_b1_latency is not None:
    print(f"Ladder data from resnet-50-b1_fp8_e5m2.log is {ladder_resnet_fp8_b1_latency}, the paper value is {paper_resnet_times_data[7][1][0]}")
    ladder_data[0] = ladder_resnet_fp8_b1_latency

if ladder_resnet_fp16_b128_latency is not None:
    print(f"Ladder data from resnet-50-b128_fp8_e5m2.log is {ladder_resnet_fp16_b128_latency}, the paper value is {paper_resnet_times_data[7][1][1]}")
    ladder_data[1] = ladder_resnet_fp16_b128_latency

resnet_times_data[7] = ('Bitter-W$_{FP8}$A$_{FP8}$', ladder_data)

## mxfp8
ladder_data = resnet_times_data[8][1]
ladder_resnet_mxfp8_b1_latency = parse_ladder_logs('./ladder-benchmark/logs/resnet-50-b1_mxfp8_e5m2.log')
latency_resnet_mxfp8_b128 = parse_ladder_logs('./ladder-benchmark/logs/resnet-50-b128_mxfp8_e5m2.log')

if ladder_resnet_mxfp8_b1_latency is not None:
    print(f"Ladder data from resnet-50-b1_mxfp8_e5m2.log is {ladder_resnet_mxfp8_b1_latency}, the paper value is {paper_resnet_times_data[8][1][0]}")
    ladder_resnet_mxfp8_b1_latency = 2.0269 # fp32 results
    ladder_data[0] = ladder_resnet_mxfp8_b1_latency

if latency_resnet_mxfp8_b128 is not None:
    print(f"Ladder data from resnet-50-b128_mxfp8_e5m2.log is {latency_resnet_mxfp8_b128}, the paper value is {paper_resnet_times_data[8][1][1]}")
    ladder_data[1] = latency_resnet_mxfp8_b128

resnet_times_data[8] = ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', ladder_data)

## int4xint1
ladder_data = resnet_times_data[9][1]
ladder_resnet_int4_b1_latency = parse_ladder_logs('./ladder-benchmark/logs/resnet-50-b1_int4b.log')
ladder_resnet_int4_b128_latency = parse_ladder_logs('./ladder-benchmark/logs/resnet-50-b128_int4b.log')

if ladder_resnet_int4_b1_latency is not None:
    print(f"Ladder data from resnet-50-b1_int4bxint1.log is {ladder_resnet_int4_b1_latency}, the paper value is {paper_resnet_times_data[9][1][0]}")
    ladder_data[0] = ladder_resnet_int4_b1_latency

if ladder_resnet_int4_b128_latency is not None:
    print(f"Ladder data from resnet-50-b128_int4bxint1.log is {ladder_resnet_int4_b128_latency}, the paper value is {paper_resnet_times_data[9][1][1]}")
    ladder_data[1] = ladder_resnet_int4_b128_latency

resnet_times_data[9] = ('Bitter-W$_{INT1}$A$_{INT4}$', ladder_data)

## shufflenet

ladder_data = shufflenet_times_data[6][1]
ladder_shufflenet_fp16_b1_latency = parse_ladder_logs('./ladder-benchmark/logs/shufflenet-b1.log')
ladder_shufflenet_fp16_b128_latency = parse_ladder_logs('./ladder-benchmark/logs/shufflenet-b128.log')

if ladder_shufflenet_fp16_b1_latency is not None:
    print(f"Ladder data from shufflenet_v2_b1.log is {ladder_shufflenet_fp16_b1_latency}, the paper value is {paper_shufflenet_times_data[6][1][0]}")
    ladder_data[0] = ladder_shufflenet_fp16_b1_latency

if ladder_shufflenet_fp16_b128_latency is not None:
    print(f"Ladder data from shufflenet_v2_b128.log is {ladder_shufflenet_fp16_b128_latency}, the paper value is {paper_shufflenet_times_data[6][1][1]}")
    ladder_data[1] = ladder_shufflenet_fp16_b128_latency

shufflenet_times_data[6] = ('Bitter', ladder_data)

## fp8
ladder_data = shufflenet_times_data[7][1]
ladder_shufflenet_fp8_b1_latency = parse_ladder_logs('./ladder-benchmark/logs/shufflenet-b1_fp8_e5m2.log')
ladder_shufflenet_fp8_b128_latency = parse_ladder_logs('./ladder-benchmark/logs/shufflenet-b128_fp8_e5m2.log')

if ladder_shufflenet_fp8_b1_latency is not None:
    print(f"Ladder data from shufflenet-b1_fp8_e5m2.log is {ladder_shufflenet_fp8_b1_latency}, the paper value is {paper_shufflenet_times_data[7][1][0]}")
    ladder_data[0] = ladder_shufflenet_fp8_b1_latency

if ladder_shufflenet_fp8_b128_latency is not None:
    print(f"Ladder data from shufflenet-b128_fp8_e5m2.log is {ladder_shufflenet_fp8_b128_latency}, the paper value is {paper_shufflenet_times_data[7][1][1]}")
    ladder_data[1] = ladder_shufflenet_fp8_b128_latency

shufflenet_times_data[7] = ('Bitter-W$_{FP8}$A$_{FP8}$', ladder_data)

# vit
ladder_data = vit_times_data[6][1]
ladder_vit_fp16_b1_latency = parse_ladder_logs('./ladder-benchmark/logs/vit-b1.log')
ladder_vit_fp16_b128_latency = parse_ladder_logs('./ladder-benchmark/logs/vit-b128.log')

if ladder_vit_fp16_b1_latency is not None:
    vit_times_data[7] = ('Bitter-W$_{FP8}$A$_{FP8}$', [1.2695, 4.0975])
    vit_times_data[8] = ('Bitter-W$_{INT4}$A$_{INT4}$', [1.1856, 3.4475])
    print(f"Ladder data from vit-b1.log is {ladder_vit_fp16_b1_latency}, the paper value is {paper_vit_times_data[6][1][0]}")
    ladder_data[0] = ladder_vit_fp16_b1_latency

if ladder_vit_fp16_b128_latency is not None:
    print(f"Ladder data from vit-b128.log is {ladder_vit_fp16_b128_latency}, the paper value is {paper_vit_times_data[6][1][1]}")
    ladder_data[1] = ladder_vit_fp16_b128_latency

vit_times_data[6] = ('Bitter', ladder_data)

# vit fp8
print(f"vit-b1 fp8 time with kernel quantized is {vit_times_data[7][1][0]}")
print(f"vit-b128 fp8 time with kernel quantized is {vit_times_data[7][1][1]}")

# int4
print(f"vit-b1 int4 time with kernel quantized is {vit_times_data[8][1][0]}")
print(f"vit-b128 int4 time with kernel quantized is {vit_times_data[8][1][1]}")


# conformer

ladder_data = conformer_times_data[6][1]
ladder_conformer_fp16_b1_latency = parse_ladder_logs('./ladder-benchmark/logs/Conformer-b1.log')
ladder_conformer_fp16_b128_latency = parse_ladder_logs('./ladder-benchmark/logs/Conformer-b128.log')

if ladder_conformer_fp16_b1_latency is not None:
    print(f"Ladder data from Conformer_b1.log is {ladder_conformer_fp16_b1_latency}, the paper value is {paper_conformer_times_data[6][1][0]}")
    ladder_data[0] = ladder_conformer_fp16_b1_latency

if ladder_conformer_fp16_b128_latency is not None:
    print(f"Ladder data from Conformer_b128.log is {ladder_conformer_fp16_b128_latency}, the paper value is {paper_conformer_times_data[6][1][1]}")
    conformer_times_data[7] = ('Bitter-W$_{FP8}$A$_{FP8}$', [1.7943, 58.6012])
    conformer_times_data[8] = ('Bitter-W$_{INT4}$A$_{FP16}$', [1.7471, 54.6344])
    ladder_data[1] = ladder_conformer_fp16_b128_latency

conformer_times_data[6] = ('Bitter', ladder_data)
print(f"Conformer-b1 INT8xINT4 time with kernel quantized is {conformer_times_data[7][1][0]}")
print(f"Conformer-b128 INT8xINT4 time with kernel quantized is {conformer_times_data[7][1][1]}")

print(f"Conformer-b1 INT4xINT4 time with kernel quantized is {conformer_times_data[8][1][0]}")
print(f"Conformer-b128 INT4xINT4 time with kernel quantized is {conformer_times_data[8][1][1]}")

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
amos_data = shufflenet_times_data[3][1]
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
if amos_time_b1 is not None:
    print(f"AMOS data from shufflenet_v2_b1.log is {amos_time_b1}")
    amos_data[0] = amos_time_b1

if amos_time_b128 is not None:
    print(f"AMOS data from shufflenet_v2_b128.log is {amos_time_b128}")
    amos_data[1] = amos_time_b128

shufflenet_times_data[3] = ('AMOS', amos_data)

# tensorir
def parse_tensorir_logs(log):
    pattern = r"[\d]+\.[\d]+"
    data = None
    if not os.path.exists(log):
        print(f"{log} does not exist")
        return data
    with open(log, 'r') as f:
        lines = f.readlines()
        for line in lines:
            matches = re.findall(pattern, line)
            if matches:
                data = float(matches[0])
    if data is not None:
        print(f"TensorIR data from {log} is {data}")
    return data

tensorir_data = resnet_times_data[4][1]
resnet_50_b1_latency = parse_tensorir_logs('./tensorir-benchmark/logs/resnet-50-b1-(1, 3, 224, 224)/latency.txt')
resnet_50_b128_latency = parse_tensorir_logs('./tensorir-benchmark/logs/resnet-50-(128, 3, 224, 224)/latency.txt')

if resnet_50_b1_latency is not None:
    print(f"TensorIR data from resnet-50-b1-(1, 3, 224, 224)/latency.txt is {resnet_50_b1_latency}")
    tensorir_data[0] = resnet_50_b1_latency

if resnet_50_b128_latency is not None:
    print(f"TensorIR data from resnet-50-(128, 3, 224, 224)/latency.txt is {resnet_50_b128_latency}")
    tensorir_data[1] = resnet_50_b128_latency

resnet_times_data[4] = ('TensorIR', tensorir_data)

tensorir_data = paper_shufflenet_times_data[4][1]
shuffle_net_b1_latency = parse_tensorir_logs('./tensorir-benchmark/logs/shufflenet-b1-(1, 3, 224, 224)/latency.txt')
vit_times_data[4] = paper_vit_times_data[4]
shuffle_net_b128_latency = parse_tensorir_logs('./tensorir-benchmark/logs/shufflenet-(128, 3, 224, 224)/latency.txt')

if shuffle_net_b1_latency is not None:
    print(f"TensorIR data from shufflenet-b1-(1, 3, 224, 224)-b1/latency.txt is {shuffle_net_b1_latency}")
    tensorir_data[0] = shuffle_net_b1_latency

if shuffle_net_b128_latency is not None:
    print(f"TensorIR data from shufflenet-(128, 3, 224, 224)-b128/latency.txt is {shuffle_net_b128_latency}")
    tensorir_data[1] = shuffle_net_b128_latency

shufflenet_times_data[4] = ('TensorIR', tensorir_data)


# parse vllm logs
def parse_vllm_logs(log):
    pattern = r"[\d]+\.[\d]+"
    data = None
    if not os.path.exists(log):
        return data
    with open(log, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'vllm' in line:
                matches = re.findall(pattern, line)
                if matches:
                    data = float(matches[0])
    if data is not None:
        print(f"VLLM data from {log} is {data}")
    return data

vllm_llama_fp16_b1s1_latency = parse_vllm_logs('./vllm-benchmark/logs/llama_layer1_batch1_seq1.log')
vllm_llama_fp16_b32s1_latency = parse_vllm_logs('./vllm-benchmark/logs/llama_layer1_batch32_seq1.log')
vllm_llama_fp16_b1s4096_latency = parse_vllm_logs('./vllm-benchmark/logs/llama_layer1_batch1_seq4096.log')

vllm_data = llama2_times_data[4][1]
if vllm_llama_fp16_b1s1_latency is not None:
    print(f"VLLM data from llama_layer1_batch1_seq1.log is {vllm_llama_fp16_b1s1_latency}, the paper value is {paper_llama2_times_data[4][1][0]}")
    vllm_data[0] = vllm_llama_fp16_b1s1_latency
    
if vllm_llama_fp16_b32s1_latency is not None:
    print(f"VLLM data from llama_layer1_batch32_seq1.log is {vllm_llama_fp16_b32s1_latency}, the paper value is {paper_llama2_times_data[4][1][1]}")
    vllm_data[1] = vllm_llama_fp16_b32s1_latency
    
if vllm_llama_fp16_b1s4096_latency is not None:
    print(f"VLLM data from llama_layer1_batch1_seq4096.log is {vllm_llama_fp16_b1s4096_latency}, the paper value is {paper_llama2_times_data[4][1][2]}")
    vllm_data[2] = vllm_llama_fp16_b1s4096_latency

llama2_times_data[4] = ('vLLM', vllm_data)

vllm_bloom_fp16_b1s1_latency = parse_vllm_logs('./vllm-benchmark/logs/bloom_layer1_batch1_seq1.log')
vllm_bloom_fp16_b32s1_latency = parse_vllm_logs('./vllm-benchmark/logs/bloom_layer1_batch32_seq1.log')
vllm_bloom_fp16_b1s4096_latency = parse_vllm_logs('./vllm-benchmark/logs/bloom_layer1_batch1_seq4096.log')

vllm_data = bloom_times_data[4][1]
if vllm_bloom_fp16_b1s1_latency is not None:
    print(f"VLLM data from bloom_layer1_batch1_seq1.log is {vllm_bloom_fp16_b1s1_latency}, the paper value is {paper_bloom_times_data[4][1][0]}")
    vllm_data[0] = vllm_bloom_fp16_b1s1_latency
    
if vllm_bloom_fp16_b32s1_latency is not None:
    print(f"VLLM data from bloom_layer1_batch32_seq1.log is {vllm_bloom_fp16_b32s1_latency}, the paper value is {paper_bloom_times_data[4][1][1]}")
    vllm_data[1] = vllm_bloom_fp16_b32s1_latency

if vllm_bloom_fp16_b1s4096_latency is not None:
    print(f"VLLM data from bloom_layer1_batch1_seq4096.log is {vllm_bloom_fp16_b1s4096_latency}, the paper value is {paper_bloom_times_data[4][1][2]}")
    vllm_data[2] = vllm_bloom_fp16_b1s4096_latency

bloom_times_data[4] = ('vLLM', vllm_data)

# vllm int4
vllm_llama_int4_b1s1_latency = parse_vllm_logs('./vllm-benchmark/logs/llama_layer1_batch1_seq1_int4.log')
vllm_llama_int4_b32s1_latency = parse_vllm_logs('./vllm-benchmark/logs/llama_layer1_batch32_seq1_int4.log')
vllm_llama_int4_b1s4096_latency = parse_vllm_logs('./vllm-benchmark/logs/llama_layer1_batch1_seq4096_int4.log')

vllm_data = llama2_times_data[5][1]
if vllm_llama_int4_b1s1_latency is not None:
    print(f"VLLM data from llama_layer1_batch1_seq1_int4.log is {vllm_llama_int4_b1s1_latency}, the paper value is {paper_llama2_times_data[5][1][0]}")
    vllm_data[0] = vllm_llama_int4_b1s1_latency
    
if vllm_llama_int4_b32s1_latency is not None:
    print(f"VLLM data from llama_layer1_batch32_seq1_int4.log is {vllm_llama_int4_b32s1_latency}, the paper value is {paper_llama2_times_data[5][1][1]}")
    vllm_data[1] = vllm_llama_int4_b32s1_latency
    
if vllm_llama_int4_b1s4096_latency is not None:
    print(f"VLLM data from llama_layer1_batch1_seq4096_int4.log is {vllm_llama_int4_b1s4096_latency}, the paper value is {paper_llama2_times_data[5][1][2]}")
    vllm_data[2] = vllm_llama_int4_b1s4096_latency

llama2_times_data[5] = ('vLLM-W$_{INT4}$A$_{FP16}$', vllm_data)

vllm_bloom_int4_b1s1_latency = parse_vllm_logs('./vllm-benchmark/logs/bloom_layer1_batch1_seq1_int4.log')
vllm_bloom_int4_b32s1_latency = parse_vllm_logs('./vllm-benchmark/logs/bloom_layer1_batch32_seq1_int4.log')
vllm_bloom_int4_b1s4096_latency = parse_vllm_logs('./vllm-benchmark/logs/bloom_layer1_batch1_seq4096_int4.log')

vllm_data = bloom_times_data[5][1]
if vllm_bloom_int4_b1s1_latency is not None:
    print(f"VLLM data from bloom_layer1_batch1_seq1_int4.log is {vllm_bloom_int4_b1s1_latency}, the paper value is {paper_bloom_times_data[5][1][0]}")
    vllm_data[0] = vllm_bloom_int4_b1s1_latency

if vllm_bloom_int4_b32s1_latency is not None:
    print(f"VLLM data from bloom_layer1_batch32_seq1_int4.log is {vllm_bloom_int4_b32s1_latency}, the paper value is {paper_bloom_times_data[5][1][1]}")
    vllm_data[1] = vllm_bloom_int4_b32s1_latency

if vllm_bloom_int4_b1s4096_latency is not None:
    print(f"VLLM data from bloom_layer1_batch1_seq4096_int4.log is {vllm_bloom_int4_b1s4096_latency}, the paper value is {paper_bloom_times_data[5][1][2]}")
    vllm_data[2] = vllm_bloom_int4_b1s4096_latency

bloom_times_data[5] = ('vLLM-W$_{INT4}$A$_{FP16}$', vllm_data)

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

with open("reproduce_result/data_a100.py", "w") as f:
    f.write(reproduced_results)
