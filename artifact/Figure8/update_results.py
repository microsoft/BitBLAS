# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import json
import re

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
    ('PyTorch-Inductor', [1.5469, 1.4564, 36.0870]),
    ('ONNXRuntime', [1.3485, 1.5380, 59.7324]),
    ('TensorRT', [1.1949, 1.3198, 50.8773]),
    ('Welder', [1.2515, 1.3723, 35.1400]),
    ('vLLM', [1.1963, 1.2767, 30.0324]),
    ('vLLM-W$_{INT4}$A$_{FP16}$', [0.6496, 1.2200, 128.5498]),
    ('Bitter', [1.0248, 1.3557, 34.7507]),
    ('Bitter-W$_{INT4}$A$_{FP16}$', [0.3563, 1.1973, 29.5409]),
    ('Bitter-W$_{NF4}$A$_{FP16}$', [0.5382, 1.3303, 30.6802]),
    ('Bitter-W$_{FP8}$A$_{FP8}$', [0.5758, 1.1959, 29.3180]),
    ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [0.8369, 1.4239, 35.8447]),
    ('Bitter-W$_{INT1}$A$_{INT8}$', [0.1629, 0.7379, 24.8855])
]

bloom_times_data = [
    ('PyTorch-Inductor', [3.1897, 3.4344, 91.9848]),
    ('ONNXRuntime', [3.1635, 3.9333, 174.4114]),
    ('TensorRT', [3.1197, 3.2010, 112.8276]),
    ('Welder', [3.0718, 3.4384, 115.7473]),
    ('vLLM', [3.0248, 3.2594, 93.5494]),
    ('vLLM-W$_{INT4}$A$_{FP16}$', [1.5464, 3.3417, 405.5566]),
    ('Bitter', [2.7872, 3.0271, 96.1634]),
    ('Bitter-W$_{INT4}$A$_{FP16}$', [0.8449, 2.2279, 91.9331]),
    ('Bitter-W$_{NF4}$A$_{FP16}$', [1.3007, 2.6248, 101.0426]),
    ('Bitter-W$_{FP8}$A$_{FP8}$', [1.5856, 2.1796, 88.7062]),
    ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [2.0269, 3.1147, 104.8811]),
    ('Bitter-W$_{INT1}$A$_{INT8}$', [0.3245, 1.2000, 70.5538])
]

resnet_times_data = [
    ('PyTorch-Inductor', [3.5764, 11.7311]),
    ('ONNXRuntime', [3.1224, 44.0471]),
    ('TensorRT', [1.3033, 12.4359]),
    ('AMOS', [2.841980, 96.208380]),
    ('TensorIR', [1.597476, 17.47901487356322]),
    ('Welder', [1.8076, 16.7814]),
    ('Bitter', [1.0877, 8.3388]),
    ('Bitter_W$_{FP8}$A$_{FP8}$', [1.1718, 7.7374]),
    ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [2.0571, 9.2503]),
    ('Bitter_W$_{INT1}$A$_{INT4}$', [1.1516, 7.1390])
]

shufflenet_times_data = [
    ('PyTorch-Inductor', [4.1854, 4.0865]),
    ('ONNXRuntime', [1.9846, 7.4959]),
    ('TensorRT', [1.1394, 5.3273]),
    ('AMOS', [0.7063, 21.2220]),
    ('TensorIR', [0.5235584, 5.187201209621993]),
    ('Welder', [0.3597, 3.9318]),
    ('Bitter', [0.3097, 3.2603]),
    ('Bitter_W$_{FP8}$A$_{FP8}$', [0.3102, 3.4112])
]

conformer_times_data = [
    ('PyTorch-Inductor', [7.9124, 70.3407]),
    ('ONNXRuntime', [6.1475, 218.4175]),
    ('TensorRT', [2.0827, 62.5842]),
    ('AMOS', [0.0, 0.0]),
    ('TensorIR', [0.0, 0.0]),
    ('Welder', [1.9198, 88.3134]),
    ('Bitter', [2.1430, 59.4452]),
    ('Bitter-W$_{INT4}$A$_{INT8}$', [1.7943, 58.6012]),
    ('Bitter-W$_{INT4}$A$_{INT4}$', [1.7471, 54.6344])
]

vit_times_data = [
    ('PyTorch-Inductor', [3.5411, 4.7605]),
    ('ONNXRuntime', [2.5414, 12.5239]),
    ('TensorRT', [0.6701, 2.9387]),
    ('AMOS', [0.0, 0.0]),
    ('TensorIR', [1.2807664, 6.145351825]),
    ('Welder', [1.1366, 5.2987]),
    ('Bitter', [1.1806, 4.4487]),
    ('Bitter-W$_{FP8}$A$_{FP8}$', [1.2695, 4.0975]),
    ('Bitter-W$_{INT4}$A$_{INT4}$', [1.1856, 3.4475])
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

welder_times_data = [1.2515, 1.3723, 35.1400]
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

welder_times_data = [3.0718, 3.4384, 115.7473]    
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

welder_times_data = [1.8076, 16.7814]
layency_resnet_b1 = get_welder_results('./welder-benchmark/compiled_models/resnet50_b1_cutlass/run.log')
layency_resnet_b128 = get_welder_results('./welder-benchmark/compiled_models/resnet50_b128_cutlass/run.log')

if layency_resnet_b1 is not None:
    welder_times_data[0] = layency_resnet_b1
if layency_resnet_b128 is not None:
    welder_times_data[1] = layency_resnet_b128

resnet_times_data[3] = ('Welder', welder_times_data)

welder_times_data = [0.3597, 3.9318]
layency_shufflenet_b1 = get_welder_results('./welder-benchmark/compiled_models/shufflenet_v2_b1_cutlass/run.log')
layency_shufflenet_b128 = get_welder_results('./welder-benchmark/compiled_models/shufflenet_v2_b128_cutlass/run.log')
if layency_shufflenet_b1 is not None:
    welder_times_data[0] = layency_shufflenet_b1
if layency_shufflenet_b128 is not None:
    welder_times_data[1] = layency_shufflenet_b128

shufflenet_times_data[3] = ('Welder', welder_times_data)

welder_times_data = [1.9198, 88.3134]
layency_conformer_b1 = get_welder_results('./welder-benchmark/compiled_models/Conformer_b1_cutlass/run.log')
layency_conformer_b128 = get_welder_results('./welder-benchmark/compiled_models/Conformer_b128_cutlass/run.log')

if layency_conformer_b1 is not None:
    welder_times_data[0] = layency_conformer_b1
if layency_conformer_b128 is not None:
    welder_times_data[1] = layency_conformer_b128

conformer_times_data[3] = ('Welder', welder_times_data)

welder_times_data = [1.1366, 5.2987]
layency_vit_b1 = get_welder_results('./welder-benchmark/compiled_models/vit_b1_cutlass/run.log')
layency_vit_b128 = get_welder_results('./welder-benchmark/compiled_models/vit_b128_cutlass/run.log')
if layency_vit_b1 is not None:
    welder_times_data[0] = layency_vit_b1
if layency_vit_b128 is not None:
    welder_times_data[1] = layency_vit_b128

vit_times_data[3] = ('Welder', welder_times_data)

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
ladder_data = [1.0248, 1.3557, 34.7507]
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

ladder_data = [0.3563, 1.1973, 29.5409]
ladder_llama_int4_b1s1_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b1_s1_q0_b4.log')
ladder_llama_int4_b32s1_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b32_s1_q0_b4.log')
ladder_llama_int4_b1s4096_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b1_s4096_q0_b4.log')

if ladder_llama_int4_b1s1_latency is not None:
    ladder_data[0] = ladder_llama_int4_b1s1_latency

if ladder_llama_int4_b32s1_latency is not None:
    ladder_data[1] = ladder_llama_int4_b32s1_latency

llama2_times_data[7] = ('Bitter-W$_{INT4}$A$_{FP16}$', ladder_data)


ladder_data = [0.5382, 1.3303, 30.6802]
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
ladder_data = [0.5758, 1.1959, 29.3180]
ladder_llama_fp8_b1s1_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b1_s1_q0_fp_e5m2.log')
ladder_llama_fp8_b32s1_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b32_s1_q0_fp_e5m2.log')
ladder_llama_fp8_b1s4096_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b1_s4096_q0_fp_e5m2.log')

if ladder_llama_fp8_b1s1_latency is not None:
    ladder_data[0] = ladder_llama_fp8_b1s1_latency
if ladder_llama_fp8_b32s1_latency is not None:
    ladder_data[1] = ladder_llama_fp8_b32s1_latency


llama2_times_data[9] = ('Bitter-W$_{FP8}$A$_{FP16}$', ladder_data)

# mxfp8
ladder_data = [0.8369, 1.4239, 35.8447]
ladder_llama_mxfp8_b1s1_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b1_s1_q0_fp_mxfp8.log')
ladder_llama_mxfp8_b32s1_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b32_s1_q0_mxfp8.log')
ladder_llama_mxfp8_b1s4096_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b1_s4096_q0_fp_mxfp8.log')

if ladder_llama_mxfp8_b1s1_latency is not None:
    ladder_data[0] = ladder_llama_mxfp8_b1s1_latency
if ladder_llama_mxfp8_b32s1_latency is not None:
    ladder_data[1] = ladder_llama_mxfp8_b32s1_latency

llama2_times_data[10] = ('Bitter-W$_{MXFP8}$A$_{FP16}$', ladder_data)

# int8xint1
ladder_data = [0.1629, 0.7379, 24.8855]
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

ladder_data = [2.7872, 3.0271, 96.1634]
ladder_bloom_fp16_b1s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s1_q-1.log')
ladder_bloom_fp16_b32s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b32_s1_q0_b1_int.log')
ladder_bloom_fp16_b1s4096_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s4096_q-1.log')

if ladder_bloom_fp16_b1s1_latency is not None:
    ladder_data[0] = ladder_bloom_fp16_b1s1_latency
if ladder_bloom_fp16_b32s1_latency is not None:
    ladder_data[1] = ladder_bloom_fp16_b32s1_latency
if ladder_bloom_fp16_b1s4096_latency is not None:
    ladder_data[2] = ladder_bloom_fp16_b1s4096_latency

bloom_times_data[6] = ('Bitter', ladder_data)

ladder_data = [0.8449, 2.2279, 91.9331]
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

ladder_data = [1.3007, 2.6248, 101.0426]
# nf4
ladder_bloom_nf4_b1s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s1_q0_nf4.log')
ladder_bloom_nf4_b32s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b32_s1_q0_nf4.log')
ladder_bloom_nf4_b1s4096_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s4096_q0_nf4.log')

if ladder_bloom_nf4_b1s1_latency is not None:
    ladder_data[0] = ladder_bloom_nf4_b1s1_latency
if ladder_bloom_nf4_b32s1_latency is not None:
    ladder_data[1] = ladder_bloom_nf4_b32s1_latency

bloom_times_data[8] = ('Bitter-W$_{NF4}$A$_{FP16}$', ladder_data)

# fp8
ladder_data = [1.5856, 2.1796, 88.7062]

ladder_bloom_fp8_b1s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s1_q0_fp_e5m2.log')
ladder_bloom_fp8_b32s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b32_s1_q0_fp_e5m2.log')
ladder_bloom_fp8_b1s4096_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s4096_q0_fp_e5m2.log')

if ladder_bloom_fp8_b1s1_latency is not None:
    ladder_data[0] = ladder_bloom_fp8_b1s1_latency
if ladder_bloom_fp8_b32s1_latency is not None:
    ladder_data[1] = ladder_bloom_fp8_b32s1_latency

bloom_times_data[9] = ('Bitter-W$_{FP8}$A$_{FP16}$', ladder_data)

# mxfp8
ladder_data = [2.0269, 3.1147, 104.8811]
ladder_bloom_mxfp8_b1s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s1_q0_fp_mxfp8.log')
ladder_bloom_mxfp8_b32s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b32_s1_q0_mxfp8.log')
ladder_bloom_mxfp8_b1s4096_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s4096_q0_fp_mxfp8.log')

if ladder_bloom_mxfp8_b1s1_latency is not None:
    ladder_data[0] = ladder_bloom_mxfp8_b1s1_latency
if ladder_bloom_mxfp8_b32s1_latency is not None:
    ladder_data[1] = ladder_bloom_mxfp8_b32s1_latency

bloom_times_data[10] = ladder_data

# int8xint1
ladder_data = [0.3245, 1.2000, 70.5538]
ladder_bloom_int4_b1s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s1_q0_b1_int.log')
ladder_bloom_int4_b32s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b32_s1_q0_b1_int.log')
ladder_bloom_int4_b1s4096_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s4096_q0_b1_int.log')

if ladder_bloom_int4_b1s1_latency is not None:
    ladder_data[0] = ladder_bloom_int4_b1s1_latency
if ladder_bloom_int4_b32s1_latency is not None:
    ladder_data[1] = ladder_bloom_int4_b32s1_latency
    
bloom_times_data[11] = ('Bitter-W$_{INT1}$A$_{INT8}$', ladder_data)

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
