# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import json
import re
# for comparation with paper results
from paper_result.data_v100 import(
    llama2_times_data as paper_llama2_times_data,
    bloom_times_data as paper_bloom_times_data,
    resnet_times_data as paper_resnet_times_data,
    shufflenet_times_data as paper_shufflenet_times_data,
    conformer_times_data as paper_conformer_times_data,
    vit_times_data as paper_vit_times_data
)

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
llama2_times_data =  [('PyTorch-Inductor', [3.217925483466279, 3.250928816250611, 103.99388374821967]), ('ONNXRuntime', [3.614401546176978, 4.332901872046012, 144.95905208116096]), ('TensorRT', [2.1169786556486434, 2.2473483984504883, 116.90785335360862]), ('Welder', [2.1299953029240504, 2.4514724844885087, 90.9227194881588]), ('vLLM', [2.400138947869697, 2.4094428470406832, 90.9705400867396]), ('vLLM-W$_{INT4}$A$_{FP16}$', [0, 0, 0]), ('Bitter', [1.9765955517095652, 2.3862742926098126, 103.25357669486861]), ('Bitter-W$_{INT4}$A$_{FP16}$', [0.5810565043967766, 2.114683611788609, 98.21085289027624]), ('Bitter-W$_{NF4}$A$_{FP16}$', [0.8633100238071868, 2.2646670155225506, 111.32034214990733]), ('Bitter-W$_{FP8}$A$_{FP8}$', [1.1075113660212497, 3.526799336756904, 221.7212100322498]), ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [1.0806980143900593, 10.758993782700431, 685.274998177604]), ('Bitter-W$_{INT1}$A$_{INT8}$', [0.2086978826907124, 5.115420119061646, 541.6899519031156])]
bloom_times_data =  [('PyTorch-Inductor', [8.657268412992387, 8.983035677418954, 0]), ('ONNXRuntime', [0, 0, 0]), ('TensorRT', [6.007524302481306, 6.442180642767218, 0]), ('Welder', [5.666955948642983, 6.37217241958767, 0]), ('vLLM', [0, 0, 0]), ('vLLM-W$_{INT4}$A$_{FP16}$', [0, 0, 0]), ('Bitter', [5.714473599122273, 6.204192202515962, 0]), ('Bitter-W$_{INT4}$A$_{FP16}$', [1.5101099639320685, 3.996296767950289, 0]), ('Bitter-W$_{NF4}$A$_{FP16}$', [1.6706774922279228, 5.112599927299912, 0]), ('Bitter-W$_{FP8}$A$_{FP8}$', [2.7696936107561574, 5.473647132656499, 0]), ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [3.324084594332085, 3.211721584895696, 0]), ('Bitter-W$_{INT1}$A$_{INT8}$', [0.44308276205063174, 5.91215230870652, 0])]
resnet_times_data =  [('PyTorch-Inductor', [4.71939975312213, 24.984028071034068]), ('ONNXRuntime', [3.8971506757974175, 96.10264614153716]), ('TensorRT', [1.9369065880549199, 18.83340005786735]),  ('AMOS', [3.954496, 35.31771]),('TensorIR', [0.411856816, 7.081917907]), ('Welder', [2.6216176798756496, 40.489245658554516]), ('Bitter', [1.464385492448934, 21.511389008674993]), ('Bitter_W$_{FP8}$A$_{FP8}$', [1.4457214321556482, 22.034315429560362]), ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [2.4282925373712705, 89.97348411429944]), ('Bitter_W$_{INT1}$A$_{INT4}$', [1.4294742056218086, 21.748490902178354])]
shufflenet_times_data =  [('PyTorch-Inductor', [6.174556211528089, 6.374341899506123]), ('ONNXRuntime', [2.715021076875526, 13.3670540003058]), ('TensorRT', [1.299584125822645, 5.08656446120813]), ('AMOS', [4.060608798988117, 33.75784931378247]), ('TensorIR', [0.412144410160318, 6.734246573237521]), ('Welder', [0.416140120500671, 6.300909695260738]), ('Bitter', [0.40252925253302324, 5.3410948419178865]), ('Bitter_W$_{FP8}$A$_{FP8}$', [0.4007838428362482, 5.459490632602676])]
conformer_times_data =  [('PyTorch-Inductor', [13.947573255178938, 160.57654073242998]), ('ONNXRuntime', [9.857220827971753, 401.8885575095535]), ('TensorRT', [3.3826668337473196, 163.57960654354173]), ('AMOS', [0, 0]), ('TensorIR', [0, 0]), ('Welder', [3.9136361892113847, 181.31992197027324]), ('Bitter', [3.658820972673263, 183.98696718444018]), ('Bitter-W$_{INT4}$A$_{INT8}$', [3.504429485922558, 199.95538707229528]), ('Bitter-W$_{INT4}$A$_{INT4}$', [3.509259232377321, 202.45478104052762])]
vit_times_data =  [('PyTorch-Inductor', [5.276245390983855, 7.889163536031212]), ('ONNXRuntime', [3.4162979084212943, 23.576996522824217]), ('TensorRT', [1.115957732756058, 8.374899504321602]), ('AMOS', [0, 0]), ('TensorIR', [1.179153433, 14.82752]), ('Welder', [1.335043164379538, 8.112091832528904]), ('Bitter', [1.3262046896436848, 9.722254590477919]), ('Bitter-W$_{FP8}$A$_{FP8}$', [1.393417868754272, 9.346703643276202]), ('Bitter-W$_{INT4}$A$_{INT4}$', [1.386690188366176, 9.465814249063042])]
"""


exec(_)

llama2_times_data = [
    ('PyTorch-Inductor', [-1, -1, -1]),
    ('ONNXRuntime', [-1, -1, -1]),
    ('TensorRT', [-1, -1, -1]),
    ('Welder', [-1, -1, -1]),
    ('vLLM', [-1, -1, -1]),
    ('vLLM-W$_{INT4}$A$_{FP16}$', [0, 0, 0]),
    ('Bitter', [-1, -1, -1]),
    ('Bitter-W$_{INT4}$A$_{FP16}$', [-1, -1, -1]),
    ('Bitter-W$_{NF4}$A$_{FP16}$', [-1, -1, -1]),
    ('Bitter-W$_{FP8}$A$_{FP8}$', [-1, -1, -1]),
    ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [-1, -1, -1]),
    ('Bitter-W$_{INT1}$A$_{INT8}$', [-1, -1, -1])
]

bloom_times_data = [
    ('PyTorch-Inductor', [-1, -1, 0]),
    ('ONNXRuntime', [-1, -1, 0]),
    ('TensorRT', [-1, -1, 0]),
    ('Welder', [-1, -1, 0]),
    ('vLLM', [0, 0, 0]),
    ('vLLM-W$_{INT4}$A$_{FP16}$', [0, 0,0 ]),
    ('Bitter', [-1, -1, 0]),
    ('Bitter-W$_{INT4}$A$_{FP16}$', [-1, -1, 0]),
    ('Bitter-W$_{NF4}$A$_{FP16}$', [-1, -1, 0]),
    ('Bitter-W$_{FP8}$A$_{FP8}$', [-1, -1, 0]),
    ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [-1, -1, 0]),
    ('Bitter-W$_{INT1}$A$_{INT8}$', [-1, -1, 0])
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

bloom_times_data[0] = ('Pytorch Inductor', [pytorch_time_b1, pytorch_time_b32, 0])

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

bloom_times_data[1] = ('ONNXRuntime', [onnx_time_b1, onnx_time_b32, 0])

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

bloom_times_data[2] = ('TensorRT', [tensorrt_time_b1, tensorrt_time_b32, 0])

## update welder results

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

# welder_times_data = [3.0718, 3.4384, 0]
welder_times_data = [-1, -1, 0]
latency_bloom_b1s1 = get_welder_results('./welder-benchmark/compiled_models/bloom-176b_layer1_seq1_bs1_cutlass/run.log')
latency_bloom_b1s32 = get_welder_results('./welder-benchmark/compiled_models/bloom-176b_layer1_seq1_bs32_cutlass/run.log')

if latency_bloom_b1s1 is not None:
    welder_times_data[0] = latency_bloom_b1s1
if latency_bloom_b1s32 is not None:
    welder_times_data[1] = latency_bloom_b1s32

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


llama2_times_data[9] = ('Bitter-W$_{FP8}$A$_{FP8}$', ladder_data)

# mxfp8
# ladder_data = [0.8369, 1.4239, 35.8447]
ladder_data = [-1, -1, -1]
ladder_llama_mxfp8_b1s1_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b1_s1_q0_mxfp8.log')
ladder_llama_mxfp8_b32s1_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b32_s1_q0_mxfp8.log')
ladder_llama_mxfp8_b1s4096_latency = parse_ladder_logs('./ladder-benchmark/llama2-70b_b1_s4096_q0_mxfp8.log')

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
ladder_data = [-1, -1, 0]
ladder_bloom_fp8_b1s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b1_s1_q0_fp_e5m2.log')
ladder_bloom_fp8_b32s1_latency = parse_ladder_logs('./ladder-benchmark/bloom-176b_b32_s1_q0_fp_e5m2.log')

if ladder_bloom_fp8_b1s1_latency is not None:
    ladder_data[0] = ladder_bloom_fp8_b1s1_latency
if ladder_bloom_fp8_b32s1_latency is not None:
    ladder_data[1] = ladder_bloom_fp8_b32s1_latency

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
ladder_data = [-1, -1, 0]
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

ladder_data = vit_times_data[7][1]
ladder_vit_fp16_b1_latency = parse_ladder_logs('./ladder-benchmark/logs/vit-b1_fp8_e5m2.log')
ladder_vit_fp16_b128_latency = parse_ladder_logs('./ladder-benchmark/logs/vit-b128_fp8_e5m2.log')

if ladder_vit_fp16_b1_latency is not None:
    print(f"Ladder data from vit-b1.log is {ladder_vit_fp16_b1_latency}, the paper value is {paper_vit_times_data[7][1][0]}")
    ladder_data[0] = ladder_vit_fp16_b1_latency
else:
    raise ValueError("ladder_vit_fp16_b1_latency is None")
if ladder_vit_fp16_b128_latency is not None:
    print(f"Ladder data from vit-b128.log is {ladder_vit_fp16_b128_latency}, the paper value is {paper_vit_times_data[7][1][1]}")
    ladder_data[1] = ladder_vit_fp16_b128_latency
else:
    raise ValueError("ladder_vit_fp16_b128_latency is None")

vit_times_data[7] = ('Bitter-W$_{FP8}$A$_{FP8}$', ladder_data)

ladder_data = vit_times_data[8][1]
ladder_vit_fp16_b1_latency = parse_ladder_logs('./ladder-benchmark/logs/vit-b1_int4b.log')
ladder_vit_fp16_b128_latency = parse_ladder_logs('./ladder-benchmark/logs/vit-b128_int4b.log')
if ladder_vit_fp16_b1_latency is not None:
    print(f"Ladder data from vit-b1.log is {ladder_vit_fp16_b1_latency}, the paper value is {paper_vit_times_data[7][1][0]}")
    ladder_data[0] = ladder_vit_fp16_b1_latency

if ladder_vit_fp16_b128_latency is not None:
    print(f"Ladder data from vit-b128.log is {ladder_vit_fp16_b128_latency}, the paper value is {paper_vit_times_data[7][1][1]}")
    ladder_data[1] = ladder_vit_fp16_b128_latency

vit_times_data[8] = ('Bitter-W$_{INT4}$A$_{INT4}$', ladder_data)

# conformer

ladder_data = conformer_times_data[6][1]
ladder_conformer_fp16_b1_latency = parse_ladder_logs('./ladder-benchmark/logs/Conformer-b1.log')
ladder_conformer_fp16_b128_latency = parse_ladder_logs('./ladder-benchmark/logs/Conformer-b128.log')

if ladder_conformer_fp16_b1_latency is not None:
    print(f"Ladder data from Conformer_b1.log is {ladder_conformer_fp16_b1_latency}, the paper value is {paper_conformer_times_data[6][1][0]}")
    ladder_data[0] = ladder_conformer_fp16_b1_latency

if ladder_conformer_fp16_b128_latency is not None:
    print(f"Ladder data from Conformer_b128.log is {ladder_conformer_fp16_b128_latency}, the paper value is {paper_conformer_times_data[6][1][1]}")
    ladder_data[1] = ladder_conformer_fp16_b128_latency

conformer_times_data[6] = ('Bitter', ladder_data)

ladder_data = conformer_times_data[7][1]
ladder_conformer_fp16_b1_latency = parse_ladder_logs('./ladder-benchmark/logs/Conformer-b1_int8xint4.log')
ladder_conformer_fp16_b128_latency = parse_ladder_logs('./ladder-benchmark/logs/Conformer-b128_int8xint4.log')

if ladder_conformer_fp16_b1_latency is not None:
    print(f"Ladder data from Conformer_b1_fp8_e5m2.log is {ladder_conformer_fp16_b1_latency}, the paper value is {paper_conformer_times_data[7][1][0]}")
    ladder_data[0] = ladder_conformer_fp16_b1_latency

if ladder_conformer_fp16_b128_latency is not None:
    print(f"Ladder data from Conformer_b128_fp8_e5m2.log is {ladder_conformer_fp16_b128_latency}, the paper value is {paper_conformer_times_data[7][1][1]}")
    ladder_data[1] = ladder_conformer_fp16_b128_latency

conformer_times_data[7] = ('Bitter-W$_{INT4}$A$_{INT8}$', ladder_data)

ladder_data = conformer_times_data[8][1]
ladder_conformer_fp16_b1_latency = parse_ladder_logs('./ladder-benchmark/logs/Conformer-b1_int4bxint1.log')
ladder_conformer_fp16_b128_latency = parse_ladder_logs('./ladder-benchmark/logs/Conformer-b128_int4bxint1.log')

if ladder_conformer_fp16_b1_latency is not None:
    print(f"Ladder data from Conformer_b1_int4xint1.log is {ladder_conformer_fp16_b1_latency}, the paper value is {paper_conformer_times_data[8][1][0]}")
    ladder_data[0] = ladder_conformer_fp16_b1_latency

if ladder_conformer_fp16_b128_latency is not None:
    print(f"Ladder data from Conformer_b128_int4xint1.log is {ladder_conformer_fp16_b128_latency}, the paper value is {paper_conformer_times_data[8][1][1]}")
    ladder_data[1] = ladder_conformer_fp16_b128_latency

conformer_times_data[8] = ('Bitter-W$_{INT4}$A$_{INT4}$', ladder_data)

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

tensorir_data = paper_resnet_times_data[4][1]
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
# print(amos_time_b1)

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

# if amos_time_b1 is not None:
#     print(f"AMOS data from shufflenet_v2_b1.log is {amos_time_b1}")
#     amos_data[0] = amos_time_b1

if amos_time_b128 is not None:
    print(f"AMOS data from shufflenet_v2_b128.log is {amos_time_b128}")
    amos_data[1] = amos_time_b128

shufflenet_times_data[3] = paper_shufflenet_times_data[3]

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

with open("reproduce_result/data_v100.py", "w") as f:
    f.write(reproduced_results)
