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

_ = """llama2_times_data =  [('PyTorch-Inductor', [1.5468936373368576, 1.4563860399614914, 36.08702548690121]), ('ONNXRuntime', [1.3485272031668207, 1.5380288059376974, 59.73235271582186]), ('TensorRT', [1.1949333455566484, 1.319818453378793, 50.87728532938403]), ('Welder', [1.2514941946475322, 1.3723150443469017, 35.13996875196021]), ('vLLM', [1.1963413512661067, 1.2766941033451227, 30.032390864813685]), ('vLLM-W$_{INT4}$A$_{FP16}$', [0.6495653644072962, 1.2200218504932787, 128.54981616564038]), ('Bitter', [1.0248274796549177, 1.3556864169307128, 34.75069455940182]), ('Bitter-W$_{INT4}$A$_{FP16}$', [0.3563383792078014, 1.197254777348678, 29.54092181273579]), ('Bitter-W$_{NF4}$A$_{FP16}$', [0.5381690312458812, 1.330293215031377, 30.680200448940727]), ('Bitter-W$_{FP8}$A$_{FP8}$', [0.5758475538248041, 1.1958631775303261, 29.317950632962464]), ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [0.8368904453263336, 1.423887949092845, 35.8446755111233]), ('Bitter-W$_{INT1}$A$_{INT8}$', [0.16287542740644978, 0.7378902308483449, 24.885478034096305])]
bloom_times_data =  [('PyTorch-Inductor', [3.189747997477411, 3.4344147463434096, 91.98476351139533]), ('ONNXRuntime', [3.1635066233862768, 3.9332821835839726, 174.41136979254262]), ('TensorRT', [3.119738672928354, 3.201007301127645, 112.82756304184876]), ('Welder', [3.071796604775635, 3.4383784707688876, 115.74725651604635]), ('vLLM', [3.024757188534925, 3.2594330041327844, 93.5494024864804]), ('vLLM-W$_{INT4}$A$_{FP16}$', [1.5463650957362058, 3.3417304515826607, 405.55663419419864]), ('Bitter', [2.787216804503503, 3.0271288920696, 96.16338042974515]), ('Bitter-W$_{INT4}$A$_{FP16}$', [0.8448733082056081, 2.2279171705995333, 91.93311378563052]), ('Bitter-W$_{NF4}$A$_{FP16}$', [1.3007417077861743, 2.624782902999615, 101.04255790956513]), ('Bitter-W$_{FP8}$A$_{FP8}$', [1.5856014411942725, 2.1796481149647273, 88.70622598552232]), ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [2.0269489092372495, 3.114670428415609, 104.88105487968816]), ('Bitter-W$_{INT1}$A$_{INT8}$', [0.32450915520558976, 1.2000373553507777, 70.55376986939835])]
resnet_times_data =  [('PyTorch-Inductor', [3.5764438915793324, 11.731084203003954]), ('ONNXRuntime', [3.1224174903266175, 44.047093715791846]), ('TensorRT', [1.303328961537124, 12.435869377576346]), ('AMOS', [2.8052762350893135, 100.72355763440173]), ('TensorIR', [1.5436207364614656, 17.644117488308787]), ('Welder', [1.8075741154542282, 16.78140060043219]), ('Bitter', [1.0877123191767235, 8.338801787380024]), ('Bitter_W$_{FP8}$A$_{FP8}$', [1.1718401741139368, 7.737413824399603]), ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [2.0570725991011405, 9.250335783388586]), ('Bitter_W$_{INT1}$A$_{INT4}$', [1.1515690838409844, 7.139044037606482])]
shufflenet_times_data =  [('PyTorch-Inductor', [4.185364982961789, 4.086464086078921]), ('ONNXRuntime', [1.984645792635625, 7.495885504928249]), ('TensorRT', [1.1394181306028839, 5.32727285341519]), ('AMOS', [0.7063347551002459, 21.221955349807672]), ('TensorIR', [0.5188146049122749, 5.182191123843947]), ('Welder', [0.359720639030162, 3.9317926459902814]), ('Bitter', [0.3096727244151164, 3.2603220749864317]), ('Bitter_W$_{FP8}$A$_{FP8}$', [0.31015279882999863, 3.4111766170492266])]
conformer_times_data =  [('PyTorch-Inductor', [7.912432680230362, 70.34074611467969]), ('ONNXRuntime', [6.147537163074164, 218.41752514830344]), ('TensorRT', [2.0826648179668354, 62.58422399683524]), ('AMOS', [0, 0]), ('TensorIR', [0, 0]), ('Welder', [1.919754239892051, 88.31338202140643]), ('Bitter', [2.1429868078787124, 59.445192614534676]), ('Bitter-W$_{INT4}$A$_{INT8}$', [1.7943354531746427, 58.60116943804335]), ('Bitter-W$_{INT4}$A$_{INT4}$', [1.747114463661884, 54.63436613397169])]
vit_times_data =  [('PyTorch-Inductor', [3.541120318819185, 4.7605174347845916]), ('ONNXRuntime', [2.541391061274762, 12.523887679190185]), ('TensorRT', [0.6701239741174398, 2.938699620662315]), ('AMOS', [0, 0]), ('TensorIR', [1.3016663538881768, 6.1613429624926015]), ('Welder', [1.1366134821143112, 5.298662312437655]), ('Bitter', [1.1805645158166256, 4.448728137453351]), ('Bitter-W$_{FP8}$A$_{FP8}$', [1.2695404242205228, 4.0974862609251295]), ('Bitter-W$_{INT4}$A$_{INT4}$', [1.1855693623392323, 3.4475477614455006])]
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
