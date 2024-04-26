import os
import json
import re
from reproduce_result import (
    compilation_cost
)

# update amos tune time
resnet_50_b1_logs = './amos-benchmark/logs/resnet50_b1.log'
### match x(float) from Time taken:  231111.167464733124
pattern = r"Time taken:  [\d]+\.[\d]+"
with open(resnet_50_b1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            amos_time_b1 = float(matches[0].split(' ')[-1])
print(amos_time_b1)
compilation_cost["ResNet(1)"]["AMOS"] = amos_time_b1 / 3600

resnet_50_b128_logs = './amos-benchmark/logs/resnet50_b128.log'
### match x(float) from Time taken:  231111.167464733124
pattern = r"Time taken:  [\d]+\.[\d]+"
with open(resnet_50_b128_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            amos_time_b128 = float(matches[0].split(' ')[-1])
print(amos_time_b128)
compilation_cost["ResNet(128)"]["AMOS"] = amos_time_b128 / 3600

shufflenet_b1_logs = './amos-benchmark/logs/shufflenet_v2_b1.log'
### match x(float) from Time taken:  231111.167464733124
pattern = r"Time taken:  [\d]+\.[\d]+"
with open(shufflenet_b1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            amos_time_b1 = float(matches[0].split(' ')[-1])
print(amos_time_b1)
compilation_cost["ShuffleNet(1)"]["AMOS"] = amos_time_b1

shufflenet_b128_logs = './amos-benchmark/logs/shufflenet_v2_b128.log'
### match x(float) from Time taken:  231111.167464733124
pattern = r"Time taken:  [\d]+\.[\d]+"
with open(shufflenet_b128_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            amos_time_b128 = float(matches[0].split(' ')[-1])
print(amos_time_b128)

compilation_cost["ShuffleNet(128)"]["AMOS"] = amos_time_b128 / 3600

# update tensorir tune time
resnet_50_b1_logs = './tensorir-benchmark/logs/resnet-50-b1-(1, 3, 224, 224)/cost_time.txt'
### match x(float) from 231111.167464733124
pattern = r"[\d]+\.[\d]+"
with open(resnet_50_b1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            tensorir_time_b1 = float(matches[0])
print(tensorir_time_b1)

compilation_cost["ResNet(1)"]["TensorIR"] = tensorir_time_b1 / 3600

resnet_50_b128_logs = './tensorir-benchmark/logs/resnet-50-b128-(128, 3, 224, 224)/cost_time.txt'
### match x(float) from 231111.167464733124
pattern = r"[\d]+\.[\d]+"
with open(resnet_50_b128_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            tensorir_time_b128 = float(matches[0])
print(tensorir_time_b128)

compilation_cost["ResNet(128)"]["TensorIR"] = tensorir_time_b128 / 3600

shufflenet_b1_logs = './tensorir-benchmark/logs/shufflenet-b1-(1, 3, 224, 224)/cost_time.txt'
### match x(float) from 231111.167464733124
pattern = r"[\d]+\.[\d]+"
with open(shufflenet_b1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            tensorir_time_b1 = float(matches[0])
print(tensorir_time_b1)

compilation_cost["ShuffleNet(1)"]["TensorIR"] = tensorir_time_b1 / 3600

shufflenet_b128_logs = './tensorir-benchmark/logs/shufflenet-b128-(128, 3, 224, 224)/cost_time.txt'
### match x(float) from 231111.167464733124
pattern = r"[\d]+\.[\d]+"
with open(shufflenet_b128_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            tensorir_time_b128 = float(matches[0])
print(tensorir_time_b128)

compilation_cost["ShuffleNet(128)"]["TensorIR"] = tensorir_time_b128 / 3600


# update welder tune time
resnet_50_b1_logs = './welder-benchmark/compiled_models/resnet-50-b1_cutlass/tune_time_cost.txt'
### match x(float) Compiler tuning time: x seconds
pattern = r"Compiler tuning time: [\d]+ seconds"
with open(resnet_50_b1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            welder_time_b1 = int(matches[0].split(' ')[-2])
print(welder_time_b1)

compilation_cost["ResNet(1)"]["Welder"] = welder_time_b1 / 60

resnet_50_b128_logs = './welder-benchmark/compiled_models/resnet-50-b128_cutlass/tune_time_cost.txt'
### match x(float) Compiler tuning time: x seconds
pattern = r"Compiler tuning time: [\d]+ seconds"
with open(resnet_50_b128_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            welder_time_b128 = int(matches[0].split(' ')[-2])
print(welder_time_b128)

compilation_cost["ResNet(128)"]["Welder"] = welder_time_b128 / 60

shufflenet_b1_logs = './welder-benchmark/compiled_models/shufflenet-b1_cutlass/tune_time_cost.txt'
### match x(float) Compiler tuning time: x seconds
pattern = r"Compiler tuning time: [\d]+ seconds"
with open(shufflenet_b1_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            welder_time_b1 = int(matches[0].split(' ')[-2])
print(welder_time_b1)

compilation_cost["ShuffleNet(1)"]["Welder"] = welder_time_b1 / 60

shufflenet_b128_logs = './welder-benchmark/compiled_models/shufflenet-b128_cutlass/tune_time_cost.txt'
### match x(float) Compiler tuning time: x seconds
pattern = r"Compiler tuning time: [\d]+ seconds"
with open(shufflenet_b128_logs, 'r') as f:
    lines = f.readlines()
    for line in lines:
        matches = re.findall(pattern, line)
        if matches:
            welder_time_b128 = int(matches[0].split(' ')[-2])
print(welder_time_b128)

compilation_cost["ShuffleNet(128)"]["Welder"] = welder_time_b128 / 60

# write the results to back
reproduced_results = f"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
compilation_cost = {json.dumps(compilation_cost, indent=4)}
"""

with open("reproduce_result/__init__.py", "w") as f:
    f.write(reproduced_results)
