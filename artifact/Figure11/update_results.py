import os
import json
import re
import logging
logger = logging.getLogger(__name__)
# set logger to show the line number
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
llama_providers = ["BS1 SEQ1", "BS32 SEQ1", "BS1 SEQ4096"]
bloom_providers = ["BS1 SEQ1", "BS32 SEQ1", "BS1 SEQ4096"]

_ = """
llama_providers = ['BS1 SEQ1', 'BS32 SEQ1', 'BS1 SEQ4096']
llama_times_data = [
    ('PyTorch-Inductor', [2700, 2624, 6878]),
    ('ONNXRuntime', [2716, 2803, 16078]),
    ('TensorRT', [5187, 4954, 6342]),
    ('vLLM', [5008, 4763, 5034]),
    ('vLLM-W$_{INT4}$A$_{FP16}$', [1123, 1100, 6128]),
    ('Welder', [2106, 2139, 6790]),
    ('Bitter', [2075, 2121, 6460]),
    ('Bitter-W$_{INT4}$A$_{FP16}$', [879, 817, 5216]),
    ('Bitter-W$_{NF4}$A$_{FP16}$', [866, 852, 5313]),
    ('Bitter-W$_{FP8}$A$_{FP8}$', [1306, 1192, 5769]),
    ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [1305, 1299, 5947]),
    ('Bitter-W$_{INT1}$A$_{INT8}$', [522, 532, 5300]),
]

bloom_providers = ['BS1 SEQ1', 'BS32 SEQ1', 'BS1 SEQ4096']
bloom_times_data = [
    ('PyTorch-Inductor', [11503, 12257, 15383]),
    ('ONNXRuntime', [7540, 7038, 62636]),
    ('TensorRT', [5566, 5875, 21209]),
    ('vLLM', [29011, 31764, 29199]),
    ('vLLM-W$_{INT4}$A$_{FP16}$', [22327, 21910, 21931]),
    ('Welder', [5130, 5036, 20109]),
    ('Bitter', [5169, 5117, 20977]),
    ('Bitter-W$_{INT4}$A$_{FP16}$', [3277, 3391, 18891]),
    ('Bitter-W$_{NF4}$A$_{FP16}$', [3374, 3374, 19772]),
    ('Bitter-W$_{FP8}$A$_{FP8}$', [4052, 3846, 18649]),
    ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [4037, 3944, 20280]),
    ('Bitter-W$_{INT1}$A$_{INT8}$', [3006, 3032, 17854]),
]
"""

exec(_)
# update the pytorch inductor results
for model in ["llama", "bloom"]:
    for batch_size, seq_len in [
            (1, 1),
            (32, 1),
            (1, 4096)
        ]:
        print(f"PyTorch-Inductor {model} batch size {batch_size} seq len {seq_len}")
        log_path = f"./logs/{model}_pytorch_b{batch_size}_s{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        print(log_path)
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            llama_times_data[0][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(llama_times_data[0][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            bloom_times_data[0][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(bloom_times_data[0][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])

# update the onnxruntime results
for model in ["llama", "bloom"]:
    for batch_size, seq_len in [
            (1, 1),
            (32, 1),
            (1, 4096)
        ]:
        print(f"OnnxRuntime {model} batch size {batch_size} seq len {seq_len}")
        log_path = f"./logs/{model}_onnxruntime_b{batch_size}_s{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            llama_times_data[1][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(llama_times_data[1][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            bloom_times_data[1][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(bloom_times_data[1][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])

# update the tensorrt results
for model in ["llama", "bloom"]:
    for batch_size, seq_len in [
            (1, 1),
            (32, 1),
            (1, 4096)
        ]:
        log_path = f"./logs/{model}_tensorrt_b{batch_size}_s{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            llama_times_data[2][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(llama_times_data[2][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            bloom_times_data[2][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(bloom_times_data[2][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])

# update the welder results
for model in ["llama", "bloom"]:
    for batch_size, seq_len in [
            (1, 1),
            (32, 1),
            (1, 4096)
        ]:
        log_path = f"./logs/{model}_welder_b{batch_size}_s{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            llama_times_data[5][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(llama_times_data[5][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            bloom_times_data[5][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(bloom_times_data[5][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])
            
# update the vllm results
for model in ["llama", "bloom"]:
    for batch_size, seq_len in [
            (1, 1),
            (32, 1),
            (1, 4096)
        ]:
        log_path = f"./logs/{model}_vllm_b{batch_size}_s{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            llama_times_data[3][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(llama_times_data[3][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            bloom_times_data[3][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(bloom_times_data[3][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])

# update the welder results
for model in ["llama", "bloom"]:
    for batch_size, seq_len in [
            (1, 1),
            (32, 1),
            (1, 4096)
        ]:
        log_path = f"./logs/{model}_ladder_b{batch_size}_s{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            llama_times_data[5][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(llama_times_data[5][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            bloom_times_data[5][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(bloom_times_data[5][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])


# update the ladder_fp16 results
for model in ["llama", "bloom"]:
    for batch_size, seq_len in [
            (1, 1),
            (32, 1),
            (1, 4096)
        ]:
        log_path = f"./logs/{model}_ladder_b{batch_size}_s{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            llama_times_data[7][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(llama_times_data[7][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            bloom_times_data[7][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(bloom_times_data[7][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])

# update the ladder_fp16_int4 results
for model in ["llama", "bloom"]:
    for batch_size, seq_len in [
            (1, 1),
            (32, 1),
            (1, 4096)
        ]:
        log_path = f"./logs/{model}_ladder_fp16_int4_b{batch_size}_s{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            llama_times_data[7][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(llama_times_data[7][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            bloom_times_data[7][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(bloom_times_data[7][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])

# update the ladder_fp16_nf4 results
for model in ["llama", "bloom"]:
    for batch_size, seq_len in [
            (1, 1),
            (32, 1),
            (1, 4096)
        ]:
        log_path = f"./logs/{model}_ladder_fp16_nf4_b{batch_size}_s{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            llama_times_data[8][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(llama_times_data[8][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            bloom_times_data[8][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(bloom_times_data[8][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])

# update the ladder_fp8_fp8 results
for model in ["llama", "bloom"]:
    for batch_size, seq_len in [
            (1, 1),
            (32, 1),
            (1, 4096)
        ]:
        log_path = f"./logs/{model}_ladder_fp8_fp8_b{batch_size}_s{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            llama_times_data[9][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(llama_times_data[9][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            bloom_times_data[9][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(bloom_times_data[9][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])

# update the ladder_fp16_mxfp8xmxfp8 results
for model in ["llama", "bloom"]:
    for batch_size, seq_len in [
            (1, 1),
            (32, 1),
            (1, 4096)
        ]:
        log_path = f"./logs/{model}_ladder_fp16_mxfp8xmxfp8_b{batch_size}_s{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            llama_times_data[10][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(llama_times_data[10][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            bloom_times_data[10][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(bloom_times_data[10][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])
            
# update the ladder_fp16_int8xint1 results
for model in ["llama", "bloom"]:
    for batch_size, seq_len in [
            (1, 1),
            (32, 1),
            (1, 4096)
        ]:
        log_path = f"./logs/{model}_ladder_fp16_int8xint1_b{batch_size}_s{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            llama_times_data[11][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(llama_times_data[11][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            bloom_times_data[11][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")] = data
            print(bloom_times_data[11][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])


# write the results to back
reproduced_results = f"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

llama_providers = {llama_providers}
llama_times_data = {llama_times_data}

bloom_providers = {bloom_providers}
bloom_times_data = {bloom_times_data}
"""

with open("reproduce_result/__init__.py", "w") as f:
    f.write(reproduced_results)
