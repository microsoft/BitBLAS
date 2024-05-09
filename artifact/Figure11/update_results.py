import os
import json
import re
llama_providers = ["BS1 SEQ1", "BS32 SEQ1", "BS1 SEQ4096"]
bloom_providers = ["BS1 SEQ1", "BS32 SEQ1", "BS1 SEQ4096"]

_ = """
llama_times_data = [
    ("PyTorch-Inductor", [2660, 2642, 6754]),
    ("ONNXRuntime", [2748, 2780, 16206]),
    ("TensorRT", [5140, 5148, 6260]),
    ("vLLM", [4866, 4868, 4866]),
    ("vLLM-W$_{INT4}$A$_{FP16}$", [1072, 1072, 6400]),
    ("Welder", [2076, 2084, 6626]),
    ("Bitter", [2064, 2070, 6580]),
    ("Bitter-W$_{INT4}$A$_{FP16}$", [840, 846, 5356]),
    ("Bitter-W$_{NF4}$A$_{FP16}$", [852, 853, 5364]),
    ("Bitter-W$_{FP8}$A$_{FP8}$", [1248, 1254, 5764]),
    ("Bitter-W$_{MXFP8}$A$_{MXFP8}$", [1248, 1254, 5764]),
    ("Bitter-W$_{INT1}$A$_{INT8}$", [534, 540, 5050]),
]
bloom_times_data = [
    ("PyTorch-Inductor", [12088, 12072, 15674]),
    ("ONNXRuntime", [7356, 6844, 64718]),
    ("TensorRT", [5771, 5783, 21292]),
    ("vLLM", [30512, 30516, 30512]),
    ("vLLM-W$_{INT4}$A$_{FP16}$", [22608, 22612, 22608]),
    ("Welder", [5148, 5160, 20046]),
    ("Bitter", [5136, 5156, 20592]),
    ("Bitter-W$_{INT4}$A$_{FP16}$", [3372, 3392, 18828]),
    ("Bitter-W$_{NF4}$A$_{FP16}$", [3382, 3384, 18844]),
    ("Bitter-W$_{FP8}$A$_{FP8}$", [3960, 3980, 19416]),
    ("Bitter-W$_{MXFP8}$A$_{MXFP8}$", [3960, 3980, 19416]),
    ("Bitter-W$_{INT1}$A$_{INT8}$", [2931, 2951, 18387]),
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
        log_path = f"./logs/{model}_pytorch_{batch_size}_{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            print(llama_times_data[0][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            print(bloom_times_data[0][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])

# update the onnxruntime results
for model in ["llama", "bloom"]:
    for batch_size, seq_len in [
            (1, 1),
            (32, 1),
            (1, 4096)
        ]:
        log_path = f"./logs/{model}_onnxruntime_{batch_size}_{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            print(llama_times_data[1][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            print(bloom_times_data[1][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])

# update the tensorrt results
for model in ["llama", "bloom"]:
    for batch_size, seq_len in [
            (1, 1),
            (32, 1),
            (1, 4096)
        ]:
        log_path = f"./logs/{model}_tensorrt_{batch_size}_{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            print(llama_times_data[2][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            print(bloom_times_data[2][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])

# update the welder results
for model in ["llama", "bloom"]:
    for batch_size, seq_len in [
            (1, 1),
            (32, 1),
            (1, 4096)
        ]:
        log_path = f"./logs/{model}_welder_{batch_size}_{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            print(llama_times_data[5][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            print(bloom_times_data[5][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])
            
# update the vllm results
for model in ["llama", "bloom"]:
    for batch_size, seq_len in [
            (1, 1),
            (32, 1),
            (1, 4096)
        ]:
        log_path = f"./logs/{model}_vllm_{batch_size}_{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            print(llama_times_data[3][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            print(bloom_times_data[3][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])

# update the welder results
for model in ["llama", "bloom"]:
    for batch_size, seq_len in [
            (1, 1),
            (32, 1),
            (1, 4096)
        ]:
        log_path = f"./logs/{model}_ladder_{batch_size}_{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            print(llama_times_data[4][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            print(bloom_times_data[4][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])

# update the ladder_fp16_int4 results
for model in ["llama", "bloom"]:
    for batch_size, seq_len in [
            (1, 1),
            (32, 1),
            (1, 4096)
        ]:
        log_path = f"./logs/{model}_ladder_fp16_int4_{batch_size}_{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            print(llama_times_data[6][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            print(bloom_times_data[6][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])

# update the ladder_fp16_nf4 results
for model in ["llama", "bloom"]:
    for batch_size, seq_len in [
            (1, 1),
            (32, 1),
            (1, 4096)
        ]:
        log_path = f"./logs/{model}_ladder_fp16_nf4_{batch_size}_{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            print(llama_times_data[7][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            print(bloom_times_data[7][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])

# update the ladder_fp8_fp8 results
for model in ["llama", "bloom"]:
    for batch_size, seq_len in [
            (1, 1),
            (32, 1),
            (1, 4096)
        ]:
        log_path = f"./logs/{model}_ladder_fp8_fp8_{batch_size}_{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            print(llama_times_data[8][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            print(bloom_times_data[8][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])

# update the ladder_fp16_mxfp8xmxfp8 results
for model in ["llama", "bloom"]:
    for batch_size, seq_len in [
            (1, 1),
            (32, 1),
            (1, 4096)
        ]:
        log_path = f"./logs/{model}_ladder_fp16_mxfp8xmxfp8_{batch_size}_{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            print(llama_times_data[9][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            print(bloom_times_data[9][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])
            
# update the ladder_fp16_int8xint1 results
for model in ["llama", "bloom"]:
    for batch_size, seq_len in [
            (1, 1),
            (32, 1),
            (1, 4096)
        ]:
        log_path = f"./logs/{model}_ladder_fp16_int8xint1_{batch_size}_{seq_len}_data.json"
        if not os.path.exists(log_path):
            continue
        data = list(json.load(open(log_path)).values())[-1]
        if model == "llama":
            print(llama_times_data[10][1][llama_providers.index(f"BS{batch_size} SEQ{seq_len}")])
        elif model == "bloom":
            print(bloom_times_data[10][1][bloom_providers.index(f"BS{batch_size} SEQ{seq_len}")])


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
