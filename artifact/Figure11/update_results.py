import os
import json
import re
from reproduce_result import (
        llama_providers,
        llama_times_data,
        bloom_providers,
        bloom_times_data
    )
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
