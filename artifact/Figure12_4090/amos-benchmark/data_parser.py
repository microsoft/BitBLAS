from matmul_data_config import *
import time

import re  
  
def parse_log(file_path):  
    with open(file_path, "r") as log_file:  
        log_data = log_file.read()  
  
    # Extract cost of gemm-float16-float16-layer  
    cost_pattern = r"Cost of gemm-float16-float16-layer-\((\d+),\s(\d+),\s(\d+)\) is (\d+\.\d+) ms"  
    cost_matches = re.finditer(cost_pattern, log_data)  
    
    for cost_match in cost_matches:  
        M, N, K, cost = int(cost_match.group(1)), int(cost_match.group(2)), int(cost_match.group(3)), float(cost_match.group(4))  
        print(f"M: {M}, N: {N}, K: {K}, Cost: {cost} ms")  
  
    # Extract all time costs  
    time_costs = []  
    time_cost_pattern = r"Time cost: (\d+\.\d+)"  
    time_cost_matches = re.finditer(time_cost_pattern, log_data)  
  
    for time_cost_match in time_cost_matches:  
        time_cost = float(time_cost_match.group(1))  
        time_costs.append(time_cost)  
  
    print("Time costs:")  
    for idx, time_cost in enumerate(time_costs, start=1):  
        print(f"{idx}. {time_cost}")  
  
# Usage  
file_path = "/workspace/v-leiwang3/lowbit-benchmark/1.matmul_bench/amosgemm/amos_tunning_simple.log"  
parse_log(file_path)  
