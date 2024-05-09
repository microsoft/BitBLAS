# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import argparse
import os
import subprocess

def run_command(command):
    result = subprocess.run(command, shell=True, executable='/bin/bash', text=True)
    
    if result.returncode == 0:
        print("Command executed successfully")
    else:
        print(f"Command failed with error: {result.stderr}")

CHECKPOINT_PATH = os.path.join(os.getcwd(), "../../checkpoints/Figure11")
os.environ["CHECKPOINT_PATH"] = CHECKPOINT_PATH

parser = argparse.ArgumentParser()

parser.add_argument("--reproduce", action="store_true", help="reproduce, otherwise use the paper results", default=False)
parser.add_argument(
    "--force_tune",
    action="store_true",
    help="force_tune, otherwise use the checkpoints if available",
    default=False,
)

args = parser.parse_args()
reproduce = args.reproduce
force_tune = args.force_tune

if not reproduce:
    print("Using the paper results")
    os.system(f"python3 plot_memory_usage.py")
else:
    print("Reproducing the results")
    # initialize the checkpoints
    # initialize tensorrt engine
    run_command(f"cd tensorrt-benchmark; ./initialize_tensorrt.sh")
    # initialize welder
    run_command(f"cd welder-benchmark; ./initialize_welder.sh")
    # initialize ladder
    run_command(f"cd ladder-benchmark; ./initialize_ladder.sh")
    # initialize vllm
    for model in ["llama", "bloom"]:
        for batch_size, seq_len in [
                (1, 1),
                (1, 4096)
            ]:
            # reproduce the results for inductor
            os.system(f"python -u measure_memory.py --framework pytorch --model {model} --batch_size {batch_size} --seq_len {seq_len} 2>&1 | tee logs/{model}_pytorch_{batch_size}_{seq_len}.log")
            # reproduce the results for onnxruntime
            os.system(f"python -u measure_memory.py --framework onnxruntime --model {model} --batch_size {batch_size} --seq_len {seq_len} 2>&1 | tee logs/{model}_onnxruntime_{batch_size}_{seq_len}.log")
            # reproduce the results for tensorrt
            os.system(f"python -u measure_memory.py --framework tensorrt --model {model} --batch_size {batch_size} --seq_len {seq_len} 2>&1 | tee logs/{model}_tensorrt_{batch_size}_{seq_len}.log")
            # reproduce the results for welder
            os.system(f"python -u measure_memory.py --framework welder --model {model} --batch_size {batch_size} --seq_len {seq_len} 2>&1 | tee logs/{model}_welder_{batch_size}_{seq_len}.log")
            # reproduce the results for ladder
            os.system(f"python -u measure_memory.py --framework ladder --model {model} --batch_size {batch_size} --seq_len {seq_len} 2>&1 | tee logs/{model}_ladder_{batch_size}_{seq_len}.log")
            # reproduce the results for ladder_fp16_int4
            os.system(f"python -u measure_memory.py --framework ladder_fp16_int4 --model {model} --batch_size {batch_size} --seq_len {seq_len} 2>&1 | tee logs/{model}_ladder_fp16_int4_{batch_size}_{seq_len}.log")
            # reproduce the results for ladder_fp16_nf4
            os.system(f"python -u measure_memory.py --framework ladder_fp16_nf4 --model {model} --batch_size {batch_size} --seq_len {seq_len} 2>&1 | tee logs/{model}_ladder_fp16_nf4_{batch_size}_{seq_len}.log")
            # reproduce the results for ladder_fp8_fp8
            os.system(f"python -u measure_memory.py --framework ladder_fp8_fp8 --model {model} --batch_size {batch_size} --seq_len {seq_len} 2>&1 | tee logs/{model}_ladder_fp8_fp8_{batch_size}_{seq_len}.log")
            # reproduce the results for ladder_fp16_mxfp8xmxfp8
            os.system(f"python -u measure_memory.py --framework ladder_mxfp8_mxfp8 --model {model} --batch_size {batch_size} --seq_len {seq_len} 2>&1 | tee logs/{model}_ladder_fp16_mxfp8xmxfp8_{batch_size}_{seq_len}.log")
            # reproduce the results for ladder_fp16_int8xint1
            os.system(f"python -u measure_memory.py --framework ladder_int8_int1 --model {model} --batch_size {batch_size} --seq_len {seq_len} 2>&1 | tee logs/{model}_ladder_fp16_int8xint1_{batch_size}_{seq_len}.log")
    
    os.system(f"python3 plot_memory_usage.py --reproduce")
