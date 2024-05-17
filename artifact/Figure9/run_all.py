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

CHECKPOINT_PATH = os.path.join(os.getcwd(), "../checkpoints/Figure9")

parser = argparse.ArgumentParser()

parser.add_argument("--reproduce", action="store_true", help="reproduce, otherwise use the paper results", default=False)
parser.add_argument("--force_tune_amos", action="store_true", help="force_tune_amos, otherwise use the checkpoints if available", default=False)
parser.add_argument("--force_tune_tensorir", action="store_true", help="force_tune_tensorir, otherwise use the checkpoints if available", default=False)
parser.add_argument("--force_tune_welder", action="store_true", help="force_tune_welder, otherwise use the checkpoints if available", default=False)
parser.add_argument("--force_tune_ladder", action="store_true", help="force_tune_ladder, otherwise use the checkpoints if available", default=False)

args = parser.parse_args()
reproduce = args.reproduce
force_tune_amos = args.force_tune_amos
force_tune_tensorir = args.force_tune_tensorir
force_tune_welder = args.force_tune_welder
force_tune_ladder = args.force_tune_ladder

if not reproduce:
    print("Using the paper results")
    os.system(f"python3 plot_figures.py")
else:
    print("Reproducing the results")
    # reproduce the results for amos
    if force_tune_amos:
        run_command(f"cd amos-benchmark;CHECKPOINT_PATH={CHECKPOINT_PATH} ./benchmark_amos.sh --force_tune; cd ..")
    else:
        run_command(f"cd amos-benchmark;CHECKPOINT_PATH={CHECKPOINT_PATH} ./benchmark_amos.sh; cd ..")
    # reproduce the results for ladder
    if force_tune_ladder:
        run_command(f"cd ladder-benchmark;CHECKPOINT_PATH={CHECKPOINT_PATH} ./benchmark_ladder.sh --force_tune; cd ..")
    else:
        run_command(f"cd ladder-benchmark;CHECKPOINT_PATH={CHECKPOINT_PATH} ./benchmark_ladder.sh; cd ..")
    # reproduce the results for onnxruntime
    run_command(f"cd onnxruntime-benchmark;CHECKPOINT_PATH={CHECKPOINT_PATH} ./benchmark_onnxruntime.sh; cd ..")
    # reproduce the results for pytorch inductor
    run_command(f"cd pytorch-inductor-benchmark;CHECKPOINT_PATH={CHECKPOINT_PATH} ./benchmark_inductor.sh; cd ..")
    # reproduce the results for tensorir
    if force_tune_tensorir:
        run_command(f"cd tensorir-benchmark;CHECKPOINT_PATH={CHECKPOINT_PATH} ./benchmark_tensorir.sh --force_tune; cd ..")
    else:
        run_command(f"cd tensorir-benchmark;CHECKPOINT_PATH={CHECKPOINT_PATH} ./benchmark_tensorir.sh; cd ..")
    # reproduce the results for tensorrt
    run_command(f"cd tensorrt-benchmark;CHECKPOINT_PATH={CHECKPOINT_PATH} ./benchmark_tensorrt.sh; cd ..")
    # reproduce the results for vllm
    run_command(f"cd vllm-benchmark;CHECKPOINT_PATH={CHECKPOINT_PATH} ./benchmark_vllm.sh; cd ..")
    # reproduce the results for welder
    if force_tune_welder:
        run_command(f"cd welder-benchmark;CHECKPOINT_PATH={CHECKPOINT_PATH} ./benchmark_welder.sh --force_tune; cd ..")
    else:
        run_command(f"cd welder-benchmark;CHECKPOINT_PATH={CHECKPOINT_PATH} ./benchmark_welder.sh; cd ..")
    # update the reproduce results from logs
    run_command(f"python3 update_results.py")
    # plot from the reproduced results
    run_command(f"python3 plot_figures.py --reproduce")
