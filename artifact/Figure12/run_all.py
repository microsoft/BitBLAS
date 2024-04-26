# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import argparse
import os
import subprocess

CHECKPOINT_PATH = os.path.join(os.getcwd(), "../../checkpoints/Figure12")

def run_command(command, working_dir=None):
    """Run command in the shell and handle errors."""
    try:
        subprocess.run(command, shell=True, check=True, cwd=working_dir)
    except subprocess.CalledProcessError as e:
        return

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
    os.system(f"python3 plot_operator_performance.py")
else:
    print("Reproducing the results")
    # reproduce the results for amos
    run_command("./benchmark_amos.sh", working_dir="amos-benchmark")
    # reproduce the results for tensorir
    run_command("./benchmark_tensorir.sh", working_dir="tensorir-benchmark")
    # reproduce the results for cublas
    run_command("./benchmark_cublas.sh", working_dir="cublas-benchmark")
    # reproduce the results for cudnn
    run_command("./benchmark_cudnn.sh", working_dir="cudnn-benchmark")
    # reproduce the results for vllm
    run_command("./benchmark_vllm.sh", working_dir="vllm-benchmark")
    # reproduce the results for ladder
    run_command("./benchmark_ladder.sh", working_dir="ladder-benchmark")
    # plot from the reproduced results
    os.system(f"python3 plot_operator_performance.py --reproduce")
