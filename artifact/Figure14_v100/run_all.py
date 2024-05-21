# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import argparse
import os
import subprocess

CHECKPOINT_PATH = os.path.join(os.getcwd(), "../../checkpoints/Figure14")
os.environ["CHECKPOINT_PATH"] = CHECKPOINT_PATH

parser = argparse.ArgumentParser()

def run_command(command, working_dir=None):
    """Run command in the shell and handle errors."""
    try:
        result = subprocess.run(command, shell=True, executable='/bin/bash', text=True, cwd=working_dir)
    except subprocess.CalledProcessError as e:
        return

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
    os.system(f"python3 plot_scaling_bitwidth.py")
else:
    print("Reproducing the results")
    # reproduce the results for kernel
    os.system("cd kernel-benchmark; ./benchmark_kernel.sh")
    # reproduce the results for end2end
    os.system("cd ladder-benchmark; ./benchmark_ladder.sh")
    # update the figures
    os.system("python3 update_results_kernel.py")
    os.system("python3 update_results_end2end.py")
    # plot from the reproduced results
    os.system(f"python3 plot_scaling_bitwidth.py --reproduce")
