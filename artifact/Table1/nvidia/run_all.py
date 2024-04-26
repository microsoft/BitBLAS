# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import argparse
import subprocess

def run_command(command, working_dir=None):
    """Run command in the shell and handle errors."""
    try:
        subprocess.run(command, shell=True, check=True, cwd=working_dir)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

# Handling arguments
parser = argparse.ArgumentParser(description="Run all benchmarks for NVIDIA GPUs.")

parser.add_argument("--device", type=str, default="A100", help="Specify the GPU device (default: A100)")
parser.add_argument("--reproduce", action="store_true", help="Reproduce results, otherwise use the paper results")
parser.add_argument("--force_tune", action="store_true", help="Force tuning, otherwise use available checkpoints if not reproducing")

args = parser.parse_args()


# Setting the environment variable for CHECKPOINT_PATH
os.environ['CHECKPOINT_PATH'] = os.path.join(os.getcwd(), "../../checkpoints/Table1", args.device.capitalize())

# Process arguments
device = args.device
reproduce = args.reproduce
force_tune = args.force_tune

if not reproduce:
    print("Using the paper results")
    run_command(f"python3 plot_nvidia_table1.py --device {device}")
else:
    print("Reproducing results for device: {}".format(device))
    # Execute benchmarks in specified order and directories
    # cublas-benchmark
    run_command("./compile_and_run.sh", working_dir="cublas-benchmark")
    # amos-benchmark
    if force_tune:
        run_command("./benchmark_amos.sh --force_tune", working_dir="amos-benchmark")
    else:
        run_command("./benchmark_amos.sh", working_dir="amos-benchmark")
    # tensorir-benchmark    
    if force_tune:
        run_command("./benchmark_tensorir.sh --force_tune", working_dir="tensorir-benchmark")
    else:
        run_command("./benchmark_tensorir.sh", working_dir="tensorir-benchmark")
    # roller-benchmark
    run_command("./benchmark_roller.sh", working_dir="roller-benchmark")

    # Finally, run the plotting script with the appropriate flags
    run_command(f"python3 plot_nvidia_table1.py --device {device} --reproduce")
