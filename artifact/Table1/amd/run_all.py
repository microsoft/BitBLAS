# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import argparse
import subprocess

def run_command(command):
    """Run command in the shell and handle errors."""
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def change_directory(path):
    """Change working directory."""
    os.chdir(path)
    print(f"Changed directory to {os.getcwd()}")

# Setting the environment variable for CHECKPOINT_PATH
os.environ['CHECKPOINT_PATH'] = os.path.join(os.getcwd(), "../../checkpoints/TABLE1")

# Handling arguments
parser = argparse.ArgumentParser(description="Run all benchmarks for NVIDIA GPUs.")

parser.add_argument("--device", type=str, default="MI250", help="Specify the GPU device (default: A100)")
parser.add_argument("--reproduce", action="store_true", help="Reproduce results, otherwise use the paper results")
parser.add_argument("--force_tune", action="store_true", help="Force tuning, otherwise use available checkpoints if not reproducing")

args = parser.parse_args()

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
    change_directory("cublas-benchmark")
    run_command("./compile_and_run.sh")
    change_directory("..")

    # amos-benchmark
    change_directory("amos-benchmark")
    if force_tune:
        run_command(f"./benchmark_amos.sh --force_tune")
    else:
        run_command(f"./benchmark_amos.sh")
    change_directory("..")

    # TensorIR-benchmark
    change_directory("tensorir-benchmark")
    if force_tune:
        run_command(f"./benchmark_tensorir.sh --force_tune")
    else:
        run_command(f"./benchmark_tensorir.sh")
    change_directory("..")

    # roller-benchmark
    change_directory("roller-benchmark")
    run_command("./benchmark_roller.sh")
    change_directory("..")

    # Finally, run the plotting script with the appropriate flags
    run_command(f"python3 plot_nvidia_table1.py --device {device} --reproduce")
