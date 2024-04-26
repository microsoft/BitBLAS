# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import argparse
import os
import subprocess

def run_command(command, working_dir=None):
    """Run command in the shell and handle errors."""
    try:
        subprocess.run(command, shell=True, check=True, cwd=working_dir)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

os.environ['CHECKPOINT_PATH'] = os.path.join(os.getcwd(), "../checkpoints/Table2")

parser = argparse.ArgumentParser()

parser.add_argument("--reproduce", action="store_true", help="reproduce, otherwise use the paper results", default=False)
parser.add_argument("--force_tune_amos", action="store_true", help="force_tune, otherwise use the checkpoints if available", default=False)
parser.add_argument("--force_tune_tensorir", action="store_true", help="force_tune, otherwise use the checkpoints if available", default=False)
parser.add_argument("--force_tune_welder", action="store_true", 
help="force_tune, otherwise use the checkpoints if available", default=False)
parser.add_argument("--force_tune_ladder", action="store_true", help="force_tune, otherwise use the checkpoints if available", default=False)

parser.add_argument(
    "--force_tune",
    action="store_true",
    help="force_tune, otherwise use the checkpoints if available",
    default=False,
)

args = parser.parse_args()
reproduce = args.reproduce
force_tune = args.force_tune
force_tune_amos = args.force_tune_amos
force_tune_tensorir = args.force_tune_tensorir
force_tune_welder = args.force_tune_welder
force_tune_ladder = args.force_tune_ladder


if not reproduce:
    print("Using the paper results")
    os.system(f"python3 plot_nvidia_table2.py")
else:
    print("Reproducing the results")
    # reproduce the results for amos
    os.system("cd amos-benchmark")
    # amos-benchmark
    if force_tune_amos:
        run_command("./benchmark_amos.sh --force_tune", working_dir="amos-benchmark")
    else:
        run_command("./benchmark_amos.sh", working_dir="amos-benchmark")
    # tensorir-benchmark    
    if force_tune_tensorir:
        run_command("./benchmark_tensorir.sh --force_tune", working_dir="tensorir-benchmark")
    else:
        run_command("./benchmark_tensorir.sh", working_dir="tensorir-benchmark")
    # welder-benchmark
    if force_tune_welder:
        run_command("./benchmark_welder.sh --force_tune", working_dir="welder-benchmark")
    else:
        run_command("./benchmark_welder.sh", working_dir="welder-benchmark")
    # ladder-benchmark
    if force_tune_ladder:
        run_command("./benchmark_ladder.sh --force_tune", working_dir="ladder-benchmark")
    else:
        run_command("./benchmark_ladder.sh", working_dir="ladder-benchmark")

    os.system(f"python3 update_results.py")
    # Finally, run the plotting script with the appropriate flags
    os.system(f"python3 plot_nvidia_table2.py --reproduce")
