# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import argparse
import os

CHECKPOINT_PATH = os.path.join(os.getcwd(), "../../checkpoints/Figure13")
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
    os.system(f"python3 plot_optimization_breakdown.py")
else:
    print("Reproducing the results")
    # reproduce the results for welder-roller
    run_command("./benchmark_welder_roller.sh", working_dir="welder-roller")
    # reproduce the results for transform
    run_command("./benchmark_transform.sh", working_dir="transform")
    # reproduce the results for ptx
    run_command("./benchmark_ptx.sh", working_dir="ptx")
    # reproduce the results for holistic
    run_command("./benchmark_holistic.sh", working_dir="holistic")
    
    # plot from the reproduced results
    os.system(f"python3 plot_optimization_breakdown.py --reproduce")
