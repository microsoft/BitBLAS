import argparse
import os

CHECKPOINT_PATH = os.path.join(os.getcwd(), "../../checkpoints/Figure10")

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
    os.system(f"python3 plot_figures_a6000.py")
else:
    print("Reproducing the results")
    # reproduce the results for amos
    os.system("cd amos-benchmark")
