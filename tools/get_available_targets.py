# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas.utils import get_all_nvidia_targets
from tabulate import tabulate

def main():
    # Get all available Nvidia targets
    targets = get_all_nvidia_targets()
    
    # Print available targets to console in a table format
    table = [[i + 1, target] for i, target in enumerate(targets)]
    headers = ["Index", "Target"]
    print(tabulate(table, headers, tablefmt="pretty"))

if __name__ == "__main__":
    main()