# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

# Get the absolute path of the current Python script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the VERSION file located in the package directory
version_file_path = os.path.join(current_dir, "VERSION")

# If the VERSION file is not found, locate it in the project root directory instead
if not os.path.exists(version_file_path):
    version_file_path = os.path.join(current_dir, "..", "VERSION")

# Read and store the version information from the VERSION file
# Use 'strip()' to remove any leading/trailing whitespace or newline characters
with open(version_file_path, "r") as version_file:
    __version__ = version_file.read().strip()

# Define the public API for the module
__all__ = ["__version__"]
