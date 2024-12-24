# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

# Get the absolute path of the current Python script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path of the project root directory (one level above the current directory)
project_root_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Define the path to the VERSION file located in the project root directory
version_file_path = os.path.join(project_root_dir, "VERSION")

# Read and store the version information from the VERSION file
# Use 'strip()' to remove any leading/trailing whitespace or newline characters
with open(version_file_path, "r") as version_file:
    __version__ = version_file.read().strip()

# Define the public API for the module
__all__ = ["__version__"]
