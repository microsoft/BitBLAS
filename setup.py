# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import io
import subprocess
import shutil
from packaging.version import Version, parse
from setuptools import setup, find_packages
from setuptools.command.install import install
from typing import List
import re
import requests
import tarfile
from io import BytesIO
import os
import sys

ROOT_DIR = os.path.dirname(__file__)
MAIN_CUDA_VERSION = "12.1"

# BitBLAS only supports Linux platform
assert sys.platform.startswith(
    "linux"
), "BitBLAS only supports Linux platform (including WSL)."


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements


def find_version(filepath: str) -> str:
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


def get_nvcc_cuda_version() -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output(
        ["nvcc", "-V"], universal_newlines=True
    )
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_bitblas_version() -> str:
    version = find_version(get_path("python/bitblas", "__init__.py"))

    cuda_version = str(get_nvcc_cuda_version())
    if cuda_version != MAIN_CUDA_VERSION:
        cuda_version_str = cuda_version.replace(".", "")[:3]
        version += f"+cu{cuda_version_str}"
    return version


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""


def download_and_extract_llvm(version, is_aarch64=False, extract_path="3rdparty"):
    """
    Downloads and extracts the specified version of LLVM for the given platform.
    Args:
        version (str): The version of LLVM to download.
        is_aarch64 (bool): True if the target platform is aarch64, False otherwise.
        extract_path (str): The directory path where the archive will be extracted.

    Returns:
        str: The path where the LLVM archive was extracted.
    """
    ubuntu_version = "16.04"
    if version >= "16.0.0":
        ubuntu_version = "20.04"
    elif version >= "13.0.0":
        ubuntu_version = "18.04"

    base_url = (
        f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{version}"
    )
    file_name = f"clang+llvm-{version}-{'aarch64-linux-gnu' if is_aarch64 else f'x86_64-linux-gnu-ubuntu-{ubuntu_version}'}.tar.xz"

    download_url = f"{base_url}/{file_name}"

    # Download the file
    print(f"Downloading {file_name} from {download_url}")
    response = requests.get(download_url)
    # Check if the request was successful
    response.raise_for_status()

    # Ensure the extract path exists
    os.makedirs(extract_path, exist_ok=True)

    # if the file already exists, remove it
    if os.path.exists(os.path.join(extract_path, file_name)):
        os.remove(os.path.join(extract_path, file_name))

    # Extract the file
    print(f"Extracting {file_name} to {extract_path}")
    with tarfile.open(fileobj=BytesIO(response.content), mode="r:xz") as tar:
        tar.extractall(path=extract_path, members=None)

    print("Download and extraction completed successfully.")
    return os.path.abspath(os.path.join(extract_path, file_name.replace(".tar.xz", "")))


LLVM_VERSION = "10.0.1"
IS_AARCH64 = False  # Set to True if on an aarch64 platform
EXTRACT_PATH = "3rdparty"  # Default extraction path


class BitBLASInstallCommand(install):
    """Customized setuptools install command - builds TVM after setting up LLVM."""

    def run(self):
        # Recursively update submodules
        self.update_submodules()
        # Set up LLVM for TVM
        llvm_path = self.setup_llvm_for_tvm()
        # Build TVM
        self.build_tvm(llvm_path)
        # Continue with the standard installation process
        install.run(self)

    def update_submodules(self):
        """Updates git submodules."""
        try:
            subprocess.check_call(
                ["git", "submodule", "update", "--init", "--recursive"]
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Failed to update submodules: {error}")

    def setup_llvm_for_tvm(self):
        """Downloads and extracts LLVM, then configures TVM to use it."""
        # Assume the download_and_extract_llvm function and its dependencies are defined elsewhere in this script
        extract_path = download_and_extract_llvm(LLVM_VERSION, IS_AARCH64, EXTRACT_PATH)
        llvm_config_path = os.path.join(extract_path, "bin", "llvm-config")
        return llvm_config_path

    def build_tvm(self, llvm_config_path):
        """Configures and builds TVM."""
        os.chdir("3rdparty/tvm")
        if not os.path.exists("build"):
            os.makedirs("build")
        os.chdir("build")
        # Copy the config.cmake as a baseline
        if not os.path.exists("config.cmake"):
            shutil.copy("../cmake/config.cmake", "config.cmake")
        # Set LLVM path and enable CUDA in config.cmake
        with open("config.cmake", "a") as config_file:
            config_file.write(f"set(USE_LLVM {llvm_config_path})\n")
            config_file.write("set(USE_CUDA ON)\n")
        # Run CMake and make
        try:
            subprocess.check_call(["cmake", ".."])
            subprocess.check_call(["make", "-j"])
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Failed to build TVM: {error}")
        finally:
            # Go back to the original directory
            os.chdir("../../..")

setup(
    name="bitblas",
    version=get_bitblas_version(),
    packages=find_packages(
        where="python",
    ),
    package_dir={'': 'python'},
    author="Microsoft Research",
    description="A light weight framework to generate high performance CUDA/HIP code for BLAS operators.",
    long_description=read_readme(),
    license="MIT",
    keywords="BLAS, CUDA, HIP, Code Generation, TVM",
    url="https://github.com/microsoft/BitBLAS",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics, Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    cmdclass={
        "install": BitBLASInstallCommand,
    },
)
