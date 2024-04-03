# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import subprocess
import shutil
from setuptools import setup, find_packages
from setuptools.command.install import install

import requests
import tarfile
from io import BytesIO
import os

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
    
    base_url = f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{version}"
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
    return os.path.abspath(
        os.path.join(extract_path, file_name.replace(".tar.xz", ""))
    )

LLVM_VERSION = "10.0.1"
IS_AARCH64 = False  # Set to True if on an aarch64 platform
EXTRACT_PATH = "3rdparty"  # Default extraction path

class BitBLASInstallCommand(install):
    """Customized setuptools install command - builds TVM after setting up LLVM."""

    def run(self):
        # install_requirements
        self.install_requirements()
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
            subprocess.check_call(['git', 'submodule', 'update', '--init', '--recursive'])
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
            subprocess.check_call(['cmake', '..'])
            subprocess.check_call(['make', '-j'])
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Failed to build TVM: {error}")
        finally:
            # Go back to the original directory
            os.chdir("../../..")

    def install_requirements(self):
        """Installs dependencies listed in requirements.txt."""
        try:
            subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Failed to install requirements: {error}")


setup(
    name="bitblas",
    version="0.0.0.dev",
    packages=find_packages(),
    install_requires=[],
    author="Microsoft Research",
    author_email="leiwang1999@outlook.com",
    description="A light weight framework to generate high performance CUDA/HIP code for BLAS operators.",
    license="MIT",
    keywords="BLAS, CUDA, HIP, Code Generation, TVM",
    url="https://github.com/microsoft/BitBLAS",
    cmdclass={
        "install": BitBLASInstallCommand,
    },
)
