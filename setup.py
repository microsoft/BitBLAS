# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import io
import subprocess
import shutil
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist
import distutils.dir_util
from typing import List
import re
import tarfile
from io import BytesIO
import os
import sys
import urllib.request
from distutils.version import LooseVersion

PACKAGE_NAME = "bitblas"
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


def get_nvcc_cuda_version():
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output(
        ["nvcc", "-V"], universal_newlines=True
    )
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = LooseVersion(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_bitblas_version(with_cuda=True) -> str:
    version = find_version(get_path("python/bitblas", "__init__.py"))
    if not with_cuda:
        return version
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
    with urllib.request.urlopen(download_url) as response:
        if response.status != 200:
            raise Exception(f"Download failed with status code {response.status}")
        file_content = response.read()
    # Ensure the extract path exists
    os.makedirs(extract_path, exist_ok=True)

    # if the file already exists, remove it
    if os.path.exists(os.path.join(extract_path, file_name)):
        os.remove(os.path.join(extract_path, file_name))

    # Extract the file
    print(f"Extracting {file_name} to {extract_path}")
    with tarfile.open(fileobj=BytesIO(file_content), mode="r:xz") as tar:
        tar.extractall(path=extract_path)

    print("Download and extraction completed successfully.")
    return os.path.abspath(os.path.join(extract_path, file_name.replace(".tar.xz", "")))

package_data = {
    "bitblas": ["py.typed"],
}

LLVM_VERSION = "10.0.1"
IS_AARCH64 = False  # Set to True if on an aarch64 platform
EXTRACT_PATH = "3rdparty"  # Default extraction path


def update_submodules():
    """Updates git submodules."""
    try:
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "--recursive"]
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Failed to update submodules: {error}")

def build_tvm(llvm_config_path):
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

def setup_llvm_for_tvm():
    """Downloads and extracts LLVM, then configures TVM to use it."""
    # Assume the download_and_extract_llvm function and its dependencies are defined elsewhere in this script
    extract_path = download_and_extract_llvm(LLVM_VERSION, IS_AARCH64, EXTRACT_PATH)
    llvm_config_path = os.path.join(extract_path, "bin", "llvm-config")
    return extract_path, llvm_config_path

class BitBLASInstallCommand(install):
    """Customized setuptools install command - builds TVM after setting up LLVM."""

    def run(self):
        # Recursively update submodules
        # update_submodules()
        # Set up LLVM for TVM
        _, llvm_path = setup_llvm_for_tvm()
        # Build TVM
        build_tvm(llvm_path)
        # Continue with the standard installation process
        install.run(self)

class BitBLASBuilPydCommand(build_py):
    """Customized setuptools install command - builds TVM after setting up LLVM."""

    def run(self):
        build_py.run(self)
        # custom build tvm
        # update_submodules()
        # Set up LLVM for TVM
        extract_path, llvm_path = setup_llvm_for_tvm()
        # Build TVM
        build_tvm(llvm_path)
        # Copy the built TVM to the package directory 
        TVM_PREBUILD_ITEMS = [
            "3rdparty/tvm/build/libtvm_runtime.so",
            "3rdparty/tvm/build/libtvm.so",
            "3rdparty/tvm/build/config.cmake",
            "3rdparty/tvm/python",
            "3rdparty/tvm/licenses",
            "3rdparty/tvm/conftest.py",
            "3rdparty/tvm/CONTRIBUTORS.md",
            "3rdparty/tvm/KEYS",
            "3rdparty/tvm/LICENSE",
            "3rdparty/tvm/README.md",
            "3rdparty/tvm/mypy.ini",
            "3rdparty/tvm/pyproject.toml",
            "3rdparty/tvm/version.py",
        ]
        LLVM_PREBUILD_ITEMS = [
            extract_path,
        ]
        for item in TVM_PREBUILD_ITEMS + LLVM_PREBUILD_ITEMS:
            source_dir = os.path.join(ROOT_DIR, item)
            target_dir = os.path.join(self.build_lib, PACKAGE_NAME, item)
            if os.path.isdir(source_dir):
                self.mkpath(target_dir)
                distutils.dir_util.copy_tree(source_dir, target_dir)
            else:
                target_dir = os.path.dirname(target_dir)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                shutil.copy2(source_dir, target_dir)

class BitBLASSdistCommand(sdist):
    """Customized setuptools sdist command - includes the pyproject.toml file."""
    def make_distribution(self):
        self.distribution.metadata.name = PACKAGE_NAME
        self.distribution.metadata.version = get_bitblas_version(with_cuda=False)
        super().make_distribution()

setup(
    name=PACKAGE_NAME,
    version=get_bitblas_version(),
    packages=find_packages(
        where="python"
    ),
    package_dir={"": "python"},
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
    tests_require=[
        "yapf>=0.32.0",
        "toml>=0.10.2",
        "tomli>=2.0.1",
        "ruff>=0.1.5",
        "codespell>=2.2.6"
    ],
    package_data=package_data,
    include_package_data=True,
    data_files=[
        "requirements.txt",
    ],
    cmdclass={
        "install": BitBLASInstallCommand,
        "build_py": BitBLASBuilPydCommand,
        "sdist": BitBLASSdistCommand,
    },
)
