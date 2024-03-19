# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from setuptools import setup, find_packages
from setuptools.command.install import install


class ApacheTVMInstallCommand(install):
    """Customized setuptools install command - builds the submodule first."""

    def run(self):
        os.system("./maint/scripts/installation.sh")
        install.run(self)


setup(
    name="bitblas",
    version="0.1",
    packages=find_packages(),
    install_requires=[],
    author="Microsoft Research",
    author_email="leiwang1999@outlook.com",
    description=
    "A light weight framework to generate high performance CUDA/HIP code for BLAS operators.",
    license="MIT",
    keywords="BLAS, CUDA, HIP, Code Generation, TVM",
    url="https://github.com/microsoft/BitBLAS",
    cmdclass={
        "install": ApacheTVMInstallCommand,
    },
)
