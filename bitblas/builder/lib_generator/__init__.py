# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Optional
from bitblas.base.arch import TileDevice
import ctypes
import os
import os.path as osp
import sys
import tempfile
import subprocess
import logging

logger = logging.getLogger(__name__)


class LibraryGenerator(object):
    srcpath: Optional[str] = None
    libpath: Optional[str] = None
    lib_code: Optional[str] = None

    def __init__(self, arch: TileDevice):
        self.arch = arch

    def update_lib_code(self, lib_code: str):
        self.lib_code = lib_code

    # Assume currently we only support CUDA compilation
    def load_lib(self):
        return ctypes.CDLL(self.libpath)

    def compile_lib(self, timeout: float = None, with_tl: bool = False):
        arch = self.arch
        platform = arch.platform
        if platform == "CUDA":
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False)
            compute_version = arch.compute_capability
            libpath = src.name.replace(".cu", ".so")

            command = [
                "nvcc",
                "-std=c++17",
                "-Xcudafe",
                "--diag_suppress=177",
                "--compiler-options",
                "'-fPIC'",
                "-lineinfo",
                "--shared",
                src.name,
                "-lcuda",
                "-gencode",
                f"arch=compute_{compute_version},code=sm_{compute_version}",
            ]

        elif platform == "CDNA":
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False)
            libpath = src.name.replace(".cpp", ".so")

            command = [
                "hipcc",
                "-std=c++17",
                "-fPIC",
                "--shared",
                src.name,
            ]

        else:
            raise ValueError(f"Unsupported platform: {platform}")

        if with_tl:
            install_tilelang_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../..", "3rdparty", "tilelang")
            develop_tilelang_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../../..", "3rdparty", "tilelang")

            tilelang_root = next((path for path in [install_tilelang_path, develop_tilelang_path]
                                  if os.path.exists(path) and path not in sys.path), None)

            if "TL_TEMPLATE_PATH " in os.environ:
                tl_template_path = os.environ["TL_TEMPLATE_PATH"]
            else:
                tl_template_path = osp.abspath(osp.join(tilelang_root, "src"))

            tl_template_path = osp.abspath(osp.join(tilelang_root, "src"))
            if "TL_CUTLASS_PATH" in os.environ:
                cutlass_path = os.environ["TL_CUTLASS_PATH"]
            else:
                cutlass_path = osp.abspath(osp.join(tilelang_root, "3rdparty/cutlass/include"))

            command += [
                "-I" + tl_template_path,
                "-I" + cutlass_path,
            ]
            command += ["-diag-suppress=20013"]
        command += ["-o", libpath]

        src.write(self.lib_code)
        src.flush()
        try:
            ret = subprocess.run(command, timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning(f"Compilation Timeout! {command}")
            return None
        if ret.returncode != 0:
            logger.warning(f"Compilation Failed! {command}")
            return None
        self.srcpath = src.name
        self.libpath = libpath

    def remove_lib(self):
        if self.libpath:
            os.remove(self.libpath)
        self.libpath = None

    def get_source_path(self):
        return self.srcpath

    def get_lib_path(self):
        return self.libpath

    def set_lib_path(self, libpath):
        self.libpath = libpath

    def set_src_path(self, srcpath):
        self.srcpath = srcpath
