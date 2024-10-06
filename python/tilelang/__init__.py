# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import sys
import ctypes

SKIP_LOADING_TILELANG_SO = os.environ.get("SKIP_LOADING_TILELANG_SO", "0")

# tvm path
install_tvm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3rdparty", "tvm")
install_cutlass_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "3rdparty", "cutlass")
if os.path.exists(install_tvm_path) and install_tvm_path not in sys.path:
    os.environ["PYTHONPATH"] = install_tvm_path + "/python:" + os.environ.get("PYTHONPATH", "")
    os.environ["TL_CUTLASS_PATH"] = install_cutlass_path + "/include"
    sys.path.insert(0, install_tvm_path + "/python")

develop_tvm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "3rdparty", "tvm")
tvm_library_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "build", "tvm")
develop_cutlass_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "3rdparty", "cutlass")
if os.path.exists(develop_tvm_path) and develop_tvm_path not in sys.path:
    os.environ["PYTHONPATH"] = develop_tvm_path + "/python:" + os.environ.get("PYTHONPATH", "")
    os.environ["TL_CUTLASS_PATH"] = develop_cutlass_path + "/include"
    os.environ["TVM_LIBRARY_PATH"] = tvm_library_path
    sys.path.insert(0, develop_tvm_path + "/python")

import tvm
from . import transform
from . import libinfo
from .engine import lower
from .utils import Profiler, ConvertTorch, TensorSupplyType, cached


def _load_tile_lang_lib():
    """Load Tile Lang lib"""
    if sys.platform.startswith("win32") and sys.version_info >= (3, 8):
        for path in libinfo.get_dll_directories():
            os.add_dll_directory(path)
    # pylint: disable=protected-access
    lib_name = "tilelang" if tvm._ffi.base._RUNTIME_ONLY else "tilelang_module"
    # pylint: enable=protected-access
    lib_path = libinfo.find_lib_path(lib_name, optional=False)
    return ctypes.CDLL(lib_path[0]), lib_path[0]

# only load once here
if SKIP_LOADING_TILELANG_SO == "0":
    _LIB, _LIB_PATH = _load_tile_lang_lib()
