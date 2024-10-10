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

# Handle TVM_IMPORT_PYTHON_PATH to import tvm from the specified path
TVM_IMPORT_PYTHON_PATH = os.environ.get("TVM_IMPORT_PYTHON_PATH", None)

if TVM_IMPORT_PYTHON_PATH is not None:
    os.environ["PYTHONPATH"] = TVM_IMPORT_PYTHON_PATH + ":" + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, TVM_IMPORT_PYTHON_PATH + "/python")
else:
    install_tvm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3rdparty", "tvm")
    if os.path.exists(install_tvm_path) and install_tvm_path not in sys.path:
        os.environ["PYTHONPATH"] = install_tvm_path + "/python:" + os.environ.get("PYTHONPATH", "")
        sys.path.insert(0, install_tvm_path + "/python")

    develop_tvm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "3rdparty", "tvm")
    tvm_library_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "build", "tvm")
    if os.path.exists(develop_tvm_path) and develop_tvm_path not in sys.path:
        os.environ["PYTHONPATH"] = develop_tvm_path + "/python:" + os.environ.get("PYTHONPATH", "")
        sys.path.insert(0, develop_tvm_path + "/python")
    if os.environ.get("TVM_LIBRARY_PATH") is None:
        os.environ["TVM_LIBRARY_PATH"] = tvm_library_path

install_cutlass_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "3rdparty", "cutlass"
)
if os.path.exists(install_cutlass_path) and install_cutlass_path not in sys.path:
    os.environ["TL_CUTLASS_PATH"] = install_cutlass_path + "/include"

develop_cutlass_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "3rdparty", "cutlass"
)
if os.path.exists(develop_cutlass_path) and develop_cutlass_path not in sys.path:
    os.environ["TL_CUTLASS_PATH"] = develop_cutlass_path + "/include"


import tvm
import tvm._ffi.base

from . import libinfo

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

from .utils import (
    Profiler,  # noqa: F401
    ConvertTorch,  # noqa: F401
    TensorSupplyType,  # noqa: F401
    cached,  # noqa: F401
)
from .layout import (
    Layout, # noqa: F401
    Fragment, # noqa: F401
)
from . import transform, autotuner # noqa: F401
from . import language, transform, engine  # noqa: F401

from .engine import lower  # noqa: F401
