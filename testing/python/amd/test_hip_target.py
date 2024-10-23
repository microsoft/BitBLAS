# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tvm

target_backend = "hip"
try:
    target = tvm.target.Target(target_backend)
    print(f"Success: '{target_backend}' backend is available in TVM.")
except ValueError as e:
    print(f"Error: '{target_backend}' backend is not available in TVM. Details: {e}")
