# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from .matmul_dequantize_impl import select_implementation as matmul_dequantize_select_implementation  # noqa: F401
from .matmul_impl import select_implementation as matmul_select_implementation  # noqa: F401
