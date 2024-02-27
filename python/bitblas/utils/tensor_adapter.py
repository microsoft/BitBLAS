# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tvm
from typing import Union


def tvm_tensor_to_torch(tensor: Union[tvm.te.Tensor, tvm.nd.NDArray]):
    import torch
    from torch.utils.dlpack import from_dlpack

    if isinstance(tensor, tvm.te.Tensor):
        return torch.from_numpy(tensor.numpy())
    elif isinstance(tensor, tvm.nd.NDArray):
        return from_dlpack(tensor)
    else:
        raise RuntimeError("Not supported type: ", type(tensor))
