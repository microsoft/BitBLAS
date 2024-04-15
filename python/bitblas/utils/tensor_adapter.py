# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tvm
from typing import Union
from enum import IntEnum
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack

from tvm.relay import TensorType
from tvm._ffi.base import _LIB, c_str
from tvm._ffi._ctypes.types import TVMValue, check_call
from tvm._ffi.runtime_ctypes import (
    TVMArrayHandle,)
import ctypes

TVMPyCapsuleDestructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_c_str_dltensor = c_str("dltensor")
_c_str_used_dltensor = c_str("used_dltensor")


def get_values_from_torch_tensors(tensors, num_args):
    values = (TVMValue * num_args)()
    dlpack_tensors = [to_dlpack(torch_tensor) for torch_tensor in tensors]
    for i, dltensor in enumerate(dlpack_tensors):
        dltensor = ctypes.py_object(dltensor)
        if ctypes.pythonapi.PyCapsule_IsValid(dltensor, _c_str_dltensor):
            ptr = ctypes.pythonapi.PyCapsule_GetPointer(dltensor, _c_str_dltensor)
            # enforce type to make sure it works for all ctypes
            ptr = ctypes.cast(ptr, ctypes.c_void_p)
            handle = TVMArrayHandle()
            check_call(_LIB.TVMArrayFromDLPack(ptr, ctypes.byref(handle)))
            # ndarray = tvm.runtime.ndarray._make_array(handle, False, False)
            ctypes.pythonapi.PyCapsule_SetName(dltensor, _c_str_used_dltensor)
            ctypes.pythonapi.PyCapsule_SetDestructor(dltensor, TVMPyCapsuleDestructor(0))
            values[i].v_handle = ctypes.cast(handle, ctypes.c_void_p)
        else:
            raise ValueError("Invalid DLTensor")
    return values


class TensorSupplyType(IntEnum):
    Integer = 1
    Uniform = 2
    Normal = 3
    Randn = 4
    Zero = 5
    One = 6


def get_tensor_supply(supply_type: TensorSupplyType, opt_shapes: dict = None):

    def var_wrapper(v, opt_shapes):
        if isinstance(v, tvm.tir.Var):
            assert opt_shapes
            assert v.name in opt_shapes
            return opt_shapes[v.name]
        elif isinstance(v, tvm.tir.IntImm):
            return v.value
        else:
            raise RuntimeError("Not supported type: ", type(v))

    def get_tensor(tensor: TensorType) -> torch.Tensor:
        dtype = torch.__getattribute__(str(tensor.dtype))
        device = torch.cuda.current_device()
        shape = [var_wrapper(i, opt_shapes) for i in tensor.shape]
        if supply_type == TensorSupplyType.Integer:
            return torch.randint(low=-2, high=3, size=shape, device=device, dtype=dtype)
        elif supply_type == TensorSupplyType.Uniform:
            return torch.empty(*shape, device=device, dtype=dtype).uniform_(-1.0, 1.0)
        elif supply_type == TensorSupplyType.Normal:
            return torch.empty(*shape, device=device, dtype=dtype).normal_(-1.0, 1.0)
        elif supply_type == TensorSupplyType.Randn:
            return torch.randn(*shape, device=device).to(dtype)
        elif supply_type == TensorSupplyType.Zero:
            return torch.zeros(*shape, device=device, dtype=dtype)
        elif supply_type == TensorSupplyType.One:
            return torch.ones(*shape, device=device, dtype=dtype)
        else:
            raise NotImplementedError(supply_type)

    return get_tensor


def tvm_tensor_to_torch(tensor: Union[tvm.te.Tensor, tvm.nd.NDArray]):
    if isinstance(tensor, tvm.te.Tensor):
        return torch.from_numpy(tensor.numpy())
    elif isinstance(tensor, tvm.nd.NDArray):
        return from_dlpack(tensor)
    else:
        raise RuntimeError("Not supported type: ", type(tensor))
