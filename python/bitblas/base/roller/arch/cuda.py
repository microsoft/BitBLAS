# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tvm
from tvm.target import Target
from .arch_base import TileDevice
from typing import List, Dict


def check_sm_version(arch: str) -> int:
    sm_version = arch.replace("sm_", "")
    return int(sm_version) if sm_version.isdigit() else -1


class TensorInstruction(object):

    def __init__(
        self,
        name: str,
        intrin_group: Dict,
        shape: List[int],
    ):
        self.name: str = name
        self.intrin_group: Dict = intrin_group
        # only maintain the shape of M and N
        self.shape: List[int] = shape


class CUDA(TileDevice):

    def __init__(self, target: Target):
        self.target = target
        self.sm_version = check_sm_version(self.target.arch)
        device = tvm.runtime.cuda(0)
        if not device.exist:
            raise RuntimeError("Cannot find cuda device 0.")
        self.device: tvm.runtime.Device = device
        self.platform: str = "CUDA"
        self.smem_cap = device.max_shared_memory_per_block
        self.compute_max_core = device.multi_processor_count
        self.warp_size = device.warp_size
        self.compute_capability = device.compute_version.replace(".", "")
        self.reg_cap: int = 65536
        self.max_smem_usage: int = 2 * self.smem_cap
        self.sm_partition: int = 4
        self.l2_cache_size_bytes: int = target.l2_cache_size_bytes
        # the number of transaction size in bytes
        self.transaction_size: List[int] = [32, 128]  # in bytes
        # bandwidth in MB/s, will be used for recommend basic tile size
        # TODO(lei): find some way to get the real bandwidth
        # However, the ratio of bandwidth between different devices can
        # be similar. The bandwidth can work for another devices as well.
        self.bandwidth: List[int] = [750, 12080]
        # get the available tensor instructions during runtime to avoid
        # the dependency of the tensor intrinsics registration
        self.available_tensor_instructions: List[TensorInstruction] = None

    def get_avaliable_tensorintrin_shapes(self):
        from tvm.tir.tensor_intrin.cuda import get_wmma_intrin_group, get_mma_intrin_group

        self.available_tensor_instructions = (
            TensorInstruction("mma", get_mma_intrin_group, [16, 16]),
            TensorInstruction("wmma", get_wmma_intrin_group, [16, 16]),
        )
        return [t.shape for t in self.available_tensor_instructions]
