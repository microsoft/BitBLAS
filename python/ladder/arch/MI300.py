# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tvm


class MI300:
    def __init__(self):
        self.reg_cap = 32768
        self.smem_cap = 65536
        self.compute_max_core = 104
        self.warp_size = 64
        self.sm_partition = 4
        self.transaction_size = [32, 128]   # in bytes
        self.max_smem_usage = 65536
        self.bandwidth = [1300, 14000]
        self.platform = "ROCm-CDNA2"
        self.compute_capability = "gfx942"
        self.target = tvm.target.Target("hip --mcpu=gfx942")
