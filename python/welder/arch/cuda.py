import tvm

class cuda:
    def __init__(self):
        device = tvm.runtime.cuda(0)
        if not device.exist:
            raise RuntimeError("Cannot find cuda device 0.")
        self.platform = "CUDA"
        self.smem_cap = device.max_shared_memory_per_block
        self.compute_max_core = device.multi_processor_count
        self.warp_size = device.warp_size
        self.compute_capability = device.compute_version.replace(".", "")
        self.reg_cap = 65536
        self.max_smem_usage = 2 * self.smem_cap
        self.sm_partition = 4
        self.transaction_size = [32, 128]   # in bytes
        self.bandwidth = [750, 12080]
        self.target = tvm.target.cuda(arch="sm_" + self.compute_capability)

        if self.compute_capability >= "80":
            self.cutlass_mma = [16, 8, 16]
        elif self.compute_capability >= "70":
            self.cutlass_mma = [32, 32, 4]
        else:
            self.cutlass_mma = None
