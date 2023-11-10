import tvm


class MI50:
    def __init__(self):
        self.reg_cap = 32768
        self.smem_cap = 65536
        self.compute_max_core = 60
        self.warp_size = 64
        self.sm_partition = 4
        self.transaction_size = [32, 128]   # in bytes
        self.max_smem_usage = 65536
        self.bandwidth = [900, 8000]
        self.platform = "ROCm"
        self.compute_capability = "gfx906"
        self.target = tvm.target.Target("hip")
