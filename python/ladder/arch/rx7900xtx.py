import tvm


class RX7900Xtx:
    def __init__(self):
        self.reg_cap = 32768
        self.smem_cap = 65536
        self.compute_max_core = 96
        self.warp_size = 32
        self.sm_partition = 4
        self.transaction_size = [32, 128]   # in bytes
        self.max_smem_usage = 65536
        self.bandwidth = [960, 3500]
        self.platform = "ROCm-RDNA3"
        self.compute_capability = "gfx1100"
        self.target = tvm.target.Target("hip")
