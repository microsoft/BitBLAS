from vllm._C import ops
import torch
import numpy as np
# out = ops.awq_gemm(reshaped_x, qweight, scales, qzeros, pack_factor)
import time
shapes = [
    (1,14336,57344),
    (32,14336,57344),
    (4096,14336,57344),
    (1,8192,28672),
    (32,8192,28672),
    (4096,8192,28672),
]
for M, N, K in shapes:
    groups = 128
    bits = 4
    pack_factor = 8
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    qweight = torch.randint(0, 255, (K, N // pack_factor), device="cuda", dtype=torch.int32)
    scales = torch.randn(K // groups, N, device="cuda", dtype=torch.float16)
    qzeros = torch.randint(0, 255, (K // groups, N // pack_factor), device="cuda", dtype=torch.int32)
    out = ops.awq_gemm(x, qweight, scales, qzeros, pack_factor)
    # benchmark
    # torch inductor 
    def get_runtime():
        tic = time.time()
        _ = ops.awq_gemm(x, qweight, scales, qzeros, pack_factor)
        return (time.time() - tic) * 1000
    with torch.no_grad():
        st = time.time()
        while time.time() - st < 1.0:
            get_runtime() # warmup
        times = [get_runtime() for i in range(100)]
        print(f"vllm llama run {M} {N} {K} avg: {np.mean(times)} ms")
        torch.cuda.synchronize()