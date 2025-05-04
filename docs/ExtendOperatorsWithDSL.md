### Using BitBLAS from DSL
```python
from bitblas.gpu.matmul_analysis import get_tensorized_func_and_tags
from bitblas.base.roller.policy import TensorCorePolicy, DefaultPolicy
from bitblas.base.arch import CUDA
from bitblas.base.utils import apply_and_build
@tvm.script.ir_module
class MatmulNT:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K], dtype=in_dtype)
        B = T.match_buffer(b, [N, K], dtype=in_dtype)
        C = T.match_buffer(c, [M, N], dtype=out_dtype)

        for i, j, k in T.grid(M, N, K):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = tvm.tir.const(0, out_dtype)
                C[vi, vj] = C[vi, vj] + A[vi, vk].astype(out_dtype) * B[
                    vj, vk
                ].astype(out_dtype)

ir_module = MatmulNT
func = ir_module["main"]
target = tvm.target.Target("nvidia/nvidia-a100")
arch = CUDA(target)
```

Get tuning policy and candidates:

```python
# Tune with SIMT Cuda Core
policy = DefaultPolicy(func=func, arch=arch)
try:
    tensorized_func, tags = get_tensorized_func_and_tags(func, arch.target)
except Exception:
    tags = None
# Tune with Tensor Core if possible
if tags:
    policy = TensorCorePolicy(func=tensorized_func, arch=arch, tags=tags)

configs = policy.emit_config(topk=20)
'''
[BitBLAS] Evaluation with config  {'block': [64, 64], 'warp': [32, 32], 'rstep': [32], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.032 ms
[BitBLAS] Evaluation with config  {'block': [32, 128], 'warp': [16, 64], 'rstep': [64], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.021 ms
[BitBLAS] Evaluation with config  {'block': [128, 32], 'warp': [64, 16], 'rstep': [64], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.023 ms
[BitBLAS] Evaluation with config  {'block': [32, 32], 'warp': [16, 16], 'rstep': [128], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.023 ms
[BitBLAS] Evaluation with config  {'block': [32, 64], 'warp': [16, 32], 'rstep': [128], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.027 ms
[BitBLAS] Evaluation with config  {'block': [64, 32], 'warp': [32, 16], 'rstep': [128], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.025 ms
[BitBLAS] Evaluation with config  {'block': [64, 128], 'warp': [32, 64], 'rstep': [32], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.023 ms
[BitBLAS] Evaluation with config  {'block': [128, 64], 'warp': [64, 32], 'rstep': [32], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.025 ms
[BitBLAS] Evaluation with config  {'block': [16, 64], 'warp': [16, 16], 'rstep': [128], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.037 ms
[BitBLAS] Evaluation with config  {'block': [64, 16], 'warp': [16, 16], 'rstep': [128], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.037 ms
[BitBLAS] Evaluation with config  {'block': [128, 128], 'warp': [64, 64], 'rstep': [32], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.026 ms
[BitBLAS] Evaluation with config  {'block': [16, 128], 'warp': [16, 32], 'rstep': [64], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.043 ms
[BitBLAS] Evaluation with config  {'block': [128, 16], 'warp': [32, 16], 'rstep': [64], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.042 ms
[BitBLAS] Evaluation with config  {'block': [32, 256], 'warp': [16, 128], 'rstep': [64], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.025 ms
[BitBLAS] Evaluation with config  {'block': [256, 32], 'warp': [128, 16], 'rstep': [64], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.029 ms
[BitBLAS] Evaluation with config  {'block': [64, 256], 'warp': [32, 128], 'rstep': [32], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.028 ms
[BitBLAS] Evaluation with config  {'block': [256, 64], 'warp': [128, 32], 'rstep': [32], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.027 ms
[BitBLAS] Evaluation with config  {'block': [128, 256], 'warp': [64, 128], 'rstep': [32], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.044 ms
[BitBLAS] Evaluation with config  {'block': [256, 128], 'warp': [128, 64], 'rstep': [32], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.040 ms
[BitBLAS] Evaluation with config  {'block': [16, 256], 'warp': [16, 64], 'rstep': [64], 'use_tc': True, 'vectorize': {'A_reindex': 8, 'B_reindex': 8}}
[BitBLAS] Time cost of this config: 0.047 ms
'''
```

Apply and build and get best code generation result:
```python
cpresults, best = apply_and_build(func, configs, arch, parallel_build=True)
# get the best code generation result.
print(best.code)
'''
extern "C" __global__ void __launch_bounds__(128) default_function_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C) {
  ...
}
'''
```

we also provide something interesting with DSL.

#### Auto Tensorization

Say we currently have two policies, one is for SIMT Cuda Core, another is for TensorCore. The decision to utilize a TensorCore policy over a SIMT Cuda Core policy can be enhanced by the integration of an auto-tensorization strategy, it allows BitBLAS to automatically select if the DSL Expression can uitlize TensorCore.

![Auto Tensorization](./images/auto_tensorize.png)

```python
# Assume func is conv2d, after this invocation, the tensorized_func is the tensorized version of the conv2d, otherwise, the tags is None.
tensorized_func, tags = get_tensorized_func_and_tags(func, arch.target)
```

#### Tune with dynamic symbolic

As in LLM Serving, the input shape is dynamic, we can use the dynamic symbolic to generate high performance kernel with dynamic shape.

```python
@tvm.script.ir_module
class MatmulNT:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        m = T.int32()
        A = T.match_buffer(a, [m, K], dtype=in_dtype)
        B = T.match_buffer(b, [N, K], dtype=in_dtype)
        C = T.match_buffer(c, [m, N], dtype=out_dtype)

        for i, j, k in T.grid(m, N, K):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = tvm.tir.const(0, out_dtype)
                C[vi, vj] = C[vi, vj] + A[vi, vk].astype(out_dtype) * B[
                    vj, vk
                ].astype(out_dtype)

from bitblas import fast_tune_with_dynamic_range
# Tune with dynamic symbolic
scheduled_ir_module = fast_tune_with_dynamic_range(
    func, target, topk=topk, parallel_build=True, 
    dynamic_range={
        "M": [1, 1024]
    }
)

# finally, we will generate a dispatch func to dispatch the kernel with dynamic symbolic.
'''
@IRModule
class MatmulNT:

    def matmul_nt_opt_m_1(A: Tensor, T_reshape: Tensor, m: int):
        ...

    def matmul_nt_opt_m_256(A: Tensor, T_reshape: Tensor, m: int):
        ...

    def dispatcher(args):
        if m <= 1:
            matmul_nt_opt_m_1(A.data, T_reshape.data, m)
        if m > 1 and m <= 256:
            matmul_nt_opt_m_256(A.data, T_reshape.data, m)
        if m > 256:
            matmul_nt_m_256(A.data, T_reshape.data, m)
'''
```

You can find some example dsl implementation in `python/bitblas/ops/impl` and `benchmark/dsl`, see more examples and tutorials in [apache/tvm](https://github.com/apache/tvm)

