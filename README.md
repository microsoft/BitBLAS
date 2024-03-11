# BitBLAS

BitBLAS is a lightweight framework designed to generate high-performance CUDA/HIP code for BLAS operators, featuring swizzling and layout propagation. It achieves performance comparable to vendor libraries across various platforms and hardware. BitBLAS aims to assist algorithm developers working on projects like BitNet, GPTQ, and similar endeavors by enabling the rapid implementation of accelerated kernels and their efficient deployment.

Some of the key features of BitBLAS include:
  - Auto Tensorize compute with TensorCore-like hardware instructions.
  - High Performance (Not only FP16xFP16, INT8xINT8, but also FP16xINT4/2/1, INT8xINT4/2/1).
  - With the flexible DSL (TIR Script) to effortlessly craft domain-specific kernels for your situations.
  - Support with dynamic symbolic throuth tvm unity -> generate source code with dynamic shape.
  - BitBLAS first proposed int8xint1 gemv/gemm with 10x/2x speedup over float16xfloat16 on A100, please checkout [op_benchmark_a100_int1_scaling](images/figures/op_benchmark_a100_int1_scaling.png) for detailed input scaling benchmark results.


## Benchmark
BitBLAS can achieve optimal performance across various compute patterns:

- GTX 3090
  - FLOAT16xFLOAT16 with TensorCore ![3090-gemm-fp16](./images/figures/op_benchmark_3090_fp16_gemm.png)
  - INT8xINT8 with TensorCore ![3090-gemm-s8](./images/figures/op_benchmark_3090_s8_gemm.png)
  - FLOAT16xAF4(LUT4) GEMV ![3090-af4-gemv](./images/figures/op_benchmark_3090_af4_gemv.png)
  - FLOAT16xAF4(LUT4) with TensorCore ![3090-af4-gemm](./images/figures/op_benchmark_3090_af4_gemm.png)

- A100
  - WeightOnly GEMV ![a100-wq-gemv](./images/figures/op_benchmark_a100_wq_gemv.png)
  - WeightOnly GEMM with TensorCore ![a100-wq-gemm](./images/figures/op_benchmark_a100_wq_gemm.png)

See more details in our [benchmark](./benchmark) directory.

## Getting Started

- Installation:
  To manually install BitBLAS, please checkout `maint/scripts/installation.sh`. Also Make sure you already have the cuda toolkit (version >= 11) installed in the system. Or you can install from `python setup.py install` or `pip install .` in the root directory. 

- [QuickStart](./docs/QuickStart.md): We provide two primary ways to do the code generation: using a high-level DSL (TensorIR Script), or using packed Operators, from the quick start guide, you can learn how to use BitBLAS to generate high performance kernels with both methods.

- [3rd Party Integration](./integration/): BitBLAS can also be easily integrated to other frameworks, the integration provides some examples of integrating BitBLAS with PyTorch, AutoGPTQ and vLLM.

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the Microsoft Open Source Code of Conduct. For more information see the Code of Conduct FAQ or contact opencode@microsoft.com with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

## Acknowledgement

We learned a lot from the following projects.

- [Apache TVM](https://github.com/apache/tvm): BitBLAS havs adopted TensorIR as our DSL. Additionally, we have customized TVM from the unity branch to incorporate specific features that were required for our project.
- [Microsoft Roller](https://github.com/microsoft/nnfusion/tree/roller): The design and algo inspiration of hardware aware tuning in BitBLAS comes from Roller,.
