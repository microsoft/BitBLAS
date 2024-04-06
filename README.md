# BitBLAS

BitBLAS is a lightweight framework designed to generate high-performance CUDA/HIP code for BLAS operators, featuring swizzling and layout propagation. It achieves performance comparable to vendor libraries across various platforms and hardware. BitBLAS aims to assist algorithm developers working on projects like BitNet, GPTQ, and similar endeavors by enabling the rapid implementation of accelerated kernels and their efficient deployment.

Some of the key features of BitBLAS include:
  - Auto Tensorize compute with TensorCore-like hardware instructions.
  - High Performance (Not only FP16xFP16, INT8xINT8, but also FP16xINT4/2/1, INT8xINT4/2/1).
  - With the flexible DSL (TIR Script) to effortlessly craft domain-specific kernels for your situations.
  - Support with dynamic symbolic throuth tvm unity -> generate source code with dynamic shape.
  - BitBLAS first proposed int8xint1 gemv/gemm with 10x/2x speedup over float16xfloat16 on A100, please checkout [op_benchmark_a100_int1_scaling](images/figures/op_benchmark_a100_int1_scaling.png) for detailed input scaling benchmark results.

## Integration Example of FasterTransformer with BitBLAS
![FasterTransformer Integration](images/gif/FasterTransformer.gif)


## Benchmark Summary

BitBLAS achieves exceptional performance across a variety of computational patterns. Below are selected results showcasing its capabilities:

- End2End Integration with Quantize Inference Kernel for AutoGPTQ and vLLM.

  <div>
    <img src="./images/figures/end2end_llama_13b_auto_gptq.png" alt="AutoGPTQ end2end performance of llama13b on A100" style="width: 24%;" />
    <img src="./images/figures/end2end_llama_70b_auto_gptq.png" alt="AutoGPTQ end2end performance of llama13b on A100" style="width: 24%;" />
    <img src="./images/figures/end2end_llama_13b_vllm.png" alt="vLLM end2end performance of llama13b on A100" style="width: 24%;" />
    <img src="./images/figures/end2end_llama_70B_vllm.png" alt="vLLM end2end performance of llama13b on A100" style="width: 24%;" />
  </div>

- Weight Only Matmul performance on A100

  <div>
    <img src="./images/figures/op_benchmark_a100_wq_gemv_e7.png" alt="gemm weight only performance on A100" style="width: 49%;" />
    <img src="./images/figures/op_benchmark_a100_wq_gemm_e7.png" alt="gemm weight only performance on A100" style="width: 49%;" />
  </div>



- TensorCore FP16/INT8 GEMM Performance Vs. Vendor Library on A100 and RTX4090

  <div>
    <img src="./images/figures/op_benchmark_consistent_gemm_fp16.png" alt="gemm fp16 performance on 4090 and a100" style="width: 49%;" />
    <img src="./images/figures/op_benchmark_consistent_gemm_int8.png" alt="gemm int8 performance on 4090 and a100" style="width: 49%;" />
  </div>

For more detailed information on benchmark sets with other formats (NF4/FP4) and other devices (GTX 3090), please refer to the [benchmark](./benchmark/README.md).

## Getting Started

- [installation](./docs/Installation.md):
  To install BitBLAS, please checkout the document [installation](./docs/Installation.md). Also Make sure you already have the cuda toolkit (version >= 11) installed in the system. Or you can easily install from `pip install bitblas` in the root directory. 

- [QuickStart](./docs/QuickStart.md): We provide two primary ways to do the code generation: using a high-level DSL (TensorIR Script), or using packed Operators, from the quick start guide, you can learn how to use BitBLAS to generate high performance kernels through Packed Operators API and example usage with PyTorch.

- [Integration](./integration/): Explore how BitBLAS seamlessly integrates with other frameworks through our examples. Discover the ease of integrating BitBLAS with PyTorch, AutoGPTQ, and vLLM in the 3rd Party Integration Examples.

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the Microsoft Open Source Code of Conduct. For more information see the Code of Conduct FAQ or contact opencode@microsoft.com with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.