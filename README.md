# BitBLAS

BitBLAS is a library to support mixed-precision BLAS operations on GPUs, for example, the $W_{wdtype}A_{adtype}$ mixed-precision matrix multiplication where $C_{cdtype}[M, N] = A_{adtype}[M, K] \times W_{wdtype}[N, K]$.
BitBLAS aims to support efficient mixed-precision DNN model deployment, especially the $W_{wdtype}A_{adtype}$ quantization in large language models (LLMs), for example, the $W_{INT4}A_{FP16}$ in [GPTQ](https://arxiv.org/abs/2210.17323), the $W_{INT2}A_{FP16}$ in [BitDistiller](https://arxiv.org/abs/2402.10631), the $W_{INT1}A_{INT8}$ and $W_{INT2}A_{INT8}$ in [BitNet](https://arxiv.org/abs/2310.11453) and [BitNet-b1.58](https://arxiv.org/abs/2402.17764).


Some of the key features of BitBLAS include:
  - High performance matrix multiplication for both GEMV (e.g., the single batch auto-regressive decode phase in LLM) and GEMM (e.g., the batched auto-regressive decode phase and the prefill phase in LLM):
    - $W_{wdtype}A_{adtype}$ mixed-precision matrix multiplication including FP16xINT4/2/1, INT8xINT4/2/1, etc. Please checkout [support matrix](#support-matrix) for detailed data types support.
    - Matrix multiplication like FP16xFP16 and INT8xINT8.
  - Auto-Tensorization for TensorCore-like hardware instructions.
  - Implemented [integration](./integration/) to [PyTorch](https://pytorch.org/), [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) and [vLLM](https://github.com/vllm-project/vllm) for LLM deployment. Please checkout [benchmark summary](#benchmark-summary) for detailed end2end LLM inference performance.
  - BitBLAS first implemented $W_{INT1}A_{INT8}$ GEMV/GEMM with 10x/2x speedup over $W_{FP16}A_{FP16}$ on A100, please checkout [op_benchmark_a100_int1_scaling](images/figures/op_benchmark_a100_int1_scaling.png) for detailed benchmark results.
  - Support customizing mixed-precision DNN operations for your specific scenarios via the flexible DSL (TIR Script).

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

## Support Matrix

| **A_dtype** | **W_dtype** | **Accum_dtype** | **CUTLASS Support** | **Bitsandbytes Support** | **Marlin Support** | **BitBLAS Support** | **Tested Platform** |
|:-----------:|:-----------:|:---------------:|:-------------------:|:------------------------:|:------------------:|:-------------------:|:-------------------:|
|     FP16    |     FP16    |     FLOAT16     |        **√**        |             ×            |          ×         |        **√**        |  V100/A100/RTX 4090 |
|     FP16    |   FP4_E2M1  |     FLOAT16     |          ×          |           **√**          |          ×         |        **√**        |  V100/A100/RTX 4090 |
|     FP16    |     INT8    |     FLOAT16     |      **>SM80**      |             ×            |          ×         |        **√**        |  V100/A100/RTX 4090 |
|     FP16    |     INT4    |     FLOAT16     |      **>SM80**      |             ×            |      **>SM80**     |        **√**        |  V100/A100/RTX 4090 |
|     FP16    |     INT2    |     FLOAT16     |          ×          |             ×            |          ×         |        **√**        |  V100/A100/RTX 4090 |
|     FP16    |     INT1    |     FLOAT16     |          ×          |             ×            |          ×         |        **√**        |  V100/A100/RTX 4090 |
|     FP16    |     NF4     |     FLOAT16     |          ×          |             √            |          ×         |        **√**        |  V100/A100/RTX 4090 |
|     FP16    |     NF2     |     FLOAT16     |          ×          |             ×            |          ×         |        **√**        |  V100/A100/RTX 4090 |
|     FP16    |     NF1     |     FLOAT16     |          ×          |             ×            |          ×         |        **√**        |  V100/A100/RTX 4090 |
|     INT8    |     INT8    |      INT32      |        **√**        |             ×            |          ×         |        **√**        |  V100/A100/RTX 4090 |
|     INT8    |     INT4    |      INT32      |          ×          |             ×            |          ×         |        **√**        |  V100/A100/RTX 4090 |
| INT8        | INT2        | INT32           | ×                   | ×                        | ×                  | **√**               | V100/A100/RTX 4090  |
| INT8        | INT1        | INT32           | ×                   | ×                        | ×                  | **√**               | V100/A100/RTX 4090  |


## Getting Started

- [Installation](./docs/Installation.md):
  To install BitBLAS, please checkout the document [installation](./docs/Installation.md). Also Make sure you already have the cuda toolkit (version >= 11) installed in the system. Or you can easily install from `pip install bitblas` in the root directory. 

- [QuickStart](./docs/QuickStart.md): BitBLAS provides two Python APIs to perform mixed-precision matrix multiplication:
  - ```bitblas.Matmul``` implements the $W_{wdtype}A_{adtype}$ mixed-precision matrix multiplication of $C_{cdtype}[M, N] = A_{adtype}[M, K] \times W_{wdtype}[N, K]$.
  - ```bitblas.Linear``` is a PyTorch ```nn.Linear```-like module to support a Linear of mixed-precision.

- [Integration](./integration/): Explore how BitBLAS seamlessly integrates with LLM deployment frameworks through our examples. Discover the ease of integrating BitBLAS with PyTorch, AutoGPTQ, and vLLM in the 3rd-party integration examples.

- [Customization](./docs/ExtendOperatorsWithDSL.md): BitBLAS supports implementing customized mixed-precision DNN operations rather than matrix multiplication with the flexible DSL (TIR Script).

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the Microsoft Open Source Code of Conduct. For more information see the Code of Conduct FAQ or contact opencode@microsoft.com with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
