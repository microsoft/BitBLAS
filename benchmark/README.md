# Speedup Benchmark vs Vendor Libraries

This part presents a benchmark comparison between our custom library, BitBLAS, and various vendor libraries (cuBLAS, CUTLASS, bitsandbytes, faster-transformer, tensorrt-llm, vLLM, and Marlin) across different matrix operation types (GEMM, GEMV) and data formats (float16xfloat16, int8xint8, float16xint4/af4). The benchmarks are conducted on NVIDIA GPUs - 24GB GTX 3090 and 80GB A100, with CUDA 12.1 installed.

## Benchmark Overview

### Tested Operations and Formats

- GEMM (General Matrix Multiply) and GEMV (General Matrix-Vector Multiply)
- Data formats: float16, int8, float16xint4/af4

### Hardware

- NVIDIA GTX 3090 (24GB)
- NVIDIA A100 (80GB)

### Software

- CUDA 12.1
- Compared libraries: cuBLAS, CUTLASS, bitsandbytes, faster-transformer, tensorrt-llm, vLLM, Marlin

## Results Summary

### GTX 3090 Benchmarks

- **Float16 and Int8 GEMM with Tensorcore**: BitBLAS matches the performance of cuBLAS and CUTLASS.
- **Float16xaf4 GEMV and GEMM**: BitBLAS achieves 2x the speed of bitsandbytes and 4x the base float16 performance.
- **Optimal performance** in float16xint4 GEMM.

### A100 Benchmarks

- **Int4 Dequantize Performance**: BitBLAS outperforms bitsandbytes, faster-transformer, tensorrt-llm, vLLM, and Marlin.

## Benchmark Configuration

The benchmark configurations for each test scenario are detailed below:

<div style="text-align:center">

|config|Provider|M|N|K|
|:---:|:---:|:---:|:---:|:---:|
|V0|None|1|16384|16384|
|V1|BLOOM|1|43008|14336|
|V2|BLOOM|1|14336|14336|
|V3|BLOOM|1|57344|14336|
|V4|BLOOM|1|14336|57344|
|V5|OPT|1|9216|9216|
|V6|OPT|1|36864|9216|
|V7|OPT|1|9216|36864|
|V8|LLAMA|1|22016|8192|
|V9|LLAMA|1|8192|22016|
|V10|LLAMA-2|1|8192|8192|
|V11|LLAMA-2|1|28672|8192|
|V12|LLAMA-2|1|8192|28672|
|M0|None|16384|16384|16384|
|M1|BLOOM|8192|43008|14336|
|M2|BLOOM|8192|14336|14336|
|M3|BLOOM|8192|57344|14336|
|M4|BLOOM|8192|14336|57344|
|M5|OPT|8192|9216|9216|
|M6|OPT|8192|36864|9216|
|M7|OPT|8192|9216|36864|
|M8|LLAMA|8192|22016|8192|
|M9|LLAMA|8192|8192|22016|
|M10|LLAMA-2|8192|8192|8192|
|M11|LLAMA-2|8192|28672|8192|
|M12|LLAMA-2|8192|8192|28672|

</div>

## Reproducing Benchmarks

To reproduce the 3rdparty frameworks' benchmark results, please refer to the [mlc-benchmark repository](https://github.com/LeiWang1999/mlc-benchmark).

## Benchmark Images

- GTX 3090
  - ![3090-gemm-fp16](../images/figures/op_benchmark_3090_fp16_gemm.png)
  - ![3090-gemm-s8](../images/figures/op_benchmark_3090_s8_gemm.png)
  - ![3090-af4-gemv](../images/figures/op_benchmark_3090_af4_gemv.png)
  - ![3090-af4-gemm](../images/figures/op_benchmark_3090_af4_gemm.png)

- A100
  - ![a100-wq-gemv](../images/figures/op_benchmark_a100_wq_gemv.png)
  - ![a100-wq-gemm](../images/figures/op_benchmark_a100_wq_gemm.png)

This streamlined document provides a concise overview of the benchmarks conducted, highlighting the key comparisons and findings.

# Speedup Benchmark vs Vendor Libraries

We benchmark GEMM and GEMV operations on 24GB GTX 3090 and 80GB A100 GPUs, and both of them are with CUDA 12.1 installed. We compare the performance of our library against vendor libraries (cuBLAS, CUTLASS) for float16 gemm and int8 gemm with tensorcore, and compared the performance of float16xint4/af4 with CUTLASS, bitsandbytes, faster-transformer, tensorrt-llm, vLLM and Marlin.

In detail, we compare the performance of our library against vendor libraries (cuBLAS, CUTLASS) on GTX 3090 for float16 gemm and int8 gemm with tensorcore, the result shows that BitBLAS can achieve the same performance as cuBLAS and CUTLASS. We also explore the inconsistent format, like float16xaf4 gemv and gemm on 3090, we find that BitBLAS can achieve 2x speedup over bitsandbytes implementation and 4x to the base float16 implementation. And get optimal performance on float16xint4 gemm as well.

On A100, we compare the int4 dequantize performance of BitBLAS with bitsandbytes, faster-transformer, tensorrt-llm, vLLM and Marlin, and BitBLAS also get optimal performance.

To reproduce the 3rdparty benchmark results, please checkout [mlc-benchmark](https://github.com/LeiWang1999/mlc-benchmark).


![3090-gemm-fp16](../images/figures/op_benchmark_3090_fp16_gemm.png)

![3090-gemm-s8](../images/figures/op_benchmark_3090_s8_gemm.png)

![3090-af4-gemv](../images/figures/op_benchmark_3090_af4_gemv.png)

![3090-af4-gemm](../images/figures/op_benchmark_3090_af4_gemm.png)


![a100-wq-gemv](../images/figures/op_benchmark_a100_wq_gemv.png)

![a100-wq-gemm](../images/figures/op_benchmark_a100_wq_gemm.png)

the benchmark configuration is as follows:
config	Provider	M	N	K
V0	None	1	16384	16384
V1	BLOOM	1	43008	14336
V2	BLOOM	1	14336	14336
V3	BLOOM	1	57344	14336
V4	BLOOM	1	14336	57344
V5	OPT	1	9216	9216
V6	OPT	1	36864	9216
V7	OPT	1	9216	36864
V8	LLAMA	1	22016	8192
V9	LLAMA	1	8192	22016
V10	LLAMA-2	1	8192	8192
V11	LLAMA-2	1	28672	8192
V12	LLAMA-2	1	8192	28672
M0	None	16384	16384	16384
M1	BLOOM	8192	43008	14336
M2	BLOOM	8192	14336	14336
M3	BLOOM	8192	57344	14336
M4	BLOOM	8192	14336	57344
M5	OPT	8192	9216	9216
M6	OPT	8192	36864	9216
M7	OPT	8192	9216	36864
M8	LLAMA	8192	22016	8192
M9	LLAMA	8192	8192	22016
M10	LLAMA-2	8192	8192	8192
M11	LLAMA-2	8192	28672	8192
M12	LLAMA-2	8192	8192	28672
