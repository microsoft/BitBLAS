# Reproduce the results of Figure 11

Figure 11 provides a comparative analysis of memory usage across two machine learning models, LLAMA and BLOOM, using various inference frameworks and precision settings. The memory usage is measured in megabytes (MB) and is benchmarked across batch sizes and sequence lengths (BS1 SEQ1, BS32 SEQ1, BS1 SEQ4096).

Run the following command to generate the results of Figure 11:

```bash
# use the paper result
python3 run_all.py
# reproduce the result
python3 run_all.py --reproduce  # This may take hours to run
```

The `run_all.py` script has the following options:

- `--reproduce`: bool, whether to reproduce the results, otherwise it will use our logges paper result, default value is `False`.

The result will be saved in the `pdf` and `png` directory, respectively. For example, the reproduced result is:

![Figure 11](./png/memory_usage_a100.png)

## Notes on Reproducing the Results

if your want to reproduce single settings, please check out:

```bash
python measure_memory.py --model llama --framework pytorch --batch_size 1 --seq_len 1
```

The output will be saved in the `log` directory. For example, the output of the above command is:

```
Measure the memory for llama batch 1 seq 1 under pytorch
{'llama_pytorch_1_1': 3872}
```

The options of `measure_memory.py` are:

- `--model`: str, the model to measure, default value is `llama`, available values are `llama` and `bloom`.
- `--framework`: str, the framework to measure, default value is `pytorch`, available values are `pytorch` `onnxruntime`, `tensorrt`, `welder`, `vllm`, `ladder`, `ladder_fp16_int4`, `ladder_fp16_nf4`, `ladder_fp8_fp8`, `ladder_mxfp8_mxfp8`, `ladder_int8_int1`
-- `--batch_size`: int, the batch size to measure, default value is `1`.
-- `--seq_len`: int, the sequence length to measure, default value is `1`.


As we do not provide Welder Execute Binaries in our checkpoints (the Welder Execute Binaries are too big), so the scripts will first compile the welder binaries from onnx model, and then run the welder binaries to measure the memory usage. The compilation process may take a while.

As we do not provide TensoRT Execute Binaries in our checkpoints (the TensorRT Execute Binaries are too big), so the scripts will also first compile the TensorRT binaries from onnx model, and then run the TensorRT binaries to measure the memory usage. The compilation process may take a while.
