# Reproduce the results of Figure 14

Figure 14 depicts the end-to-end performance comparison of various baseline configurations against our proposed method across different models (M0 to M3) and batch size sequences (BS1 SEQ1 and BS1 SEQ4096). The performance metric is speedup, presented as a ratio over the baseline performance.

Run the following command to generate the results of Figure 14:

```bash
# to use the paper result
python3 run_all.py
# to reproduce
python3 run_all.py --reproduce # This may take 1 hour to finish the ladder tuning process, if you enable --force_tune, it may take 1 day to finish the tuning process.
```

The `run_all.py` script has the following options:

- `--reproduce`: bool, whether to reproduce the results, otherwise it will use our logged paper result, default value is `False`.
- `--force_tune`: bool, whether to force tune the op with AMOS/TensorIR, otherwise use the checkpoints if available, default value is `False`.

The result will be saved in the `pdf` and `png` directory, respectively. For example, the reproduced result is:

![Figure 14](./png/different_bits.png)
