# Reproduce the results of Figure 8

The Figure 8 is about the end-to-end performance of the selected baselines and the proposed method. The end-to-end performance is measured by the inference time of the model. The inference time is measured in seconds.

Run the following command to generate the results of Figure 8:

```bash
python3 run_all.py
```

The `run_all.py` script has the following options:

- `--reproduce`: bool, whether to reproduce the results, otherwise it will use our logges paper result, default value is `False`.
- `--force_tune`: bool, whether to force tune the op with AMOS/TensorIR, otherwise use the checkpoints if available, default value is `False`.

The result will be saved in the `pdf` and `png` directory, respectively. For example, the reproduced result is:

![Figure 8](./png/end2end_a100.png)

## Notes for Reproducing the Results

- As shown in Table2, the ML Compiler AMOS and TensorIR takes too much time to tune a whole end2end model, so we provide the tuned logs and trace files in the `$CHECKPOINTS/Figure8/` directory. The `run_all.py` script will use the logs and trace files to generate the results. If you want to reproduce the results, you can set the `--force_tune` option to `True` to force tune the model with AMOS and TensorIR. (This may take days to finish the tuning process.)

- Moreover, even Ladder can have a giant reduction in tuning time, it still takes a long time to tune the all settings (around 40x models need to be tuned to reproduce all the paper data, This may takes around 10 hours to finish all of the settings tuning), we also provide the tuned logs and some of the precompiled models and trace files in the `$CHECKPOINTS/Figure8/` directory. The `run_all.py` script will use the logs and trace files to generate the results. If you want to reproduce the results, you can set the `--force_tune` option to `True` to force tune the model with Ladder. (This may take hours to finish the tuning process.)

If you want to check the one of the checkpoint that we provide, you can use the following command:

```bash
python ladder_with_fake_dense_dequantize.py --prebuilt_path $CHECKPOINT_PATH/Figure8/ladder/checkpoint/llama2-70b/llama2_bs1_seq1_async
```
