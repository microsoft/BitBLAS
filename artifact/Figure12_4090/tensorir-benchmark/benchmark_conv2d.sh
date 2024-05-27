# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

export CUDA_VISIBLE_DEVICES=0
export TVM_HOME=$(pwd)/../../baseline_framework/tvm_v0.14.0
export PYTHONPATH=$TVM_HOME/python

force_tune=0
if [[ "$1" == "--force_tune" ]]; then
    force_tune=1
fi

echo "[TensorIR] Using checkpoint path: $CHECKPOINT_PATH"

FP16_LOG_PATH="$CHECKPOINT_PATH/tensorir/logs"

if [ -d "$FP16_LOG_PATH" ] && [ $force_tune -eq 0 ]; then
    echo "[TensorIR] Log directory logs already exists in checkpoints directory. Copying to current directory..."
    # if the log directory already exists in current directory, remove it
    if [ -d "./logs" ]; then
        rm -r "./logs"
    fi
    cp "$FP16_LOG_PATH" "./logs" -r
else

echo "[TensorIR] Running benchmark..."

python3 meta_conv_nhwc.py --N 128 --C 64 --H 56 --W 56 --F 64 --K  3 --S 1 --P 1 2>&1 | tee meta_conv_nhwc_n128_c64_h56_w56_f64_k3_s1_p1.log
python3 meta_conv_nhwc.py --N 128 --C 64 --H 56 --W 56 --F 64 --K  1 --S 1 --P 0 2>&1 | tee meta_conv_nhwc_n128_c64_h56_w56_f64_k1_s1_p0.log
python3 meta_conv_nhwc.py --N 128 --C 128 --H 28 --W 28 --F 128 --K  3 --S 1 --P 1 2>&1 | tee meta_conv_nhwc_n128_c128_h28_w28_f128_k3_s1_p1.log
python3 meta_conv_nhwc.py --N 128 --C 512 --H 28 --W 28 --F 128 --K  1 --S 1 --P 0 2>&1 | tee meta_conv_nhwc_n128_c512_h28_w28_f128_k1_s1_p0.log
python3 meta_conv_nhwc.py --N 1 --C 64 --H 56 --W 56 --F 64 --K  3 --S 1 --P 1 2>&1 | tee meta_conv_nhwc_n1_c64_h56_w56_f64_k3_s1_p1.log
python3 meta_conv_nhwc.py --N 1 --C 64 --H 56 --W 56 --F 64 --K  1 --S 1 --P 1 2>&1 | tee meta_conv_nhwc_n1_c64_h56_w56_f64_k1_s1_p1.log
python3 meta_conv_nhwc.py --N 1 --C 128 --H 28 --W 28 --F 128 --K  3 --S 1 --P 1 2>&1 | tee meta_conv_nhwc_n1_c128_h28_w28_f128_k3_s1_p1.log
python3 meta_conv_nhwc.py --N 1 --C 512 --H 28 --W 28 --F 128 --K  1 --S 1 --P 0 2>&1 | tee meta_conv_nhwc_n1_c512_h28_w28_f128_k1_s1_p0.log
fi
