# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

export LADDER_HOME=$(pwd)/../../..
export LADDER_TVM_HOME=$LADDER_HOME/3rdparty/tvm
export LADDER_CUTLASS_HOME=$LADDER_HOME/3rdparty/cutlass
export PYTHONPATH=$LADDER_HOME/python
export PYTHONPATH=$LADDER_TVM_HOME/python:$PYTHONPATH
export CPLUS_INCLUDE_PATH=$LADDER_CUTLASS_HOME/include

echo "[LADDER] Using checkpoint path: $CHECKPOINT_PATH"
LADDER_LOG_PATH="$CHECKPOINT_PATH/ladder/logs"
LADDER_CHECKPOINT_PATH="$CHECKPOINT_PATH/ladder/checkpoints"

MODEL_PATH=$(pwd)/../../models

force_tune=0
if [[ "$1" == "--force_tune" ]]; then
    force_tune=1
fi

MODEL_PATH=$(pwd)/../../models

if [ $force_tune -eq 1 ]; then

    # FP8xFP8
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/resnet-50-b128 --fake_quant 0 --bits 8 --format fp_e5m2 2>&1 | tee logs/resnet-50-b128_fp8_e5m2.log
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/shufflenet-b128 --fake_quant 0 --bits 8 --format fp_e5m2 2>&1 | tee logs/shufflenet-b128_fp8_e5m2.log
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/Conformer-b128 --fake_quant 0 --bits 8 --format fp_e5m2 2>&1 | tee logs/Conformer-b128_fp8_e5m2.log
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/vit-b128 --fake_quant 0 --bits 8 --format fp_e5m2 2>&1 | tee logs/vit-b128_fp8_e5m2.log


    # INT4xINT4
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/resnet-50-b128 --fake_quant 0 --bits 4 --format int4b 2>&1 | tee logs/resnet-50-b128_int4b.log
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/shufflenet-b128 --fake_quant 0 --bits 4 --format int4b 2>&1 | tee logs/shufflenet-b128_int4b.log
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/Conformer-b128 --fake_quant 0 --bits 4 --format int4b 2>&1 | tee logs/Conformer-b128_int4b.log
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/vit-b128 --fake_quant 0 --bits 4 --format int4b 2>&1 | tee logs/vit-b128_int4b.log

    # INT4xINT1
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/resnet-50-b128 --fake_quant 0 --bits 1 --format int4b 2>&1 | tee logs/resnet-50-b128_int4bxint1.log
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/shufflenet-b128 --fake_quant 0 --bits 1 --format int4b 2>&1 | tee logs/shufflenet-b128_int4bxint1.log
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/Conformer-b128 --fake_quant 0 --bits 1 --format int4b 2>&1 | tee logs/Conformer-b128_int4bxint1.log
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/vit-b128 --fake_quant 0 --bits 1 --format int4b 2>&1 | tee logs/vit-b128_int4bxint1.log


    # MxFP8_e5m2
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/resnet-50-b128 --fake_quant 0 --bits 8 --format mxfp 2>&1 | tee logs/resnet-50-b128_mxfp8_e5m2.log
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/shufflenet-b128 --fake_quant 0 --bits 8 --format mxfp 2>&1 | tee logs/shufflenet-b128_mxfp8_e5m2.log
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/Conformer-b128 --fake_quant 0 --bits 8 --format mxfp 2>&1 | tee logs/Conformer-b128_mxfp8_e5m2.log
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/vit-b128 --fake_quant 0 --bits 8 --format mxfp 2>&1 | tee logs/vit-b128_mxfp8_e5m2.log

    # FP8xFP8
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/resnet-50 --fake_quant 0 --bits 8 --format fp_e5m2 2>&1 | tee logs/resnet-50-b128_fp8_e5m2.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/shufflenet --fake_quant 0 --bits 8 --format fp_e5m2 2>&1 | tee logs/shufflenet-b128_fp8_e5m2.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/Conformer --fake_quant 0 --bits 8 --format fp_e5m2 2>&1 | tee logs/Conformer-b128_fp8_e5m2.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/vit --fake_quant 0 --bits 8 --format fp_e5m2 2>&1 | tee logs/vit-b128_fp8_e5m2.log

    # INT4xINT4
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/resnet-50 --fake_quant 0 --bits 4 --format int4b 2>&1 | tee logs/resnet-50-b128_int4b.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/shufflenet --fake_quant 0 --bits 4 --format int4b 2>&1 | tee logs/shufflenet-b128_int4b.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/Conformer --fake_quant 0 --bits 4 --format int4b 2>&1 | tee logs/Conformer-b128_int4b.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/vit --fake_quant 0 --bits 4 --format int4b 2>&1 | tee logs/vit-b128_int4b.log
else
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/Conformer_fq_0_fp_e5m2_8_ci_False 2>&1 | tee logs/Conformer-b128_fp8_e5m2.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/Conformer_fq_0_int4b_1_ci_False 2>&1 | tee logs/Conformer-b128_int4b.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/Conformer_fq_0_int4b_4_ci_False 2>&1 | tee logs/Conformer-b128_int4bxint1.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/Conformer_fq_0_mxfp_8_ci_False 2>&1 | tee logs/Conformer-b128_mxfp8_e5m2.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/resnet-50_fq_0_fp_e5m2_8_ci_False 2>&1 | tee logs/resnet-50-b128_fp8_e5m2.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/resnet-50_fq_0_int4b_1_ci_False 2>&1 | tee logs/resnet-50-b128_int4bxint1.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/resnet-50_fq_0_int4b_4_ci_False 2>&1 | tee logs/resnet-50-b128_int4b.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/resnet-50_fq_0_mxfp_8_ci_False 2>&1 | tee logs/resnet-50-b128_mxfp8_e5m2.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/shufflenet_fq_0_fp_e5m2_8_ci_False 2>&1 | tee logs/shufflenet-b128_fp8_e5m2.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/shufflenet_fq_0_int4b_1_ci_False 2>&1 | tee logs/shufflenet-b128_int4bxint1.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/shufflenet_fq_0_int4b_4_ci_False 2>&1 | tee logs/shufflenet-b128_int4b.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/shufflenet_fq_0_mxfp_8_ci_False 2>&1 | tee logs/shufflenet-b128_mxfp8_e5m2.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/vit_fq_0_fp_e5m2_8_ci_False 2>&1 | tee logs/vit-b128_fp8_e5m2.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/vit_fq_0_int4b_1_ci_False 2>&1 | tee logs/vit-b128_int4bxint1.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/vit_fq_0_int4b_4_ci_False 2>&1 | tee logs/vit-b128_int4b.log
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/vit_fq_0_mxfp_8_ci_False 2>&1 | tee logs/vit-b128_mxfp8_e5m2.log
fi