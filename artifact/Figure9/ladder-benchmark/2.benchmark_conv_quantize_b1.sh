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
 # resnet
    ## fp8*fp8
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/resnet-50-b1 --fake_quant 0 --bits 8 --format fp_e5m2 2>&1 | tee logs/resnet-50-b1_fp8_e5m2.log
    ## mxfp8*mxfp8
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/resnet-50-b1 --fake_quant 0 --bits 8 --format mxfp 2>&1 | tee logs/resnet-50-b1_mxfp8_e5m2.log
    ## int4*int4
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/resnet-50-b1 --fake_quant 0 --bits 1 --format int4b 2>&1 | tee logs/resnet-50-b1_int4bxint1.log

    # shufflenet
    # FP8xFP8
    python -u ladder_with_fake_conv_dequantize.py --async_propagation --prefix $MODEL_PATH/shufflenet-b1 --fake_quant 0 --bits 8 --format fp_e5m2 2>&1 | tee logs/shufflenet-b1_fp8_e5m2.log

    # Conformer
    ## int8*int4
    python -u ladder_with_fake_dense_dequantize.py --async_propagation --prefix $MODEL_PATH/Conformer-b1 --fake_quant 0 --bits 4 --format int --convert_int 2>&1 | tee logs/Conformer-b1_int8xint4.log
    ## int4*int4
    python -u ladder_with_fake_dense_dequantize.py --async_propagation --prefix $MODEL_PATH/Conformer-b1 --fake_quant 0 --bits 4 --format int4b 2>&1 | tee logs/Conformer-b1_int4b.log

    # vit
    ## fp8*fp8
    python -u ladder_with_fake_dense_dequantize.py --async_propagation --prefix $MODEL_PATH/vit-b1 --fake_quant 0 --bits 8 --format fp_e5m2 2>&1 | tee logs/vit-b1_fp8_e5m2.log
    ## int4*int4
    python -u ladder_with_fake_dense_dequantize.py --async_propagation --prefix $MODEL_PATH/vit-b1 --fake_quant 0 --bits 4 --format int4b 2>&1 | tee logs/vit-b1_int4b.log
else
    # resnet
    ## fp8*fp8
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/resnet-50-b1_fq_0_fp_e5m2_8_ci_False_async_holistic_fp16 2>&1 | tee logs/resnet-50-b1_fp8_e5m2.log
    ## mxfp8*mxfp8
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/resnet-50-b1_fq_0_mxfp_8_ci_False_async 2>&1 | tee logs/resnet-50-b1_mxfp8_e5m2.log
    ## int4*int4
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/resnet-50-b1_fq_0_int4b_1_ci_False_async_holistic_fp16 2>&1 | tee logs/resnet-50-b1_int4b.log

    # shufflenet
    # FP8xFP8
    python -u ladder_with_fake_conv_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/shufflenet-b1_fq_0_fp_e5m2_8_ci_False_async 2>&1 | tee logs/shufflenet-b1_fp8_e5m2.log

    # Conformer
    ## int8*int4
    python -u ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/Conformer-b1_fq_0_int_8_int4_ci_False_holistic_fp16 2>&1 | tee logs/Conformer-b1_int8xint4.log
    ## int4*int4
    python -u ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/Conformer-b1_fq_0_int4b_4_ci_False_async 2>&1 | tee logs/Conformer-b1_int4bxint1.log

    # vit
    ## fp8*fp8
    python -u ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/vit-b1_fq_0_fp_e5m2_8_ci_False_holistic_fp16 2>&1 | tee logs/vit-b1_fp8_e5m2.log
    ## int4*int4
    python -u ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/vit-b1_fq_0_int4b_4_ci_False_holistic_fp16 2>&1 | tee logs/vit-b1_int4b.log
fi