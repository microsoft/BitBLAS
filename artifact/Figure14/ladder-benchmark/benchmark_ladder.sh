# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

export CHECKPOINT_PATH=$(pwd)/../../checkpoints/Figure14
export LADDER_HOME=$(pwd)/../../..
export LADDER_TVM_HOME=$LADDER_HOME/3rdparty/tvm
export LADDER_CUTLASS_HOME=$LADDER_HOME/3rdparty/cutlass
export PYTHONPATH=$LADDER_HOME/python
export PYTHONPATH=$LADDER_TVM_HOME/python:$PYTHONPATH
export CPLUS_INCLUDE_PATH=$LADDER_CUTLASS_HOME/include

echo "[LADDER] Using checkpoint path: $CHECKPOINT_PATH"
LADDER_LOG_PATH="$CHECKPOINT_PATH/ladder/logs"
LADDER_CHECKPOINT_PATH="$CHECKPOINT_PATH"

MODEL_PATH=$(pwd)/../../models

force_tune=0
if [[ "$1" == "--force_tune" ]]; then
    force_tune=1
fi



mkdir -p logs/llama2

if [ $force_tune -eq 1 ]; then

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant -1 2>&1 | tee logs/llama2/llama2-70b_b1_s1_fp16.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 8  2>&1 | tee logs/llama2/llama2-70b_b1_s1_fp16xint8.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 4  2>&1 | tee logs/llama2/llama2-70b_b1_s1_fp16xint4.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 2  2>&1 | tee logs/llama2/llama2-70b_b1_s1_fp16xint2.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 1  2>&1 | tee logs/llama2/llama2-70b_b1_s1_fp16xint1.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 8 --convert_int  2>&1 | tee logs/llama2/llama2-70b_b1_s1_int8xint8_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 4 --convert_int  2>&1 | tee logs/llama2/llama2-70b_b1_s1_int8xint4_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 2 --convert_int  2>&1 | tee logs/llama2/llama2-70b_b1_s1_int8xint2_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 1 --convert_int  2>&1 | tee logs/llama2/llama2-70b_b1_s1_int8xint1_int.log

# int4b
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 4 --convert_int --format int4b  2>&1 | tee logs/llama2/llama2-70b_b1_s1_int4xint4_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 2 --convert_int --format int4b 2>&1 | tee logs/llama2/llama2-70b_b1_s1_int4xint2_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 1 --convert_int  --format int4b 2>&1 | tee logs/llama2/llama2-70b_b1_s1_int4xint1_int.log


# BATCH 1 SEQ 4096
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py  --async_propagation --batch 1 --seq_len 4096 --fake_quant -1 2>&1 | tee logs/llama2/llama2-70b_b1_s4096_fp16.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py  --async_propagation --batch 1 --seq_len 4096 --fake_quant 0 --bits 8  2>&1 | tee logs/llama2/llama2-70b_b1_s4096_fp16xint8.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py  --async_propagation --batch 1 --seq_len 4096 --fake_quant 0 --bits 4  2>&1 | tee logs/llama2/llama2-70b_b1_s4096_fp16xint4.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py  --async_propagation --batch 1 --seq_len 4096 --fake_quant 0 --bits 2  2>&1 | tee logs/llama2/llama2-70b_b1_s4096_fp16xint2.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py  --async_propagation --batch 1 --seq_len 4096 --fake_quant 0 --bits 1  2>&1 | tee logs/llama2/llama2-70b_b1_s4096_fp16xint1.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py  --async_propagation --batch 1 --seq_len 4096 --fake_quant 0 --bits 8 --convert_int  2>&1 | tee logs/llama2/llama2-70b_b1_s4096_int8xint8_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py  --async_propagation --batch 1 --seq_len 4096 --fake_quant 0 --bits 4 --convert_int  2>&1 | tee logs/llama2/llama2-70b_b1_s4096_int8xint4_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py  --async_propagation --batch 1 --seq_len 4096 --fake_quant 0 --bits 2 --convert_int  2>&1 | tee logs/llama2/llama2-70b_b1_s4096_int8xint2_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py  --async_propagation --batch 1 --seq_len 4096 --fake_quant 0 --bits 1 --convert_int  2>&1 | tee logs/llama2/llama2-70b_b1_s4096_int8xint1_int.log

# int4b

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py  --async_propagation --batch 1 --seq_len 4096 --fake_quant 0 --bits 4 --format int4b  2>&1 | tee logs/llama2/llama2-70b_b1_s4096_int4xint4_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py  --async_propagation --batch 1 --seq_len 4096 --fake_quant 0 --bits 2 --format int4b 2>&1 | tee logs/llama2/llama2-70b_b1_s4096_int4xint2_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py  --async_propagation --batch 1 --seq_len 4096 --fake_quant 0 --bits 1 --format int4b 2>&1 | tee logs/llama2/llama2-70b_b1_s4096_int4xint1_int.log

else

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_bs1_seq1 2>&1 | tee llama2-70b_b1_s1_fp16.log
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_8_-1_bs1_seq1_ci_False 2>&1 | tee llama2-70b_b1_s1_fp16xint8.log
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_4_-1_bs1_seq1_ci_False 2>&1 | tee llama2-70b_b1_s1_fp16xint4.log
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_2_-1_bs1_seq1_ci_False 2>&1 | tee llama2-70b_b1_s1_fp16xint2.log
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_1_-1_bs1_seq1_ci_False 2>&1 | tee llama2-70b_b1_s1_fp16xint1.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_8_-1_bs1_seq1_ci_True 2>&1 | tee llama2-70b_b1_s1_int8xint8_int.log
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_4_-1_bs1_seq1_ci_True 2>&1 | tee llama2-70b_b1_s1_int8xint4_int.log
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_2_-1_bs1_seq1_ci_True 2>&1 | tee llama2-70b_b1_s1_int8xint2_int.log
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_1_-1_bs1_seq1_ci_True 2>&1 | tee llama2-70b_b1_s1_int8xint1_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int4b_4_-1_bs1_seq1_ci_True 2>&1 | tee llama2-70b_b1_s1_int4xint4_int.log
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int4b_2_-1_bs1_seq1_ci_True 2>&1 | tee llama2-70b_b1_s1_int4xint2_int.log
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int4b_1_-1_bs1_seq1_ci_True 2>&1 | tee llama2-70b_b1_s1_int4xint1_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_bs1_seq4096_async 2>&1 | tee  llama2-70b_b1_s4096_fp16.log
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_8_-1_bs1_seq4096_ci_False_async 2>&1 | tee llama2-70b_b1_s4096_fp16xint8.log
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_4_-1_bs1_seq4096_ci_False_async 2>&1 | tee llama2-70b_b1_s4096_fp16xint4.log
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_2_-1_bs1_seq4096_ci_False_async 2>&1 | tee llama2-70b_b1_s4096_fp16xint2.log 
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_1_-1_bs1_seq4096_ci_False_async 2>&1 | tee llama2-70b_b1_s4096_fp16xint1.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_8_-1_bs1_seq4096_ci_True_async 2>&1 | tee llama2-70b_b1_s4096_int8xint8_int.log
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_4_-1_bs1_seq4096_ci_True_async 2>&1 | tee llama2-70b_b1_s4096_int8xint4_int.log
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_2_-1_bs1_seq4096_ci_True_async 2>&1 | tee llama2-70b_b1_s4096_int8xint2_int.log
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_1_-1_bs1_seq4096_ci_True_async 2>&1 | tee llama2-70b_b1_s4096_int8xint1_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int4b_4_-1_bs1_seq4096_ci_False_async 2>&1 | tee llama2-70b_b1_s4096_int4xint4_int.log
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int4b_2_-1_bs1_seq4096_ci_False_async 2>&1 | tee llama2-70b_b1_s4096_int4xint2_int.log
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int4b_1_-1_bs1_seq4096_ci_False_async 2>&1 | tee llama2-70b_b1_s4096_int4xint1_int.log


fi
