# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
export CUDA_VISIBLE_DEVICES=0
export LADDER_HOME=$(pwd)/../../..
export LADDER_TVM_HOME=$LADDER_HOME/3rdparty/tvm
export LADDER_CUTLASS_HOME=$LADDER_HOME/3rdparty/cutlass
export PYTHONPATH=$LADDER_HOME/python
export PYTHONPATH=$LADDER_TVM_HOME/python:$PYTHONPATH
export CPLUS_INCLUDE_PATH=$LADDER_CUTLASS_HOME/include

echo "[LADDER] Using checkpoint path: $CHECKPOINT_PATH"
LADDER_LOG_PATH="$CHECKPOINT_PATH/ladder/logs"
LADDER_CHECKPOINT_PATH="$CHECKPOINT_PATH/ladder/checkpoints/llama2-70b"
# LADDER_CHECKPOINT_PATH="/root/Ladder/artifact/Figure9/ladder-benchmark/progress/e2e/llama2-70b"

MODEL_PATH=$(pwd)/../../models

force_tune=0
if [[ "$1" == "--force_tune" ]]; then
    force_tune=1
fi

echo "force_tune is $force_tune"


mkdir -p logs/llama2

if [ $force_tune -eq 1 ]; then

# /usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant -1 2>&1 | tee logs/llama2/llama2-70b_b1_s1_q-1.log

# /usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 32 --seq_len 1 --fake_quant -1 2>&1 | tee logs/llama2/llama2-70b_b32_s1_q-1.log

# /usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 4096 --fake_quant -1  2>&1 | tee logs/llama2/llama2-70b_b1_s4096_q-1.log

# /usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 4  2>&1 | tee logs/llama2/llama2-70b_b1_s1_q0_b4.log

# /usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 1 --convert_int  2>&1 | tee logs/llama2/llama2-70b_b1_s1_q0_b1_int.log

# /usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 8 --convert_int 2>&1 | tee logs/llama2/llama2-70b_b1_s1_q0_b8_int.log


# /usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 32 --seq_len 1 --fake_quant 0 --bits 4  2>&1 | tee logs/llama2/llama2-70b_b32_s1_q0_b4.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --batch 32 --seq_len 1 --fake_quant 0 --bits 1 --convert_int  2>&1 | tee logs/llama2/llama2-70b_b32_s1_q0_b1_int.log

# /usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 32 --seq_len 1 --fake_quant 0 --bits 8 --convert_int 2>&1 | tee logs/llama2/llama2-70b_b32_s1_q0_b8_int.log


# /usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 4096 --fake_quant 0 --bits 4  2>&1 | tee logs/llama2/llama2-70b_b1_s4096_q0_b4.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --batch 1 --seq_len 4096 --fake_quant 0 --bits 1 --convert_int 2>&1 | tee logs/llama2/llama2-70b_b1_s4096_q0_b1_int.log

# /usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 4096 --fake_quant 0 --bits 8 --convert_int 2>&1 | tee logs/llama2/llama2-70b_b1_s4096_q0_b8_int.log


# # nf4
# /usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama2/llama2-70b_b1_s1_q0_nf4.log

# /usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 32 --seq_len 1 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama2/llama2-70b_b32_s1_q0_nf4.log

# /usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 4096 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama2/llama2-70b_b1_s4096_q0_nf4.log

# # fp8
# /usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 8 --format fp_e5m2 2>&1 | tee logs/llama2/llama2-70b_b1_s1_q0_fp_e5m2.log

# /usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 32 --seq_len 1 --fake_quant 0 --bits 8 --format fp_e5m2 2>&1 | tee logs/llama2/llama2-70b_b32_s1_q0_fp_e5m2.log

# /usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 4096 --fake_quant 0 --bits 8 --format fp_e5m2 2>&1 | tee logs/llama2/llama2-70b_b1_s4096_q0_fp_e5m2.log

# # mxfp8
# /usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 8 --format mxfp 2>&1 | tee logs/llama2/llama2-70b_b1_s1_q0_mxfp8.log

# /usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 32 --seq_len 1 --fake_quant 0 --bits 8 --format mxfp --convert_float 2>&1 | tee logs/llama2/llama2-70b_b32_s1_q0_mxfp8.log

# /usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 4096 --fake_quant 0 --bits 8 --format mxfp --convert_float 2>&1 | tee logs/llama2/llama2-70b_b1_s4096_q0_mxfp8.log

else

# FP16
python -u ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/llama2_bs1_seq1_async 2>&1 | tee llama2-70b_b1_s1_q-1.log
python -u ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/llama2_bs32_seq1_async 2>&1 | tee lama2-70b_b32_s1_q-1.log
python -u ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/llama2_bs1_seq4096_async 2>&1 | tee lama2-70b_b1_s4096_q-1.log

# FP16XINT4
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_4_-1_bs1_seq1_ci_False 2>&1 | tee llama2-70b_b1_s1_q0_b4.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_4_-1_bs32_seq1_ci_False 2>&1 | tee llama2-70b_b32_s1_q0_b4.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_4_-1_bs1_seq4096_ci_False 2>&1 | tee llama2-70b_b1_s4096_q0_b4.log

# INT8XINT1
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_1_-1_bs1_seq1_ci_True 2>&1 | tee llama2-70b_b1_s1_q0_b1_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_1_-1_bs32_seq1_ci_True 2>&1 | tee llama2-70b_b32_s1_q0_b1_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_1_-1_bs1_seq4096_ci_True 2>&1 | tee llama2-70b_b1_s4096_q0_b1_int.log

# NF4
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_nf_4_-1_bs1_seq1_ci_False 2>&1 | tee llama2-70b_b1_s1_q0_nf4.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_nf_4_-1_bs32_seq1_ci_False 2>&1 | tee llama2-70b_b32_s1_q0_nf4.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_nf_4_-1_bs1_seq4096_ci_False 2>&1 | tee llama2-70b_b1_s4096_q0_nf4.log

# FP16xFP8
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_fp_e5m2_8_-1_bs1_seq1_ci_False 2>&1 | tee llama2-70b_b1_s1_q0_fp_e5m2.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_fp_e5m2_8_-1_bs32_seq1_ci_False 2>&1 | tee llama2-70b_b32_s1_q0_fp_e5m2.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_fp_e5m2_8_-1_bs1_seq4096_ci_False 2>&1 | tee llama2-70b_b1_s4096_q0_fp_e5m2.log

# MXFP8
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_mxfp_8_-1_bs1_seq1_ci_False 2>&1 | tee  llama2-70b_b1_s1_q0_mxfp8.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_mxfp_8_-1_bs32_seq1_ci_False_cf_True 2>&1 | tee llama2-70b_b32_s1_q0_mxfp8.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_mxfp_8_-1_bs1_seq4096_ci_False_cf_True 2>&1 | tee llama2-70b_b1_s4096_q0_mxfp8.log

fi
