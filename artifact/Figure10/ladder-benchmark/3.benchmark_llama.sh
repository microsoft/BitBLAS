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

MODEL_PATH=$(pwd)/../../models


mkdir -p logs/llama2

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant -1 2>&1 | tee logs/llama2/llama2-70b_b1_s1_q-1.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 4  2>&1 | tee logs/llama2/llama2-70b_b1_s1_q0_b4.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 1 --convert_int  2>&1 | tee logs/llama2/llama2-70b_b1_s1_q0_b1_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 8 --convert_int 2>&1 | tee logs/llama2/llama2-70b_b1_s1_q0_b8_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 32 --seq_len 1 --fake_quant -1 2>&1 | tee logs/llama2/llama2-70b_b32_s1_q-1.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 32 --seq_len 1 --fake_quant 0 --bits 4  2>&1 | tee logs/llama2/llama2-70b_b32_s1_q0_b4.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 32 --seq_len 1 --fake_quant 0 --bits 1 --convert_int  2>&1 | tee logs/llama2/llama2-70b_b32_s1_q0_b1_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 32 --seq_len 1 --fake_quant 0 --bits 8 --convert_int 2>&1 | tee logs/llama2/llama2-70b_b32_s1_q0_b8_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 4096 --fake_quant -1  2>&1 | tee logs/llama2/llama2-70b_b1_s4096_q-1.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 4096 --fake_quant 0 --bits 4  2>&1 | tee logs/llama2/llama2-70b_b1_s4096_q0_b4.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 4096 --fake_quant 0 --bits 1 --convert_int 2>&1 | tee logs/llama2/llama2-70b_b1_s4096_q0_b1_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 4096 --fake_quant 0 --bits 8 --convert_int 2>&1 | tee logs/llama2/llama2-70b_b1_s4096_q0_b8_int.log


# nf4
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama2/llama2-70b_b1_s1_q0_nf4.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 32 --seq_len 1 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama2/llama2-70b_b32_s1_q0_nf4.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 128 --seq_len 1 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama2/llama2-70b_b128_s1_q0_nf4.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 4096 --fake_quant 0 --bits 4 --format nf 2>&1 | tee logs/llama2/llama2-70b_b1_s4096_q0_nf4.log

# fp8
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 8 --format fp_e5m2 2>&1 | tee logs/llama2/llama2-70b_b1_s1_q0_fp_e5m2.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 32 --seq_len 1 --fake_quant 0 --bits 8 --format fp_e5m2 2>&1 | tee logs/llama2/llama2-70b_b32_s1_q0_fp_e5m2.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 128 --seq_len 1 --fake_quant 0 --bits 8 --format fp_e5m2 2>&1 | tee logs/llama2/llama2-70b_b128_s1_q0_fp_e5m2.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 4096 --fake_quant 0 --bits 8 --format fp_e5m2 2>&1 | tee logs/llama2/llama2-70b_b1_s4096_q0_fp_e5m2.log

# mxfp8
/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 1 --fake_quant 0 --bits 8 --format mxfp 2>&1 | tee logs/llama2/llama2-70b_b1_s1_q0_mxfp8.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 32 --seq_len 1 --fake_quant 0 --bits 8 --format mxfp 2>&1 | tee logs/llama2/llama2-70b_b32_s1_q0_mxfp8.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 128 --seq_len 1 --fake_quant 0 --bits 8 --format mxfp 2>&1 | tee logs/llama2/llama2-70b_b128_s1_q0_mxfp8.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --fast_decoding --batch 1 --seq_len 4096 --fake_quant 0 --bits 8 --format mxfp 2>&1 | tee logs/llama2/llama2-70b_b1_s4096_q0_fp_mxfp8.log
