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
LADDER_CHECKPOINT_PATH="$CHECKPOINT_PATH/ladder/checkpoints/llama2-70b"

MODEL_PATH=$(pwd)/../../models

force_tune=0
if [[ "$1" == "--force_tune" ]]; then
    force_tune=1
fi



mkdir -p logs/llama2

if [ $force_tune -eq 1 ]; then

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --async_propagation --batch 1 --seq_len 1 --fake_quant -1 2>&1 | tee logs/llama2/llama2-70b_b1_s1_q-1.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --async_propagation --batch 1 --seq_len 1 --fake_quant 0 --bits 4  2>&1 | tee logs/llama2/llama2-70b_b1_s1_q0_b4.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --async_propagation --batch 1 --seq_len 1 --fake_quant 0 --bits 1 --convert_int  2>&1 | tee logs/llama2/llama2-70b_b1_s1_q0_b1_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --async_propagation --batch 1 --seq_len 1 --fake_quant 0 --bits 8 --convert_int 2>&1 | tee logs/llama2/llama2-70b_b1_s1_q0_b8_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --async_propagation --batch 1 --seq_len 4096 --fake_quant -1  2>&1 | tee logs/llama2/llama2-70b_b1_s4096_q-1.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --async_propagation --batch 1 --seq_len 4096 --fake_quant 0 --bits 4  2>&1 | tee logs/llama2/llama2-70b_b1_s4096_q0_b4.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --async_propagation --batch 1 --seq_len 4096 --fake_quant 0 --bits 1 --convert_int 2>&1 | tee logs/llama2/llama2-70b_b1_s4096_q0_b1_int.log

/usr/bin/python -u ./ladder_with_fake_dense_dequantize.py --async_propagation --batch 1 --seq_len 4096 --fake_quant 0 --bits 8 --convert_int 2>&1 | tee logs/llama2/llama2-70b_b1_s4096_q0_b8_int.log

fi
