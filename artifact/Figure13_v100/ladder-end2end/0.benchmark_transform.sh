
export LADDER_HOME=$(pwd)/../../..
export LADDER_TVM_HOME=$LADDER_HOME/3rdparty/tvm
export LADDER_CUTLASS_HOME=$LADDER_HOME/3rdparty/cutlass
export PYTHONPATH=$LADDER_HOME/python
export PYTHONPATH=$LADDER_TVM_HOME/python:$PYTHONPATH
export CPLUS_INCLUDE_PATH=$LADDER_CUTLASS_HOME/include
export CHECKPOINT_PATH=$(pwd)/../../checkpoints/Figure13_v100
echo "[LADDER] Using checkpoint path: $CHECKPOINT_PATH"
LADDER_CHECKPOINT_PATH="$CHECKPOINT_PATH/transform"


MODEL_PATH=$(pwd)/../../models

force_tune=0
if [[ "$1" == "--force_tune" ]]; then
    force_tune=1
fi

mkdir -p logs/transform

if [ $force_tune -eq 1 ]; then

# fp16xint4
/usr/bin/python -u ./ladder_from_onnx_transform.py   --batch 1 --seq_len 1 --fake_quant 0 --bits 4  2>&1 | tee logs/transform/llama2-70b_b1_s1_q0_b4.log

# int8xint1
/usr/bin/python -u ./ladder_from_onnx_transform.py   --batch 1 --seq_len 1 --fake_quant 0 --bits 1 --convert_int  2>&1 | tee logs/transform/llama2-70b_b1_s1_q0_b1_int.log

else

# FP16
python -u ladder_from_onnx_transform.py --prebuilt_path $LADDER_CHECKPOINT_PATH/llama2_bs1_seq1_async 2>&1 | tee transform_llama2-70b_b1_s1_q-1.log

python -u ladder_from_onnx_transform.py --prebuilt_path $LADDER_CHECKPOINT_PATH/llama2_bs1_seq4096_async 2>&1 | tee transform_llama2-70b_b1_s4096_q-1.log

# FP16XINT4
python -u ladder_from_onnx_transform.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_from_onnx_transform_fq_0_int_4_-1_bs1_seq1_ci_False 2>&1 | tee transform_llama2-70b_b1_s1_q0_b4.log

python -u ladder_from_onnx_transform.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_4_-1_bs1_seq4096_ci_False 2>&1 | tee transform_llama2-70b_b1_s4096_q0_b4.log

# MXFP8
python -u ladder_from_onnx_transform.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_from_onnx_transform_fq_0_mxfp_8_-1_bs1_seq1_ci_False_cmxfp_async 2>&1 | tee transform_llama2-70b_b1_s1_q0_mxfp8.log

python -u ladder_from_onnx_transform.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_from_onnx_transform_fq_0_mxfp_8_-1_bs1_seq4096_ci_False_cmxfp_async 2>&1 | tee transform_llama2-70b_b1_s4096_q0_fp_mxfp8.log

# INT8XINT1
python -u ladder_from_onnx_transform.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_from_onnx_transform_fq_0_int_1_-1_bs1_seq1_ci_True 2>&1 | tee transform_llama2-70b_b1_s1_q0_b1_int.log

python -u ladder_from_onnx_transform.py --prebuilt_path $LADDER_CHECKPOINT_PATH/ladder_with_fake_dense_dequantize_fq_0_int_1_-1_bs1_seq4096_ci_True 2>&1 | tee transform_llama2-70b_b1_s4096_q0_b1_int.log

fi
