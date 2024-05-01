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

mkdir -p logs

pip install bitblas

python -u float16xfloat16_gemm.py | tee logs/float16xfloat16_gemm.log
python -u float16xfloat16_gemv.py | tee logs/float16xfloat16_gemv.log
python -u float16xint1_gemm.py | tee logs/float16xint1_gemm.log
python -u float16xint1_gemv.py | tee logs/float16xint1_gemv.log
python -u float16xint2_gemm.py | tee logs/float16xint2_gemm.log
python -u float16xint2_gemv.py | tee logs/float16xint2_gemv.log
python -u float16xint4_gemm.py | tee logs/float16xint4_gemm.log
python -u float16xint4_gemv.py | tee logs/float16xint4_gemv.log
python -u float16xint8_gemm.py | tee logs/float16xint8_gemm.log
python -u float16xint8_gemv.py | tee logs/float16xint8_gemv.log
python -u int4xint1_gemm.py | tee logs/int4xint1_gemm.log
python -u int4xint1_gemv.py | tee logs/int4xint1_gemv.log
python -u int4xint2_gemm.py | tee logs/int4xint2_gemm.log
python -u int4xint2_gemv.py | tee logs/int4xint2_gemv.log
python -u int4xint4_gemm.py | tee logs/int4xint4_gemm.log
python -u int4xint4_gemv.py | tee logs/int4xint4_gemv.log
python -u int8xint1_gemm.py | tee logs/int8xint1_gemm.log
python -u int8xint1_gemv.py | tee logs/int8xint1_gemv.log
python -u int8xint2_gemm.py | tee logs/int8xint2_gemm.log
python -u int8xint2_gemv.py | tee logs/int8xint2_gemv.log
python -u int8xint4_gemm.py | tee logs/int8xint4_gemm.log
python -u int8xint4_gemv.py | tee logs/int8xint4_gemv.log
python -u int8xint8_gemm.py | tee logs/int8xint8_gemm.log
python -u int8xint8_gemv.py | tee logs/int8xint8_gemv.log
