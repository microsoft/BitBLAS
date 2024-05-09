export LADDER_HOME=$(pwd)/../../..
export LADDER_TVM_HOME=$LADDER_HOME/3rdparty/tvm
export LADDER_CUTLASS_HOME=$LADDER_HOME/3rdparty/cutlass
export PYTHONPATH=$LADDER_HOME/python
export PYTHONPATH=$LADDER_TVM_HOME/python:$PYTHONPATH
export CPLUS_INCLUDE_PATH=$LADDER_CUTLASS_HOME/include

mkdir -p logs

python -u ladder_fp16xfp16_gemm.py | tee logs/ladder_fp16xfp16_gemm.log
python -u ladder_fp16xfp16_gemv.py | tee logs/ladder_fp16xfp16_gemv.log
python -u ladder_fp16xint4_gemm.py | tee logs/ladder_fp16xint4_gemm.log
python -u ladder_fp16xint4_gemv.py | tee logs/ladder_fp16xint4_gemv.log
python -u ladder_int8xint1_gemm.py | tee logs/ladder_int8xint1_gemm.log
python -u ladder_int8xint1_gemv.py | tee logs/ladder_int8xint1_gemv.log
python -u ladder_mxfp8xmxfp8_gemm.py | tee logs/ladder_mxfp8xmxfp8_gemm.log

export TVM_HOME=$(pwd)/../../baseline_framework/roller_tvm
export PYTHONPATH=$TVM_HOME/python
export PYTHONPATH=$(pwd)/../../baseline_framework/Roller/python:$PYTHONPATH

python -u ladder_mxfp8xmxfp8_gemv.py | tee logs/ladder_mxfp8xmxfp8_gemv.log
