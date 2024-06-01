export LADDER_HOME=$(pwd)/../../../..
export LADDER_TVM_HOME=$LADDER_HOME/3rdparty/tvm
export LADDER_CUTLASS_HOME=$LADDER_HOME/3rdparty/cutlass
export PYTHONPATH=$LADDER_HOME/python
export PYTHONPATH=$LADDER_TVM_HOME/python:$PYTHONPATH
export CPLUS_INCLUDE_PATH=$LADDER_CUTLASS_HOME/include

mkdir -p logs

python -u conformer_fp16.py 2>&1 | tee logs/conformer_fp16.log

python -u conformer_int4xint4.py 2>&1 | tee logs/conformer_int4xint4.log

python -u conformer_int8xint4.py 2>&1 | tee logs/conformer_int8xint4.log

python -u vit_fp16.py 2>&1 | tee logs/vit_fp16.log

python -u vit_fp16xfp8.py 2>&1 | tee logs/vit_fp16xfp8.log

python -u vit_int4xint4.py 2>&1 | tee logs/vit_int4xint4.log
