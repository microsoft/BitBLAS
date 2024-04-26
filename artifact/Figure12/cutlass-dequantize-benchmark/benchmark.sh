export CUDA_VISIBLE_DEVICES=1

export PYTHONPATH=/root/unity/python
python -u /workspace/v-leiwang3/lowbit-benchmark/1.matmul_bench/tvm_cublas_invokation/cutlass_fpa_intb.py 2>&1 | tee cutlass_fpa_intb.log
