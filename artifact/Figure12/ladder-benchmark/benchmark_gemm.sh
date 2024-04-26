#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

run_python_script() {
    echo "$1"
    python -u ./$1
    if [ $? -ne 0 ]; then
        echo "Error in executing $1"
        # exit 1
    fi
}

scripts=(
    # "fp16xfp16_gemm.py"
    # "fp16xint4_gemm.py"
    # "fp16xnf4_gemm.py"
    # "fp16xfp8_gemm.py"
    "int8xint8_gemm.py"
    "int8xint1_gemm.py"
    # "fp32xmxfp8_gemm.py"
    # "fp32xfp32_gemm.py"
)

for script in "${scripts[@]}"; do
    run_python_script $script
done
