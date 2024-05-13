# !/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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
    "int8xint1_gemm.py"
)

for script in "${scripts[@]}"; do
    # remove .py extension from script name as log file name
    log_file_name=$(echo $script | sed 's/\.py//')
    run_python_script $script | tee logs/$log_file_name.log
done