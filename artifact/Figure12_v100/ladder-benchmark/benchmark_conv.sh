# !/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

run_python_script() {
    echo "$1"
    python -u ./$1 2>&1
    if [ $? -ne 0 ]; then
        echo "Error in executing $1"
        # exit 1
    fi
}

scripts=(
    "conv_nhwc_nhwc_fp16xfp16.py"
    "conv_nhwc_nhwc_fp16xfp8_e5m2.py"    
    "direct_conv_nhwc_nhwc_fp16xfp16.py"
    "direct_conv_nhwc_nhwc_fp16xfp8_e5m2.py"
    "direct_conv_nhwc_nhwc_fp32xmxfp8_e5m2.py"
    "direct_conv_nhwc_nhwc_int8xint4.py"
)

for script in "${scripts[@]}"; do
    # remove .py extension from script name as log file name
    log_file_name=$(echo $script | sed 's/\.py//')
    run_python_script $script | tee logs/$log_file_name.log
done
