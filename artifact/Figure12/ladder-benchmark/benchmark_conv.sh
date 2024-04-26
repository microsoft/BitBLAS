#!/bin/bash

run_python_script() {
    echo "$1"
    python -u ./$1 2>&1
    if [ $? -ne 0 ]; then
        echo "Error in executing $1"
        # exit 1
    fi
}

scripts=(
    "./conv_nhwc_nhwc_fp16xfp16.py"
    "./conv_nhwc_nhwc_fp16xfp8_e5m2.py"
    "./conv_nhwc_nhwc_bfp16xmxfp8_e5m2.py"
    "./conv_nhwc_nhwc_int8xint8.py"
    "./conv_nhwc_nhwc_int4xint4.py"
    "./conv_nhwc_nhwc_int4xint1.py"
)

for script in "${scripts[@]}"; do
    run_python_script $script | tee logs/$script.log
done
