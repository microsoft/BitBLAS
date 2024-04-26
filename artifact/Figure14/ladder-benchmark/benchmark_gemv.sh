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
    # "fp16xfp16_gemv.py"
    # "fp16xint4_gemv.py"
    # "fp16xnf4_gemv.py"
    # "fp16xfp8_gemv.py"
    # "int8xint8_gemv.py"
    # "int8xint1_gemv.py"
    "fp32xmxfp8_gemv.py"
    "fp32xfp32_gemv.py"
)

for script in "${scripts[@]}"; do
    run_python_script $script
done
