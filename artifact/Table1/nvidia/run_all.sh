# !/bin/bash
# Description: Run all benchmarks for NVIDIA GPUs.
DEVICE=$1
# if DEVICE is not provided, default to A100
if [ -z "$DEVICE" ]; then
    DEVICE="A100"
fi

USE_PAPER=$2

FORCE_TUNE=$3

export CHECKPOINT_PATH=$(pwd)/../../checkpoints/$DEVICE/TABLE1

# if USE_PAPER is not provided, default to False
if [ -z "$USE_PAPER" ]; then
    USE_PAPER="False"
fi

# if use paper results, skip tuning
if [ "$USE_PAPER" = "True" ]; then
    python3 ./plot_nvidia_table1.py --device $DEVICE --reproduce
    # end the script
    exit 0
fi
# otherwise, run the benchmarks

# cublas-benchmark
cd cublas-benchmark
./compile_and_run.sh
cd ..

# amos-benchmark
cd amos-benchmark
./benchmark_amos.sh $FORCE_TUNE
cd ..

# TensorIR-benchmark
cd tensorir-benchmark
./benchmark_tensorir.sh $FORCE_TUNE
cd ..

# roller-benchmark
cd roller-benchmark
./benchmark_roller.sh
cd ..

python3 ./plot_nvidia_table1.py --device $DEVICE --reproduce
