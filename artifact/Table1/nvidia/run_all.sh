# !/bin/bash
# Description: Run all benchmarks for NVIDIA GPUs.
DEVICE=$1
# if DEVICE is not provided, default to A100
if [ -z "$DEVICE" ]; then
    DEVICE="A100"
fi

FORCE_TUNE=$2

export CHECKPOINT_PATH=$(pwd)/../../checkpoints/$DEVICE/TABLE1

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

python3 ./plot_nvidia_table1.py --device $DEVICE
