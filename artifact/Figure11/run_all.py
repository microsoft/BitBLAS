# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import argparse
import os

CHECKPOINT_PATH = os.path.join(os.getcwd(), "../../checkpoints/Figure11")
os.environ["CHECKPOINT_PATH"] = CHECKPOINT_PATH

parser = argparse.ArgumentParser()

parser.add_argument("--reproduce", action="store_true", help="reproduce, otherwise use the paper results", default=False)
parser.add_argument(
    "--force_tune",
    action="store_true",
    help="force_tune, otherwise use the checkpoints if available",
    default=False,
)

args = parser.parse_args()
reproduce = args.reproduce
force_tune = args.force_tune

if not reproduce:
    print("Using the paper results")
    os.system(f"python3 plot_memory_usage.py")
else:
    print("Reproducing the results")
    # initialize the checkpoints
    # initialize tensorrt engine
    os.system(f"cd tensorrt-benchmark; ./initialize_tensorrt.sh")
    # initialize welder
    os.system(f"cd welder-benchmark; ./initialize_welder.sh")
    # initialize ladder
    os.system(f"cd ladder-benchmark; ./initialize_ladder.sh")
    # initialize vllm
    for model in ["{model}", "bloom"]:
        for batch_size, seq_len in [
                (1, 1),
                (32, 1),
                (1, 4096)
            ]:
            # reproduce the results for inductor
            os.system(f"python measure_memory --framework pytorch --model {model} --batch_size {batch_size} --seq_len {seq_len}")
            # reproduce the results for onnxruntime
            os.system(f"python measure_memory --framework onnxruntime --model {model} --batch_size {batch_size} --seq_len {seq_len}")
            # reproduce the results for tensorrt
            os.system(f"python measure_memory --framework tensorrt --model {model} --batch_size {batch_size} --seq_len {seq_len}")
            # reproduce the results for welder
            os.system(f"python measure_memory --framework welder --model {model} --batch_size {batch_size} --seq_len {seq_len}")
            # reproduce the results for vllm
            os.system(f"python measure_memory --framework vllm --model {model} --batch_size {batch_size} --seq_len {seq_len}")
            # reproduce the results for ladder
            os.system(f"python measure_memory --framework ladder --model {model} --batch_size {batch_size} --seq_len {seq_len}")
            # reproduce the results for ladder_fp16_int4
            os.system(f"python measure_memory --framework ladder_fp16_int4 --model {model} --batch_size {batch_size} --seq_len {seq_len}")
            # reproduce the results for ladder_fp16_nf4
            os.system(f"python measure_memory --framework ladder_fp16_nf4 --model {model} --batch_size {batch_size} --seq_len {seq_len}")
            # reproduce the results for ladder_fp8_fp8
            os.system(f"python measure_memory --framework ladder_fp8_fp8 --model {model} --batch_size {batch_size} --seq_len {seq_len}")
            # reproduce the results for ladder_fp16_mxfp8xmxfp8
            os.system(f"python measure_memory --framework ladder_fp16_mxfp8xmxfp8 --model {model} --batch_size {batch_size} --seq_len {seq_len}")
            # reproduce the results for ladder_fp16_int8xint1
            os.system(f"python measure_memory --framework ladder_fp16_int8xint1 --model {model} --batch_size {batch_size} --seq_len {seq_len}")
    
    os.system(f"python3 plot_memory_usage.py --reproduce")
