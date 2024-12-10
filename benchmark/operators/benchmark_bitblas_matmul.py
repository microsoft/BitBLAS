# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas.utils.target_detector import auto_detect_nvidia_target
from bitblas import Matmul, MatmulConfig
import argparse
import json

# Initialize the parser
parser = argparse.ArgumentParser(
    description="Benchmark BitBLAS int4 on a specific target."
)

# Add arguments to the parser
parser.add_argument(
    "--target",
    type=str,
    default=auto_detect_nvidia_target(),
    help="Specify the target device for benchmarking."
)
parser.add_argument(
    "--group_size",
    type=int,
    default=None,
    help="Group size for grouped quantization."
)
parser.add_argument(
    "--A_dtype",
    type=str,
    default="float16",
    choices=["float16", "int8"], 
    help="Data type of activation A."
)
parser.add_argument(
    "--W_dtype",
    type=str,
    default="int4",
    choices=["float16", "int32", "int8", "int4", "int2", "int1", "nf4", "fp4_e2m1"],
    help="Data type of weight W."
)
parser.add_argument(
    "--accum_dtype",
    type=str,
    default="float16",
    choices=["float16", "int32"],  # Assuming these are the valid choices
    help="Data type for accumulation."
)
parser.add_argument(
    "--out_dtype",
    type=str,
    default="float16",
    choices=["float16", "float32", "int32", "int8"],  # Assuming these are the valid choices
    help="Data type for output."
)
parser.add_argument(
    "--layout",
    type=str,
    default="nt",
    choices=["nt", "nn"],  # Assuming these are the valid choices
    help="Matrix layout, 'nt' for non-transpose A and transpose W."
)
parser.add_argument(
    "--with_bias",
    action="store_true",
    help="Include bias in the benchmark."
)
parser.add_argument(
    "--with_scaling",
    action="store_true",
    help="Include scaling factor in the quantization."
)
parser.add_argument(
    "--with_zeros",
    action="store_true",
    help="Include zeros in the quantization."
)
parser.add_argument(
    "--zeros_mode",
    type=str,
    default=None,
    choices=["original", "rescale", "quantized"],  # Replace with actual modes if applicable
    help="Specify the mode for calculating zeros."
)

parser.add_argument(
    "--backend",
    type=str,
    default="tl",
    choices=["tir", "tl"],  # Replace with actual modes if applicable
    help="Specify the mode for calculating zeros."
)

default_test_shapes = json.dumps([
    ["MatmulConfig", "Matmul", [1, 16384, 16384, "float16", "int4", "float16", "float16", "nt", False, None, False, False, None]]
])

parser.add_argument(
    "--test_shapes",
    type=str,
    default=default_test_shapes,
    help="JSON string defining test shapes. Example format: '[[\"MatmulConfig\", \"Matmul\", [1,16384,16384,\"float16\",\"int4\",\"float16\",\"float16\",\"nt\",false,null,false,false,null]]]'"
)

# Parse the arguments
args = parser.parse_args()

# Assign arguments to variables
target = args.target
group_size = args.group_size
A_dtype = args.A_dtype
W_dtype = args.W_dtype
accum_dtype = args.accum_dtype
out_dtype = args.out_dtype
layout = args.layout
with_bias = args.with_bias
with_scaling = args.with_scaling
with_zeros = args.with_zeros
zeros_mode = args.zeros_mode
backend = args.backend

parsed_test_shapes = json.loads(args.test_shapes)
name_to_class = {
    "MatmulConfig": MatmulConfig,
    "Matmul": Matmul
}

test_shapes = []
for item in parsed_test_shapes:
    config_class_name, operator_class_name, input_args = item
    config_class = name_to_class[config_class_name]
    operator_class = name_to_class[operator_class_name]
    test_shapes.append((config_class, operator_class, tuple(input_args)))

benchmark_sets = []
benchmark_sets.extend(test_shapes)

# fmt:on

benchmark_results = {}
for config, operator, input_args in benchmark_sets:
    config = config(*input_args)
    op_inst = operator(config, target=target, enable_tuning=True, backend=backend)
    kernel_latency = op_inst.profile_latency()
    if op_inst.input_transform is not None:
        kernel_latency += op_inst.ladder_permutate_a.profile_latency()

    print("Time cost is: {:.3f} ms".format(kernel_latency))

    profile_config = {
        f"{operator.__name__}-{'-'.join([str(i) for i in input_args])}": {
            "BitBLAS_top20_latency": kernel_latency,
        }
    }

    benchmark_results.update(profile_config)

# Define headers for the table
headers = [
    "PrimFunc",
    "Input Arguments",
    "BitBLAS Top20 Latency",
]

col_widths = [0, 0, 0]
for config, values in benchmark_results.items():
    args = config.split("-")
    func_name = args[0]
    input_args = "-".join(args[1:])
    col_widths[0] = max((max(len(str(headers[0])), len(func_name)) + 2), col_widths[0])
    col_widths[1] = max((max(len(str(headers[1])), len(input_args)) + 2, col_widths[1]))
    col_widths[2] = max(max(len(str(headers[2])), len(f"{values['BitBLAS_top20_latency']:.3f} ms")) + 2, col_widths[2])
    break

for i, header in enumerate(headers):
    headers[i] = header.ljust(col_widths[i])

print("".join(headers))

print("-" * sum(col_widths))

for config, values in benchmark_results.items():
    args = config.split("-")
    func_name = args[0]
    input_args = "-".join(args[1:])
    row = [
        func_name,
        input_args,
        f"{values['BitBLAS_top20_latency']:.3f} ms",
    ]
    print("".join([str(i).ljust(col_widths[j]) for j, i in enumerate(row)]) + "\n")
