
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

llama_providers = ['BS1 SEQ1', 'BS32 SEQ1', 'BS1 SEQ4096']
llama_times_data = [('PyTorch-Inductor', [2654, 2662, 19284]), ('ONNXRuntime', [2520, 2520, 15896]), ('TensorRT', [2064, 2072, 8700]), ('vLLM', [5008, 4763, 5034]), ('vLLM-W$_{INT4}$A$_{FP16}$', [1123, 1100, 6128]), ('Welder', [2060, 2066, 6576]), ('Bitter', [2075, 2121, 6460]), ('Bitter-W$_{INT4}$A$_{FP16}$', [838, 844, 5192]), ('Bitter-W$_{NF4}$A$_{FP16}$', [836, 842, 5192]), ('Bitter-W$_{FP8}$A$_{FP8}$', [1244, 1250, 5600]), ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [1305, 1299, 5947]), ('Bitter-W$_{INT1}$A$_{INT8}$', [522, 532, 5300])]

bloom_providers = ['BS1 SEQ1', 'BS32 SEQ1', 'BS1 SEQ4096']
bloom_times_data = [('PyTorch-Inductor', [12084, 12094, 44344]), ('ONNXRuntime', [7128, 6616, 66008]), ('TensorRT', [5136, 426, 6826]), ('vLLM', [29011, 31764, 29199]), ('vLLM-W$_{INT4}$A$_{FP16}$', [22327, 21910, 21931]), ('Welder', [5132, 5152, 21232]), ('Bitter', [5169, 5117, 20977]), ('Bitter-W$_{INT4}$A$_{FP16}$', [1606, 1626, 17802]), ('Bitter-W$_{NF4}$A$_{FP16}$', [1604, 1624, 17928]), ('Bitter-W$_{FP8}$A$_{FP8}$', [2780, 2800, 18460]), ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [4037, 3944, 20280]), ('Bitter-W$_{INT1}$A$_{INT8}$', [3006, 3032, 17854])]
