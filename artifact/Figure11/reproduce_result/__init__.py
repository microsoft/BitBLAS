
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

llama_providers = ['BS1 SEQ1', 'BS32 SEQ1', 'BS1 SEQ4096']
llama_times_data = [('PyTorch-Inductor', [2700, 2624, 6878]), ('ONNXRuntime', [2716, 2803, 16078]), ('TensorRT', [5187, 4954, 6342]), ('vLLM', [5008, 4763, 5034]), ('vLLM-W$_{INT4}$A$_{FP16}$', [1123, 1100, 6128]), ('Welder', [2075, 2121, 6460]), ('Bitter', [2075, 2121, 6460]), ('Bitter-W$_{INT4}$A$_{FP16}$', [879, 817, 5216]), ('Bitter-W$_{NF4}$A$_{FP16}$', [866, 852, 5313]), ('Bitter-W$_{FP8}$A$_{FP8}$', [1306, 1192, 5769]), ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [1305, 1299, 5947]), ('Bitter-W$_{INT1}$A$_{INT8}$', [522, 532, 5300])]

bloom_providers = ['BS1 SEQ1', 'BS32 SEQ1', 'BS1 SEQ4096']
bloom_times_data = [('PyTorch-Inductor', [11503, 12257, 15383]), ('ONNXRuntime', [7540, 7038, 62636]), ('TensorRT', [5566, 5875, 21209]), ('vLLM', [29011, 31764, 29199]), ('vLLM-W$_{INT4}$A$_{FP16}$', [22327, 21910, 21931]), ('Welder', [5169, 5117, 20977]), ('Bitter', [5169, 5117, 20977]), ('Bitter-W$_{INT4}$A$_{FP16}$', [3277, 3391, 18891]), ('Bitter-W$_{NF4}$A$_{FP16}$', [3374, 3374, 19772]), ('Bitter-W$_{FP8}$A$_{FP8}$', [4052, 3846, 18649]), ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [4037, 3944, 20280]), ('Bitter-W$_{INT1}$A$_{INT8}$', [3006, 3032, 17854])]
