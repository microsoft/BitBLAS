
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

llama_providers = ['BS1 SEQ1', 'BS32 SEQ1', 'BS1 SEQ4096']
llama_times_data = [('PyTorch-Inductor', [2700, 2624, 6878]), ('ONNXRuntime', [2716, 2803, 16078]), ('TensorRT', [5187, 4954, 6342]), ('vLLM', [5008, 4763, 5034]), ('vLLM-W$_{INT4}$A$_{FP16}$', [1123, 1100, 6128]), ('Welder', [2075, 2121, 6460]), ('Ladder', [2075, 2121, 6460]), ('Ladder-W$_{INT4}$A$_{FP16}$', [879, 817, 5216]), ('Ladder-W$_{NF4}$A$_{FP16}$', [866, 852, 5313]), ('Ladder-W$_{FP8}$A$_{FP8}$', [1306, 1192, 5769]), ('Ladder-W$_{MXFP8}$A$_{MXFP8}$', [1272, 1278, 5788]), ('Ladder-W$_{INT1}$A$_{INT8}$', [532, 540, 4934])]

bloom_providers = ['BS1 SEQ1', 'BS32 SEQ1', 'BS1 SEQ4096']
bloom_times_data = [('PyTorch-Inductor', [11503, 12257, 15383]), ('ONNXRuntime', [7540, 7038, 62636]), ('TensorRT', [5566, 5875, 21209]), ('vLLM', [29011, 31764, 29199]), ('vLLM-W$_{INT4}$A$_{FP16}$', [22327, 21910, 21931]), ('Welder', [5169, 5117, 20977]), ('Ladder', [5169, 5117, 20977]), ('Ladder-W$_{INT4}$A$_{FP16}$', [3277, 3391, 18891]), ('Ladder-W$_{NF4}$A$_{FP16}$', [3374, 3374, 19772]), ('Ladder-W$_{FP8}$A$_{FP8}$', [4052, 3846, 18649]), ('Ladder-W$_{MXFP8}$A$_{MXFP8}$', [2860, 2884, 21122]), ('Ladder-W$_{INT1}$A$_{INT8}$', [726, 752, 19758])]
