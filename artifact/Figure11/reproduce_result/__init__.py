
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

llama_providers = ['BS1 SEQ1', 'BS32 SEQ1', 'BS1 SEQ4096']
llama_times_data = [('PyTorch-Inductor', [2654, 2624, 6878]), ('ONNXRuntime', [2520, 2803, 16078]), ('TensorRT', [2064, 4954, 6342]), ('vLLM', [5008, 4763, 5034]), ('vLLM-W$_{INT4}$A$_{FP16}$', [2060, 2066, 6576]), ('Welder', [2776, 2139, 6790]), ('Bitter', [838, 844, 5192]), ('Bitter-W$_{INT4}$A$_{FP16}$', [836, 842, 5192]), ('Bitter-W$_{NF4}$A$_{FP16}$', [1244, 1250, 5600]), ('Bitter-W$_{FP8}$A$_{FP8}$', [1305, 1299, 5947]), ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [522, 532, 5300]), ('Bitter-W$_{INT1}$A$_{INT8}$', [522, 532, 5300])]

bloom_providers = ['BS1 SEQ1', 'BS32 SEQ1', 'BS1 SEQ4096']
bloom_times_data = [('PyTorch-Inductor', [11503, 12257, 15383]), ('ONNXRuntime', [7540, 7038, 62636]), ('TensorRT', [5566, 5875, 21209]), ('vLLM', [29011, 31764, 29199]), ('vLLM-W$_{INT4}$A$_{FP16}$', [5132, 5152, 20588]), ('Welder', [5130, 5036, 20109]), ('Bitter', [1606, 1626, 18891]), ('Bitter-W$_{INT4}$A$_{FP16}$', [1604, 1624, 19772]), ('Bitter-W$_{NF4}$A$_{FP16}$', [2780, 2800, 18649]), ('Bitter-W$_{FP8}$A$_{FP8}$', [4037, 3944, 20280]), ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [3006, 3032, 17854]), ('Bitter-W$_{INT1}$A$_{INT8}$', [3006, 3032, 17854])]
