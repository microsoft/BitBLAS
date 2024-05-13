
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

llama_providers = ['BS1 SEQ1', 'BS32 SEQ1', 'BS1 SEQ4096']
llama_times_data = [('PyTorch-Inductor', [2660, 2642, 6754]), ('ONNXRuntime', [2619, 2856, 16744]), ('TensorRT', [5269, 5109, 6031]), ('vLLM', [0, 0, 0]), ('vLLM-W$_{INT4}$A$_{FP16}$', [2078, 2111, 6424]), ('Welder', [2098, 2184, 6927]), ('Bitter', [875, 806, 5287]), ('Bitter-W$_{INT4}$A$_{FP16}$', [844, 816, 5406]), ('Bitter-W$_{NF4}$A$_{FP16}$', [1280, 1250, 5621]), ('Bitter-W$_{FP8}$A$_{FP8}$', [1372, 1399, 5587]), ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [507, 527, 5114]), ('Bitter-W$_{INT1}$A$_{INT8}$', [534, 540, 5050])]

bloom_providers = ['BS1 SEQ1', 'BS32 SEQ1', 'BS1 SEQ4096']
bloom_times_data = [('PyTorch-Inductor', [12088, 12072, 0]), ('ONNXRuntime', [7303, 7073, 0]), ('TensorRT', [6023, 5891, 0]), ('vLLM', [0, 0, 0]), ('vLLM-W$_{INT4}$A$_{FP16}$', [5235, 5323, 0]), ('Welder', [4978, 5291, 0]), ('Bitter', [3216, 3275, 0]), ('Bitter-W$_{INT4}$A$_{FP16}$', [3493, 3317, 0]), ('Bitter-W$_{NF4}$A$_{FP16}$', [3820, 3913, 0]), ('Bitter-W$_{FP8}$A$_{FP8}$', [4594, 4519, 0]), ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [2903, 2850, 0]), ('Bitter-W$_{INT1}$A$_{INT8}$', [2931, 2951, 0])]
