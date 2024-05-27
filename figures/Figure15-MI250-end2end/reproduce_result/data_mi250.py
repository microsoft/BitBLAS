# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
llama2_providers = ["BS1 SEQ1", "BS32 SEQ1", "BS1 SEQ4096"]
llama2_times_data = [
    ("PyTorch-Inductor", [2.967426777, 3.527953625, 90.68135023]),
    ("ONNXRuntime", [2.7045, 3.6368, 105.8585]),
    ("Welder", [2.777031, 3.367959, 225.548996]),
    ("Bitter", [1.2937, 4.4948, 64.0782]),
    ("Bitter-W$_{INT4}$A$_{FP16}$", [0.734604753, 2.569160028, 59.77038117]),
    ("Bitter-W$_{NF4}$A$_{FP16}$", [0.785323892, 5.180914162, 59.77038117]),
    ("Bitter-W$_{FP8}$A$_{FP8}$", [0.739654965, 3.422708882, 57.25802382]),
    ("Bitter-W$_{MXFP8}$A$_{MXFP8}$", [1.352413267, 4.323679669, 65.57095309]),
    ("Bitter-W$_{INT1}$A$_{INT8}$", [0.410433279, 2.746247275, 62.62647136]),
]

bloom_providers = ["BS1 SEQ1", "BS32 SEQ1", "BS1 SEQ4096"]
bloom_times_data = [
    ("PyTorch-Inductor", [8.031938076, 9.028329182, 252.4367213]),
    ("ONNXRuntime", [7.1162, 8.1783, 435.6429]),
    ("Welder", [7.842051, 8.312899, 651.153931]),
    ("Bitter", [3.5806, 5.9158, 187.6232]),
    ("Bitter-W$_{INT4}$A$_{FP16}$", [1.754778496, 3.877172256, 175.2557665]),
    ("Bitter-W$_{NF4}$A$_{FP16}$", [2.011513657, 8.142474497, 187.6232]),
    ("Bitter-W$_{FP8}$A$_{FP8}$", [1.900734506, 3.946730787, 166.980928]),
    ("Bitter-W$_{MXFP8}$A$_{MXFP8}$", [4.193349517, 6.322527791, 208.7369409]),
    ("Bitter-W$_{INT1}$A$_{INT8}$", [0.980329967, 3.685529078, 176.9952694]),
]

resnet_providers = ["BS1", "BS128"]
resnet_times_data = [
    ("PyTorch-Inductor", [2.792823315, 24.40225124]),
    ("ONNXRuntime", [1.7147, 37.8371]),
    ("TensorIR", [9.2, 57.816]),
    ("Welder", [2.952049, 27.944721]),
    ("Bitter", [1.906, 18.55523986]),
    ("Bitter-W$_{FP8}$A$_{FP8}$", [1.906, 18.55523986]),
    ("Bitter-W$_{MXFP8}$A$_{MXFP8}$", [1.965079854, 19.30361308]),
    ("Bitter-W$_{INT1}$A$_{INT4}$", [1.938472377, 19.43600336]),
]

shufflenet_providers = ["BS1", "BS128"]
shufflenet_times_data = [
    ("PyTorch-Inductor", [3.478677273, 6.883788109]),
    ("ONNXRuntime", [1.5841, 11.3498]),
    ("TensorIR", [3.67, 10.891]),
    ("Welder", [6.145209, 27.944721]),
    ("Bitter", [0.4348, 4.0721]),
    ("Bitter-W$_{FP8}$A$_{FP8}$", [0.436497486, 4.065415957]),
]

conformer_providers = ["BS1", "BS128"]
conformer_times_data = [
    ("PyTorch-Inductor", [9.66853857, 182.6023865]),
    ("ONNXRuntime", [8.458, 392.1407]),
    ("Welder", [5.959653, 250.837875]),
    ("TensorIR", [0, 0]),
    ("Bitter", [4.1736, 139.0793]),
    ("Bitter-W$_{INT4}$A$_{INT8}$", [4.1736, 138.283038]),
    ("Bitter-W$_{INT4}$A$_{INT4}$", [4.1736, 138.283038]),
]

vit_providers = ["BS1", "BS128"]
vit_times_data = [
    ("PyTorch-Inductor", [4.386992455, 7.317135334]),
    ("ONNXRuntime", [3.6023, 18.7391]),
    ("TensorIR", [1.823, 29.8]),
    ("Welder", [2.166076, 14.007601]),
    ("Bitter", [1.749873785, 8.0876]),
    ("Bitter-W$_{FP8}$A$_{FP8}$", [1.79692474, 7.97369563]),
    ("Bitter-W$_{INT4}$A$_{INT4}$", [1.530730393, 7.172765242]),
]
