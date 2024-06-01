# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
llama2_providers = ['BS1 SEQ1', 'BS32 SEQ1', 'BS1 SEQ4096']
llama2_times_data = [
    ('PyTorch-Inductor', [3.162500858, 3.11050415, 100.0415564]),
    ('ONNXRuntime', [3.6252, 4.4371, 144.3875]),
    ('TensorRT', [2.11548, 2.35596, 121.227]),
    ('Welder', [2.144288, 2.480128, 94.676994]),
    ('vLLM', [2.348845005, 2.505731583, 90.14932156]),
    ('vLLM-W$_{INT4}$A$_{FP16}$', [0, 0, 0]),
    ('Ladder', [1.982, 2.447, 98.8685]),
    ('Ladder-W$_{INT4}$A$_{FP16}$', [0.59439808, 2.158567894, 97.91209607]),
    ('Ladder-W$_{NF4}$A$_{FP16}$', [0.830255887, 2.324443739, 107.4976185]),
    ('Ladder-W$_{FP8}$A$_{FP8}$', [1.089282821, 3.466108892, 228.1671086]),
    ('Ladder-W$_{MXFP8}$A$_{MXFP8}$', [1.119883253, 10.80787646, 699.8160501]),
    ('Ladder-W$_{INT1}$A$_{INT8}$', [0.207486801, 5.313923571, 559.9500032]),
]

bloom_providers = ['BS1 SEQ1', 'BS32 SEQ1', 'BS1 SEQ4096']
bloom_times_data = [
    ('PyTorch-Inductor', [8.333725929, 8.624098301, 0]),
    ('ONNXRuntime', [0, 0, 0]),
    ('TensorRT', [5.7852, 6.15668, 0]),
    ('Welder', [5.849088, 6.606816, 0]),
    ('vLLM', [0, 0, 0]),
    ('vLLM-W$_{INT4}$A$_{FP16}$', [0, 0, 0]),
    ('Ladder', [5.5886, 6.1652, 0]),
    ('Ladder-W$_{INT4}$A$_{FP16}$', [1.53901594, 3.901990775, 0]),
    ('Ladder-W$_{NF4}$A$_{FP16}$', [1.69907736, 5.181669567, 0]),
    ('Ladder-W$_{FP8}$A$_{FP8}$', [2.83026982, 5.502423499, 0]),
    ('Ladder-W$_{MXFP8}$A$_{MXFP8}$', [3.310397363, 3.310397363, 0]),
    ('Ladder-W$_{INT1}$A$_{INT8}$', [0.43558272, 6.1652, 0]),
]

resnet_providers = ['BS1', 'BS128']
resnet_times_data = [
    ('PyTorch-Inductor', [4.82632637, 25.97796679]),
    ('ONNXRuntime', [3.7384, 97.6342]),
    ('TensorRT', [1.85937, 18.8322]),
    ('AMOS', [19.77248, 144.9735]),
    ('TensorIR', [1.609463602, 26.15646306]),
    ('Welder', [2.656288, 42.615776]),
    ('Ladder', [1.4638, 21.0237]),
    ('Ladder_W$_{FP8}$A$_{FP8}$', [1.4638, 21.0237]),
    ('Ladder-W$_{MXFP8}$A$_{MXFP8}$', [2.48846, 94.60665]),
    ('Ladder_W$_{INT1}$A$_{INT4}$', [1.4638, 21.0237]),
]

shufflenet_providers = ['BS1', 'BS128']
shufflenet_times_data = [
    ('PyTorch-Inductor', [6.236689091, 6.174676418]),
    ('ONNXRuntime', [2.8359, 14.0666]),
    ('TensorRT', [1.33392, 5.26163]),
    ('AMOS', [3.954496, 35.31771]),
    ('TensorIR', [0.411856816, 7.081917907]),
    ('Welder', [0.40752, 6.562784]),
    ('Ladder', [0.4042, 5.2663]),
    ('Ladder_W$_{FP8}$A$_{FP8}$', [0.4042, 5.2663]),
]

conformer_providers = ['BS1', 'BS128']
conformer_times_data = [
    ('PyTorch-Inductor', [13.62011671, 168.6849737]),
    ('ONNXRuntime', [10.1335, 408.1039]),
    ('TensorRT', [3.53897, 162.431]),
    ('AMOS', [0, 0]),
    ('TensorIR', [0, 0]),
    ('Welder', [4.04784, 172.965851]),
    ('Ladder', [3.5447, 193.1576]),
    ('Ladder-W$_{INT4}$A$_{INT8}$', [3.5447, 193.1576]),
    ('Ladder-W$_{INT4}$A$_{INT4}$', [3.5447, 193.1576]),
]

vit_providers = ['BS1', 'BS128']
vit_times_data = [
    ('PyTorch-Inductor', [5.180325508, 8.272943497]),
    ('ONNXRuntime', [3.5002, 23.8669]),
    ('TensorRT', [1.17185, 8.76167]),
    ('AMOS', [0, 0]),
    ('TensorIR', [1.179153433, 14.82752]),
    ('Welder', [1.31072, 8.150656]),
    ('Ladder', [1.32948, 9.2983]),
    ('Ladder-W$_{FP8}$A$_{FP8}$', [1.32948, 9.2983]),
    ('Ladder-W$_{INT4}$A$_{INT4}$', [1.32948, 9.2983]),
]
