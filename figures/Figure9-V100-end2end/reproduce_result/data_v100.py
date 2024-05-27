
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
llama2_providers = ['BS1 SEQ1', 'BS32 SEQ1', 'BS1 SEQ4096']
llama2_times_data = [('Pytorch Inductor', [2.723541259765625, 2.856614589691162, 96.69058322906494]), ('ONNXRuntime', [3.5404, 2.8377, 146.3234]), ('TensorRT', [2.04167, 2.19508, 118.927]), ('Welder', [2.150358, 2.517434, 96.209755]), ('vLLM', [2.562878131866455, 2.649393081665039, 81.92222118377686]), ('vLLM-W$_{INT4}$A$_{FP16}$', [0, 0, 0]), ('Bitter', [1.982, 2.447, 98.8685]), ('Bitter-W$_{INT4}$A$_{FP16}$', [0.5596, 2.562, 104.8577]), ('Bitter-W$_{NF4}$A$_{FP16}$', [0.6536, 2.37, 111.911]), ('Bitter-W$_{FP8}$A$_{FP8}$', [1.0306, 2.6374, 110.421]), ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [1.2601, 6.9955, 675.9145]), ('Bitter-W$_{INT1}$A$_{INT8}$', [0.1866, 9.5937, 654.0658])]

bloom_providers = ['BS1 SEQ1', 'BS32 SEQ1', 'BS1 SEQ4096']
bloom_times_data = [('Pytorch Inductor', [8.05161714553833, 8.260955810546875, 0]), ('ONNXRuntime', [6.6034, 6.9575, 0]), ('TensorRT', [5.84538, 6.09689, 0]), ('Welder', [5.820163, 6.572861, 0]), ('vLLM', [0, 0, 0]), ('vLLM-W$_{INT4}$A$_{FP16}$', [0, 0, 0]), ('Bitter', [5.5893, 6.1652, 0]), ('Bitter-W$_{INT4}$A$_{FP16}$', [1.4762, 4.1783, 0]), ('Bitter-W$_{NF4}$A$_{FP16}$', [1.7943, 4.7051, 0]), ('Bitter-W$_{FP8}$A$_{FP8}$', [2.8474, 5.1218, 0]), ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [3.5547, 11.7867, 0]), ('Bitter-W$_{INT1}$A$_{INT8}$', [0.4374, 21.8239, 0])]

resnet_providers = ['BS1', 'BS128']
resnet_times_data = [('Pytorch Inductor', [6.372692584991455, 25.560548305511475]), ('ONNXRuntime', [5.3826, 51.4725]), ('TensorRT', [1.67877, 22.3579]), ('AMOS', [19.81741, 144.10797]), ('TensorIR', [1.609463602, 26.15646305555555]), ('Welder', [2.33711, 34.261635]), ('Bitter', [1.4659, 23.5883]), ('Bitter-W$_{FP8}$A$_{FP8}$', [23.5836, 23.5861]), ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [2.0269, 97.9593]), ('Bitter-W$_{INT1}$A$_{INT4}$', [1.4656, 30.5769])]

shufflenet_providers = ['BS1', 'BS128']
shufflenet_times_data = [('Pytorch Inductor', [6.222949028015137, 6.273815631866455]), ('ONNXRuntime', [2.5678, 12.8396]), ('TensorRT', [1.30377, 7.81188]), ('AMOS', [3.954496, 35.31771]), ('TensorIR', [0.4112031207702374, 7.081917907407408]), ('Welder', [0.393066, 6.508262]), ('Bitter', [0.342, 5.4303]), ('Bitter-W$_{FP8}$A$_{FP8}$', [0.341, 5.4876])]

conformer_providers = ['BS1', 'BS128']
conformer_times_data = [('Pytorch Inductor', [13.664772510528564, 158.91096353530884]), ('ONNXRuntime', [8.916, 372.7684]), ('TensorRT', [4.56492, 157.126]), ('AMOS', [0, 0]), ('TensorIR', [0, 0]), ('Welder', [3.831085, 180.668472]), ('Bitter', [4.4909, 247.8213]), ('Bitter-W$_{INT4}$A$_{INT8}$', [4.5012, 248.2843]), ('Bitter-W$_{INT4}$A$_{INT4}$', [3.4774, 248.2928])]

vit_providers = ['BS1', 'BS128']
vit_times_data = [('Pytorch Inductor', [5.221261978149414, 10.503861904144287]), ('ONNXRuntime', [3.6079, 22.6451]), ('TensorRT', [1.10584, 9.24]), ('AMOS', [0, 0]), ('TensorIR', [1.179153433, 14.82752]), ('Welder', [1.751808, 9.831462]), ('Bitter', [1.0901, 12.0202]), ('Bitter-W$_{FP8}$A$_{FP8}$', [1.0871, 12.021]), ('Bitter-W$_{INT4}$A$_{INT4}$', [1.0864, 12.0413])]
