
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

b1s1_llama2_providers = ['W$_{FP16}$A$_{FP16}$', 'W$_{INT4}$A$_{FP16}$', 'W$_{MXFP8}$A$_{MXFP8}$', 'W$_{INT1}$A$_{INT8}$']
b1s1_llama2_times_data = [('Welder-Roller', [1.878592, 0, 0, 0]), ('+Transform', [1.0301, 0.4442, 1.5962, 0.2713]), ('+PTX', [1.029, 0.3402, 1.5956, 0.1574]), ('+Holistic Schedule', [1.0306, 0.3403, 0.7758, 0.1596])]

b1s4096_llama2_providers = ['W$_{FP16}$A$_{FP16}$', 'W$_{INT4}$A$_{FP16}$', 'W$_{MXFP8}$A$_{MXFP8}$', 'W$_{INT1}$A$_{INT8}$']
b1s4096_llama2_times_data = [('Welder-Roller', [123.907089, 0, 0, 0]), ('+Transform', [42.1378, 36.979, 106.1893, 26.1352]), ('+PTX', [35.1925, 34.4095, 98.9012, 25.1437]), ('+Holistic Schedule', [35.2558, 34.418, 57.5565, 25.2145])]
