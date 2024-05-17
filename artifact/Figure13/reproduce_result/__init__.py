
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

b1s1_llama2_providers = ['W$_{FP16}$A$_{FP16}$', 'W$_{INT4}$A$_{FP16}$', 'W$_{MXFP8}$A$_{MXFP8}$', 'W$_{INT1}$A$_{INT8}$']
b1s1_llama2_times_data = [('Welder-Roller', [1.878592, 0, 0, 0]), ('+Transform', [1.0316, 0.4433, 1.5967, 0.2699]), ('+PTX', [1.0302, 0.339, 1.5962, 0.1579]), ('+Holistic Schedule', [1.0306, 0.3402, 0.7766, 0.1591])]

b1s4096_llama2_providers = ['W$_{FP16}$A$_{FP16}$', 'W$_{INT4}$A$_{FP16}$', 'W$_{MXFP8}$A$_{MXFP8}$', 'W$_{INT1}$A$_{INT8}$']
b1s4096_llama2_times_data = [('Welder-Roller', [123.907089, 0, 0, 0]), ('+Transform', [42.2107, 40.1831, 106.1437, 28.1074]), ('+PTX', [35.1826, 34.3856, 98.8915, 25.1482]), ('+Holistic Schedule', [35.1793, 34.3842, 35.1782, 25.1403])]
