
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

b1s1_llama2_providers = ['W$_{FP16}$A$_{FP16}$', 'W$_{INT4}$A$_{FP16}$', 'W$_{MXFP8}$A$_{MXFP8}$', 'W$_{INT1}$A$_{INT8}$']
b1s1_llama2_times_data = [('Welder-Roller', [2.173171, 0, 0, 0]), ('+Transform', [1.9759, 0.5916, 2.1099, 0.3294]), ('+PTX', [1.9767, 0.5587, 2.1068, 0.1866]), ('+Holistic Schedule', [1.9758, 0.5593, 1.217, 0.1863])]

b1s4096_llama2_providers = ['W$_{FP16}$A$_{FP16}$', 'W$_{INT4}$A$_{FP16}$', 'W$_{MXFP8}$A$_{MXFP8}$', 'W$_{INT1}$A$_{INT8}$']
b1s4096_llama2_times_data = [('Welder-Roller', [254.423248, 0, 0, 0]), ('+Transform', [103.3931, 104.4866, 2252.1765, 649.2171]), ('+PTX', [102.7355, 104.368, 2249.7719, 650.3145]), ('+Holistic Schedule', [102.8257, 104.3971, 676.1094, 649.9194])]
