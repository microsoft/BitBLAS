
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
b1s1_providers = ['End2End LLAMA']
b1s1_times_data = [('Bitter', [1.0305]), ('Bitter-W$_{INT8}$A$_{FP16}$', [0.5924999999999999]), ('Bitter-W$_{INT4}$A$_{FP16}$', [0.32849999999999996]), ('Bitter-W$_{INT2}$A$_{FP16}$', [0.2895]), ('Bitter-W$_{INT1}$A$_{FP16}$', [0.2815]), ('Bitter-W$_{INT8}$A$_{INT8}$', [0.6004999999999999]), ('Bitter-W$_{INT4}$A$_{INT8}$', [0.3175]), ('Bitter-W$_{INT2}$A$_{INT8}$', [0.19149999999999998]), ('Bitter-W$_{INT1}$A$_{INT8}$', [0.15849999999999997]), ('Bitter-W$_{INT4}$A$_{INT4}$', [0.3145]), ('Bitter-W$_{INT2}$A$_{INT4}$', [0.19249999999999998]), ('Bitter-W$_{INT1}$A$_{INT4}$', [0.15849999999999997])]

b1s4096_providers = ['End2End LLAMA']
b1s4096_times_data = [('Bitter', [33.7857]), ('Bitter-W$_{INT8}$A$_{FP16}$', [34.7037]), ('Bitter-W$_{INT4}$A$_{FP16}$', [32.58669999999999]), ('Bitter-W$_{INT2}$A$_{FP16}$', [33.5087]), ('Bitter-W$_{INT1}$A$_{FP16}$', [33.33369999999999]), ('Bitter-W$_{INT8}$A$_{INT8}$', [22.998699999999996]), ('Bitter-W$_{INT4}$A$_{INT8}$', [22.5307]), ('Bitter-W$_{INT2}$A$_{INT8}$', [22.005699999999997]), ('Bitter-W$_{INT1}$A$_{INT8}$', [22.2537]), ('Bitter-W$_{INT4}$A$_{INT4}$', [14.267433072996138]), ('Bitter-W$_{INT2}$A$_{INT4}$', [14.268457080793379]), ('Bitter-W$_{INT1}$A$_{INT4}$', [14.270300294828413])]

b1s1_matmul_providers = ['M0', 'M1', 'M2', 'M3']
b1s1_matmul_times_data = [('Bitter', [0.01, 0.081, 0.267, 0.271]), ('Bitter-W$_{INT8}$A$_{FP16}$', [0.01, 0.045, 0.143, 0.153]), ('Bitter-W$_{INT4}$A$_{FP16}$', [0.007, 0.024, 0.072, 0.079]), ('Bitter-W$_{INT2}$A$_{FP16}$', [0.007, 0.021, 0.062, 0.066]), ('Bitter-W$_{INT1}$A$_{FP16}$', [0.007, 0.021, 0.06, 0.062]), ('Bitter-W$_{INT8}$A$_{INT8}$', [0.01, 0.048, 0.148, 0.145]), ('Bitter-W$_{INT4}$A$_{INT8}$', [0.006, 0.021, 0.073, 0.074]), ('Bitter-W$_{INT2}$A$_{INT8}$', [0.006, 0.011, 0.038, 0.038]), ('Bitter-W$_{INT1}$A$_{INT8}$', [0.005, 0.011, 0.028, 0.027]), ('Bitter-W$_{INT4}$A$_{INT4}$', [0.006, 0.02, 0.073, 0.073]), ('Bitter-W$_{INT2}$A$_{INT4}$', [0.006, 0.011, 0.038, 0.039]), ('Bitter-W$_{INT1}$A$_{INT4}$', [0.005, 0.011, 0.028, 0.027])]

b1s4096_matmul_providers = ['M0', 'M1', 'M2', 'M3']
b1s4096_matmul_times_data = [('Bitter', [0.34, 2.104, 7.13, 8.457]), ('Bitter-W$_{INT8}$A$_{FP16}$', [0.35, 2.184, 7.785, 7.885]), ('Bitter-W$_{INT4}$A$_{FP16}$', [0.342, 2.07, 7.12, 7.342]), ('Bitter-W$_{INT2}$A$_{FP16}$', [0.343, 2.172, 7.396, 7.506]), ('Bitter-W$_{INT1}$A$_{FP16}$', [0.346, 2.165, 7.319, 7.493]), ('Bitter-W$_{INT8}$A$_{INT8}$', [0.204, 1.329, 4.544, 4.664]), ('Bitter-W$_{INT4}$A$_{INT8}$', [0.198, 1.297, 4.471, 4.418]), ('Bitter-W$_{INT2}$A$_{INT8}$', [0.194, 1.253, 4.331, 4.269]), ('Bitter-W$_{INT1}$A$_{INT8}$', [0.198, 1.278, 4.394, 4.333]), ('Bitter-W$_{INT4}$A$_{INT4}$', [0.11120639741420746, 0.6909952163696289, 2.0856833457946777, 2.3109631538391113]), ('Bitter-W$_{INT2}$A$_{INT4}$', [0.1114111989736557, 0.6901760101318359, 2.0865025520324707, 2.311577558517456]), ('Bitter-W$_{INT1}$A$_{INT4}$', [0.1114111989736557, 0.6901760101318359, 2.0873217582702637, 2.3117823600769043])]

