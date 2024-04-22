# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
a100_res = {
    "W$_{FP16}$A$_{FP16}$": {
        "cuBLAS": "87%",
        "rocBLAS": "x",
        "AMOS": "38%",
        "TensorIR": "56%",
        "Roller": "70%",
    },
    "W$_{INT8}$A$_{INT8}$": {
        "cuBLAS": "52%",
        "rocBLAS": "x",
        "AMOS": "45%",
        "TensorIR": "x",
        "Roller": "x",
    },
    "W$_{FP8}$A$_{FP8}$": {
        "cuBLAS": "x",
        "rocBLAS": "x",
        "AMOS": "x",
        "TensorIR": "x",
        "Roller": "x",
    },
    "W$_{NF4}$A$_{FP16}$": {
        "cuBLAS": "x",
        "rocBLAS": "x",
        "AMOS": "x",
        "TensorIR": "x",
        "Roller": "x",
    },
}

v100_res = {
    "W$_{FP16}$A$_{FP16}$": {
        "cuBLAS": "78%",
        "rocBLAS": "x",
        "AMOS": "64%",
        "TensorIR": "67%",
        "Roller": "50%",
    },
    "W$_{INT8}$A$_{INT8}$": {
        "cuBLAS": "x",
        "rocBLAS": "x",
        "AMOS": "x",
        "TensorIR": "x",
        "Roller": "x",
    },
    "W$_{FP8}$A$_{FP8}$": {
        "cuBLAS": "x",
        "rocBLAS": "x",
        "AMOS": "x",
        "TensorIR": "x",
        "Roller": "x",
    },
    "W$_{NF4}$A$_{FP16}$": {
        "cuBLAS": "x",
        "rocBLAS": "x",
        "AMOS": "x",
        "TensorIR": "x",
        "Roller": "x",
    },
}

mi250_res = {
    "W$_{FP16}$A$_{FP16}$": {
        "cuBLAS": "x",
        "rocBLAS": "46%",
        "AMOS": "x",
        "TensorIR": "22%",
        "Roller": "29%",
    },
    "W$_{INT8}$A$_{INT8}$": {
        "cuBLAS": "x",
        "rocBLAS": "x",
        "AMOS": "x",
        "TensorIR": "x",
        "Roller": "x",
    },
    "W$_{FP8}$A$_{FP8}$": {
        "cuBLAS": "x",
        "rocBLAS": "75%",
        "AMOS": "x",
        "TensorIR": "x",
        "Roller": "x",
    },
    "W$_{NF4}$A$_{FP16}$": {
        "cuBLAS": "x",
        "rocBLAS": "x",
        "AMOS": "x",
        "TensorIR": "x",
        "Roller": "x",
    },
}
