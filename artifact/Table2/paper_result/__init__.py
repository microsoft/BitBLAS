# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
compilation_cost = {
    "ResNet(1)": {"AMOS": "3852", "TensorIR": "156", "Welder": "11", "LADDER": "31"},
    "ResNet(128)": {"AMOS": "3328", "TensorIR": "128", "Welder": "13", "LADDER": "17"},
    "ShuffleNet(1)": {
        "AMOS": "2191",
        "TensorIR": "836",
        "Welder": "18",
        "LADDER": "44",
    },
    "ShuffleNet(128)": {
        "AMOS": "3121",
        "TensorIR": "400",
        "Welder": "12",
        "LADDER": "29",
    },
}
