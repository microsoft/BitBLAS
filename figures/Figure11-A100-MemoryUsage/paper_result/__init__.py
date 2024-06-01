llama_providers = ["BS1 SEQ1", "BS32 SEQ1", "BS1 SEQ4096"]
llama_times_data = [
    ("PyTorch-Inductor", [2660, 2642, 6754]),
    ("ONNXRuntime", [2748, 2780, 16206]),
    ("TensorRT", [5140, 5148, 6260]),
    ("vLLM", [4866, 4868, 4866]),
    ("vLLM-W$_{INT4}$A$_{FP16}$", [1072, 1072, 6400]),
    ("Welder", [2076, 2084, 6626]),
    ("Ladder", [2064, 2070, 6580]),
    ("Ladder-W$_{INT4}$A$_{FP16}$", [840, 846, 5356]),
    ("Ladder-W$_{NF4}$A$_{FP16}$", [852, 853, 5364]),
    ("Ladder-W$_{FP8}$A$_{FP8}$", [1248, 1254, 5764]),
    ("Ladder-W$_{MXFP8}$A$_{MXFP8}$", [1361, 1370, 5877]),
    ("Ladder-W$_{INT1}$A$_{INT8}$", [534, 540, 5050]),
]

bloom_providers = ["BS1 SEQ1", "BS32 SEQ1", "BS1 SEQ4096"]
bloom_times_data = [
    ("PyTorch-Inductor", [12088, 12072, 15674]),
    ("ONNXRuntime", [7356, 6844, 64718]),
    ("TensorRT", [5771, 5783, 21292]),
    ("vLLM", [30512, 30516, 30512]),
    ("vLLM-W$_{INT4}$A$_{FP16}$", [22608, 22612, 22608]),
    ("Welder", [5148, 5160, 20046]),
    ("Ladder", [5136, 5156, 20592]),
    ("Ladder-W$_{INT4}$A$_{FP16}$", [3372, 3392, 18828]),
    ("Ladder-W$_{NF4}$A$_{FP16}$", [3382, 3384, 18844]),
    ("Ladder-W$_{FP8}$A$_{FP8}$", [3960, 3980, 19416]),
    ("Ladder-W$_{MXFP8}$A$_{MXFP8}$", [4654, 4755, 24412]),
    ("Ladder-W$_{INT1}$A$_{INT8}$", [2931, 2951, 18387]),
]
