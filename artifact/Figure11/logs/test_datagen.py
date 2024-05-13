import json

llama_providers = ["BS1 SEQ1", "BS32 SEQ1", "BS1 SEQ4096"]
llama_times_data = [
    ('PyTorch-Inductor', [2551, 2668, 6584]),
    ('ONNXRuntime', [2753, 2896, 15994]),
    ('TensorRT', [5199, 5241, 6437]),
    ('vLLM', [5013, 4928, 4850]),
    ('vLLM-W$_{INT4}$A$_{FP16}$', [1088, 1088, 6180]),
    ('Welder', [2135, 2159, 6900]),
    ('Bitter', [2087, 2067, 6844]),
    ('Bitter-W$_{INT4}$A$_{FP16}$', [840, 842, 5500]),
    ('Bitter-W$_{NF4}$A$_{FP16}$', [861, 847, 5418]),
    ('Bitter-W$_{FP8}$A$_{FP8}$', [1259, 1262, 5821]),
    ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [1399, 1315, 5609]),
    ('Bitter-W$_{INT1}$A$_{INT8}$', [529, 547, 5060]),
]

bloom_providers = ["BS1 SEQ1", "BS32 SEQ1", "BS1 SEQ4096"]
bloom_times_data = [
    ('PyTorch-Inductor', [12312, 12206, 15288]),
    ('ONNXRuntime', [7371, 6739, 67442]),
    ('TensorRT', [5799, 5864, 22172]),
    ('vLLM', [29095, 30514, 29659]),
    ('vLLM-W$_{INT4}$A$_{FP16}$', [23052, 22029, 21488]),
    ('Welder', [5228, 5090, 19250]),
    ('Bitter', [5219, 4928, 19958]),
    ('Bitter-W$_{INT4}$A$_{FP16}$', [3483, 3486, 19068]),
    ('Bitter-W$_{NF4}$A$_{FP16}$', [3266, 3479, 19645]),
    ('Bitter-W$_{FP8}$A$_{FP8}$', [3832, 4130, 18597]),
    ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [4515, 4833, 24467]),
    ('Bitter-W$_{INT1}$A$_{INT8}$', [2947, 2966, 18060]),
]

import json
import os

# 创建存储文件的目录
directory = "."
if not os.path.exists(directory):
    os.makedirs(directory)


name_mapping = {
    "PyTorch-Inductor": "pytorch_inductor",
    "ONNXRuntime": "onnxruntime",
    "TensorRT": "tensorrt",
    "vLLM": "vllm",
    "vLLM-W$_{INT4}$A$_{FP16}$": "vllm_int4_fp16",
    "Welder": "welder",
    "Bitter": "ladder",
    "Bitter-W$_{INT4}$A$_{FP16}$": "ladder_fp16_int4",
    "Bitter-W$_{NF4}$A$_{FP16}$": "ladder_fp16_nf4",
    "Bitter-W$_{FP8}$A$_{FP8}$": "ladder_fp8_fp8",
    "Bitter-W$_{MXFP8}$A$_{MXFP8}$": "ladder_fp16_mxfp8xmxfp8",
    "Bitter-W$_{INT1}$A$_{INT8}$": "ladder_fp16_int8xint1"
}

def create_json_files(model_name, model_providers, model_times_data):
    for entry in model_times_data:
        model, times = entry
        # 使用映射表查找新模型名称
        if model in name_mapping:
            new_model = name_mapping[model]
            # 为每个配置创建JSON文件
            for index, time in enumerate(times):
                if time != 0:  # 只保存非零时间数据
                    batch_size, seq_len = model_providers[index].replace("BS", "").replace("SEQ", "_").split(" ")
                    seq_len = seq_len.replace("_", "")
                    file_name = f"{model_name}_{new_model}_b{batch_size}_s{seq_len}_data.json"
                    file_path = os.path.join(directory, file_name)
                    data = {f"{new_model}_{batch_size}_{seq_len}": time}
                    with open(file_path, 'w') as json_file:
                        json.dump(data, json_file, indent=4)
                    print(f"Created {file_path}")

# 为 'llama' 和 'bloom' 模型生成JSON文件
create_json_files("llama", llama_providers, llama_times_data)
create_json_files("bloom", bloom_providers, bloom_times_data)
