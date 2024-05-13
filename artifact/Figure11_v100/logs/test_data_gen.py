import json

llama_providers = ["BS1 SEQ1", "BS32 SEQ1", "BS1 SEQ4096"]
llama_times_data = [
    ('PyTorch-Inductor', [2740, 2606, 7062]),
    ('ONNXRuntime', [2619, 2856, 16744]),
    ('TensorRT', [5269, 5109, 6031]),
    ('vLLM', [0, 0, 0]),
    ('vLLM-W$_{INT4}$A$_{FP16}$', [0, 0, 0]),
    ('Welder', [2098, 2184, 6927]),
    ('Bitter', [2078, 2111, 6424]),
    ('Bitter-W$_{INT4}$A$_{FP16}$', [875, 806, 5287]),
    ('Bitter-W$_{NF4}$A$_{FP16}$', [844, 816, 5406]),
    ('Bitter-W$_{FP8}$A$_{FP8}$', [1280, 1250, 5621]),
    ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [1372, 1399, 5587]),
    ('Bitter-W$_{INT1}$A$_{INT8}$', [507, 527, 5114]),
]

bloom_providers = ["BS1 SEQ1", "BS32 SEQ1", "BS1 SEQ4096"]
bloom_times_data = [
    ('PyTorch-Inductor', [11790, 11675, 0]),
    ('ONNXRuntime', [7303, 7073, 0]),
    ('TensorRT', [6023, 5891, 0]),
    ('vLLM', [0, 0, 0]),
    ('vLLM-W$_{INT4}$A$_{FP16}$', [0, 0, 0]),
    ('Welder', [4978, 5291, 0]),
    ('Bitter', [5235, 5323, 0]),
    ('Bitter-W$_{INT4}$A$_{FP16}$', [3216, 3275, 0]),
    ('Bitter-W$_{NF4}$A$_{FP16}$', [3493, 3317, 0]),
    ('Bitter-W$_{FP8}$A$_{FP8}$', [3820, 3913, 0]),
    ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [4594, 4519, 0]),
    ('Bitter-W$_{INT1}$A$_{INT8}$', [2903, 2850, 0]),
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
