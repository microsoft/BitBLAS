import json


llama_providers = ['BS1 SEQ1', 'BS32 SEQ1', 'BS1 SEQ4096']
llama_times_data = [
    ('PyTorch-Inductor', [2700, 2624, 6878]),
    ('ONNXRuntime', [2716, 2803, 16078]),
    ('TensorRT', [5187, 4954, 6342]),
    ('vLLM', [5008, 4763, 5034]),
    ('vLLM-W$_{INT4}$A$_{FP16}$', [1123, 1100, 6128]),
    ('Welder', [2106, 2139, 6790]),
    ('Bitter', [2075, 2121, 6460]),
    ('Bitter-W$_{INT4}$A$_{FP16}$', [879, 817, 5216]),
    ('Bitter-W$_{NF4}$A$_{FP16}$', [866, 852, 5313]),
    ('Bitter-W$_{FP8}$A$_{FP8}$', [1306, 1192, 5769]),
    ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [1305, 1299, 5947]),
    ('Bitter-W$_{INT1}$A$_{INT8}$', [522, 532, 5300]),
]

bloom_providers = ['BS1 SEQ1', 'BS32 SEQ1', 'BS1 SEQ4096']
bloom_times_data = [
    ('PyTorch-Inductor', [11503, 12257, 15383]),
    ('ONNXRuntime', [7540, 7038, 62636]),
    ('TensorRT', [5566, 5875, 21209]),
    ('vLLM', [29011, 31764, 29199]),
    ('vLLM-W$_{INT4}$A$_{FP16}$', [22327, 21910, 21931]),
    ('Welder', [5130, 5036, 20109]),
    ('Bitter', [5169, 5117, 20977]),
    ('Bitter-W$_{INT4}$A$_{FP16}$', [3277, 3391, 18891]),
    ('Bitter-W$_{NF4}$A$_{FP16}$', [3374, 3374, 19772]),
    ('Bitter-W$_{FP8}$A$_{FP8}$', [4052, 3846, 18649]),
    ('Bitter-W$_{MXFP8}$A$_{MXFP8}$', [4037, 3944, 20280]),
    ('Bitter-W$_{INT1}$A$_{INT8}$', [3006, 3032, 17854]),
]


import json
import os

# 创建存储文件的目录
directory = "."
if not os.path.exists(directory):
    os.makedirs(directory)


name_mapping = {
    "PyTorch-Inductor": "pytorch",
    "ONNXRuntime": "onnxruntime",
    "TensorRT": "tensorrt",
    "vLLM": "vllm",
    "vLLM-W$_{INT4}$A$_{FP16}$": "vllm_fp16_int4",
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
