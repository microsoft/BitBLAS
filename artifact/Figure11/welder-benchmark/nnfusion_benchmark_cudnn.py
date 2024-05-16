# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from configs import models

welder_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "baseline_framework", "WELDER"
)

welder_tvm_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "baseline_framework", "welder_tvm"
)

welder_nnfusion_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "baseline_framework", "welder_nnfusion"
)

welder_cutlass_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "baseline_framework", "welder_cutlass"
)

for model_name, model_path in models.items():
    model_name = model_name + "_cutlass"
    compiled_model_path = os.path.join("./compiled_models", model_name)
    # 创建文件夹
    if not os.path.exists(compiled_model_path):
        # -p: make parent directories as needed
        os.makedirs(compiled_model_path)
    else:
        print(f"Folder {compiled_model_path} already exists.")
        continue

    # 创建并写入.sh文件
    sh_filepath = os.path.join(compiled_model_path, f"{model_name}.sh")
    print(f"Writing to {sh_filepath}")
    with open(sh_filepath, "w") as f:
        f.write(f"#!/bin/bash\n\n")
        f.write(f"export CUDA_VISIBLE_DEVICES=0\n")
        f.write(
            f"export PYTHONPATH={welder_path}/python\n"
        )
        f.write(
            f"export PYTHONPATH={welder_tvm_path}/python:$PYTHONPATH\n"
        )
        f.write(
            f"export CPLUS_INCLUDE_PATH={welder_cutlass_path}/include:$CPLUS_INCLUDE_PATH\n"
        )

        f.write(f"# Step 1: Getting the model block\n")
        f.write(
            f"{welder_nnfusion_path}/build/src/tools/nnfusion/nnfusion {model_path} -f onnx -ftune_output_file=model.json > get_model_block.log 2>&1\n\n"
        )

        f.write(f"# Step 2: Running the compiler\n")
        f.write("echo 'Running the compilation'\n")
        f.write(f"START_TIME=$(date +%s)\n")
        f.write(
            f"python3 -m run_compiler model.json tuned.json --device 0 --topk 20 > run_compiler.log 2>&1\n"
        )
        f.write(f"END_TIME=$(date +%s)\n")
        f.write(
            f'echo "Compiler tuning time: $(($END_TIME - $START_TIME)) seconds" > tune_time_cost.txt\n\n'
        )

        f.write(f"# Step 3: Code generation\n")
        f.write("echo 'Running Code Generation'\n")
        f.write(
            f"{welder_nnfusion_path}/build/src/tools/nnfusion/nnfusion {model_path} -f onnx -ftune_output_file=/dev/null -ftune_input_file=tuned.json -fwarmup_step=5 -frun_step=10000 > codegen.log 2>&1\n"
        )
        f.write(
            f"cd nnfusion_rt/cuda_codegen;cmake . -DCUDA_ARCH='-gencode arch=compute_80,code=compute_80';make;./main_test > run.log \n"
        )
        f.write(f"cp run.log ../../\n")
        f.write(f"sed -i 's/int steps = 100;/int steps = 10000;/' nnfusion_rt/cuda_codegen/main_test.cpp\n")

    # 让.sh文件具有执行权限
    os.chmod(sh_filepath, 0o755)

    # 在文件夹内执行.sh文件
    os.system(f"cd {compiled_model_path} && ./{model_name}.sh")
