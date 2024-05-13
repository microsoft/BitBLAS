#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/Ladder/artifact/Figure13_v100/welder_roller_end2end/../../baseline_framework/Roller/python
export PYTHONPATH=/root/Ladder/artifact/Figure13_v100/welder_roller_end2end/../../baseline_framework/roller_tvm/python:$PYTHONPATH
export CPLUS_INCLUDE_PATH=/root/Ladder/artifact/Figure13_v100/welder_roller_end2end/../../baseline_framework/welder_cutlass/include:$CPLUS_INCLUDE_PATH
# Step 1: Getting the model block
/root/Ladder/artifact/Figure13_v100/welder_roller_end2end/../../baseline_framework/welder_nnfusion/build/src/tools/nnfusion/nnfusion /root/Ladder/artifact/Figure13_v100/welder_roller_end2end/../../models/llama_70b/llama2_70b_layer1_seq4096_bs1/model.onnx -f onnx -ftune_output_file=model.json > get_model_block.log 2>&1

# Step 2: Running the compiler
echo 'Running the compilation'
START_TIME=$(date +%s)
python3 -m run_compiler model.json tuned.json --device 0 --topk 20 > run_compiler.log 2>&1
END_TIME=$(date +%s)
echo "Compiler tuning time: $(($END_TIME - $START_TIME)) seconds" > tune_time_cost.txt

# Step 3: Code generation
echo 'Running Code Generation'
/root/Ladder/artifact/Figure13_v100/welder_roller_end2end/../../baseline_framework/welder_nnfusion/build/src/tools/nnfusion/nnfusion /root/Ladder/artifact/Figure13_v100/welder_roller_end2end/../../models/llama_70b/llama2_70b_layer1_seq4096_bs1/model.onnx -f onnx -ftune_output_file=/dev/null -ftune_input_file=tuned.json -fwarmup_step=5 -frun_step=10 > codegen.log 2>&1
cd nnfusion_rt/cuda_codegen;cmake . -DCUDA_ARCH='-gencode arch=compute_70,code=compute_70';make;./main_test > run.log 
cp run.log ../../
cd ../../; rm -rf nnfusion_rt

