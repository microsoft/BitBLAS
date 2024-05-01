from time import sleep
import os
import sys
import contextlib
import subprocess
import time
import re
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['llama', 'bloom'], default='llama')
parser.add_argument('--framework', type=str, choices=['pytorch', 'onnxruntime', 'tensorrt', 'welder', 'vllm', 'vllm_fp16_int4', 'ladder', 'ladder_fp16_int4', 'ladder_fp16_nf4', 'ladder_fp8_fp8', 'ladder_mxfp8_mxfp8', 'ladder_int8_int1'], default='pytorch')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=1)
args = parser.parse_args()

model = args.model
framework = args.framework
batch_size = args.batch_size
seq_len = args.seq_len

pwd = os.getcwd()

CHECKPOINT_PATH = os.path.join(os.getcwd(), "../checkpoints/Figure11")

LADDER_HOME = f'{pwd}/../..'
LADDER_TVM_HOME = f'{LADDER_HOME}/3rdparty/tvm'
LADDER_CUTLASS_HOME = f'{LADDER_HOME}/3rdparty/cutlass'
PYTHONPATH = f'{LADDER_HOME}/python'
PYTHONPATH = f'{LADDER_TVM_HOME}/python:{PYTHONPATH}'
CPLUS_INCLUDE_PATH = f'{LADDER_CUTLASS_HOME}/include'

# MODEL_PATH=$(pwd)/../models
model_path = f'{pwd}/../models'

def analyze_log(log_path):
    if not os.path.exists(log_path):
        print(f"{log_path} does not exists")
    peak = 0
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            if 'MiB' in line:
                try:
                    peak = max(peak, int(re.split(' ',line)[0]))
                except Exception as err:
                    pass
    return peak

def pytorch_inference(model='llama', batch_size=1, seq_len=1):
    run_file = 'llama_70b.py' if model == 'llama' else 'bloom-176b.py'
    target_process = subprocess.Popen(f'cd {pwd}/pytorch-inductor-benchmark; python {run_file} --batch_size {batch_size} --seq_len {seq_len}; cd ..', shell=True)
    return target_process

def onnxruntime_inference(model='llama', batch_size=1, seq_len=1):
    run_file = 'ort_runtime.py'
    if model=='llama':
        model_file = f'{model_path}/llama_70b/llama2_70b_layer1_seq{seq_len}_bs{batch_size}/model.onnx'
    else:
        model_file = f'{model_path}/bloom-176b/bloom-176b_seq{seq_len}_bs{batch_size}/model.onnx'
    target_process = subprocess.Popen(f'cd {pwd}/onnxruntime-benchmark; python {run_file} --file {model_file} --iters 10000 ; cd ..', shell=True)
    return target_process

def tensorrt_inference(model='llama', batch_size=1, seq_len=1):
    # TRT_EXEC_PATH=$(pwd)/../../baseline_framework/TensorRT-9.0.1.4/bin
    trt_exec_path = f'{pwd}/../baseline_framework/TensorRT-9.0.1.4/bin/trtexec'
    if model=='llama':
        model_file = f'{model_path}/llama_70b/llama2_70b_layer1_seq{seq_len}_bs{batch_size}/model.trt'
    else:
        model_file = f'{model_path}/bloom-176b/bloom-176b_seq{seq_len}_bs{batch_size}/model.trt'
    target_process = subprocess.Popen(f'{trt_exec_path} --loadEngine {model_file} --fp16 --workspace=8192 --iterations=10000 ;', shell=True)
    return target_process

def vllm_inference(model='llama', batch_size=1, seq_len=1):
    # export VLLM_HOME=$(pwd)/../../baseline_framework/vLLM
    # export PYTHONPATH=$VLLM_HOME
    VLLM_HOME = f'{pwd}/../baseline_framework/vLLM'
    PYTHONPATH = f'{VLLM_HOME}'
    run_file = 'benchmark_llama.py' if model == 'llama' else 'benchmark_bloom.py'
    target_process = subprocess.Popen(f'cd {pwd}/vllm-benchmark; PYTHONPATH={PYTHONPATH} python {run_file}  --batch_size {batch_size} --seq_len {seq_len}; cd ..', shell=True)
    return target_process

def vllm_fp16_int4_inference(model='llama', batch_size=1, seq_len=1):
    run_file = 'benchmark_llama.py' if model == 'llama' else 'benchmark_bloom.py'
    target_process = subprocess.Popen(f'cd {pwd}/vllm-benchmark; python {run_file}  --batch_size {batch_size} --seq_len {seq_len}; cd ..', shell=True)
    return target_process

def welder_inference(model='llama', batch_size=1, seq_len=1):
    run_file = 'ort_runtime.py'
    if model=='llama':
        model_file = f'{model_path}/llama_70b/llama2_70b_layer1_seq{seq_len}_bs{batch_size}/model.onnx'
    else:
        model_file = f'{model_path}/bloom-176b/bloom-176b_seq{seq_len}_bs{batch_size}/model.onnx'
    target_process = subprocess.Popen(f'cd {pwd}/onnxruntime-benchmark; python {run_file} --file {model_file} --iters 10000 ; cd ..', shell=True)
    return target_process

def ladder_inference(model='llama', batch_size=1, seq_len=1):

    run_file = 'ladder_with_fake_dense_dequantize.py'
    if model=='llama':
        model_file = f'{CHECKPOINT_PATH}/ladder/checkpoints/llama2-70b/llama2_bs{batch_size}_seq{seq_len}_async'
    else:
        model_file = f'{CHECKPOINT_PATH}/ladder/checkpoints/bloom-176b/llama2_bs{batch_size}_seq{seq_len}_async'
    target_process = subprocess.Popen(f'cd {pwd}/ladder-benchmark; PYTHONPATH={PYTHONPATH} CPLUS_INCLUDE_PATH={CPLUS_INCLUDE_PATH} python {run_file} --prebuilt_path {model_file} ; cd ..', shell=True)
    return target_process

def ladder_fp16_int4_inference(model='llama', batch_size=1, seq_len=1):
    run_file = 'ladder_with_fake_dense_dequantize.py'
    if model=='llama':
        model_file = f'{CHECKPOINT_PATH}/ladder/checkpoints/llama2-70b/llama2_fq_0_int_4_-1_bs{batch_size}_seq{seq_len}_ci_False_async'
    else:
        model_file = f'{CHECKPOINT_PATH}/ladder/checkpoints/bloom-176b/llama2_fq_0_int_4_-1_bs{batch_size}_seq{seq_len}_ci_False_async'
    target_process = subprocess.Popen(f'cd {pwd}/ladder-benchmark; PYTHONPATH={PYTHONPATH} CPLUS_INCLUDE_PATH={CPLUS_INCLUDE_PATH} python {run_file} --prebuilt_path {model_file} ; cd ..', shell=True)
    return target_process

def ladder_fp16_nf4_inference(model='llama', batch_size=1, seq_len=1):
    run_file = 'ladder_with_fake_dense_dequantize.py'
    if model=='llama':
        model_file = f'{CHECKPOINT_PATH}/ladder/checkpoints/llama2-70b/llama2_fq_0_nf_4_-1_bs{batch_size}_seq{seq_len}_ci_False_async'
    else:
        model_file = f'{CHECKPOINT_PATH}/ladder/checkpoints/bloom-176b/llama2_fq_0_nf_4_-1_bs{batch_size}_seq{seq_len}_ci_False_async'
    target_process = subprocess.Popen(f'cd {pwd}/ladder-benchmark; PYTHONPATH={PYTHONPATH} CPLUS_INCLUDE_PATH={CPLUS_INCLUDE_PATH} python {run_file} --prebuilt_path {model_file} ; cd ..', shell=True)
    return target_process

def ladder_fp8_fp8_inference(model='llama', batch_size=1, seq_len=1):
    run_file = 'ladder_with_fake_dense_dequantize.py'
    if model=='llama':
        model_file = f'{CHECKPOINT_PATH}/ladder/checkpoints/llama2-70b/llama2_fq_0_fp_e5m2_8_-1_bs{batch_size}_seq{seq_len}_ci_False_async'
    else:
        model_file = f'{CHECKPOINT_PATH}/ladder/checkpoints/bloom-176b/llama2_fq_0_fp_e5m2_8_-1_bs{batch_size}_seq{seq_len}_ci_False_async'
    target_process = subprocess.Popen(f'cd {pwd}/ladder-benchmark; PYTHONPATH={PYTHONPATH} CPLUS_INCLUDE_PATH={CPLUS_INCLUDE_PATH} python {run_file} --prebuilt_path {model_file} ; cd ..', shell=True)
    return target_process

def ladder_mxfp8_mxfp8_inference(model='llama', batch_size=1, seq_len=1):
    run_file = 'ladder_with_fake_dense_dequantize.py'
    if model=='llama':
        model_file = f'{CHECKPOINT_PATH}/ladder/checkpoints/llama2-70b/llama2_fq_0_mxfp_8_-1_bs{batch_size}_seq{seq_len}_ci_False_async'
    else:
        model_file = f'{CHECKPOINT_PATH}/ladder/checkpoints/bloom-176b/llama2_fq_0_mxfp_8_-1_bs{batch_size}_seq{seq_len}_ci_False_async'
    target_process = subprocess.Popen(f'cd {pwd}/ladder-benchmark; PYTHONPATH={PYTHONPATH} CPLUS_INCLUDE_PATH={CPLUS_INCLUDE_PATH} python {run_file} --prebuilt_path {model_file} ; cd ..', shell=True)
    return target_process

def ladder_int8_int1_inference(model='llama', batch_size=1, seq_len=1):
    run_file = 'ladder_with_fake_dense_dequantize.py'
    if model=='llama':
        model_file = f'{CHECKPOINT_PATH}/ladder/checkpoints/llama2-70b/llama2_fq_0_int_1_-1_bs{batch_size}_seq{seq_len}_ci_True_async'
    else:
        model_file = f'{CHECKPOINT_PATH}/ladder/checkpoints/bloom-176b/llama2_fq_0_int_1_-1_bs{batch_size}_seq{seq_len}_ci_True_async'
    target_process = subprocess.Popen(f'cd {pwd}/ladder-benchmark; PYTHONPATH={PYTHONPATH} CPLUS_INCLUDE_PATH={CPLUS_INCLUDE_PATH} python {run_file} --prebuilt_path {model_file} ; cd ..', shell=True)
    return target_process

model_inference_mapping = {
    'pytorch': pytorch_inference,
    'onnxruntime': onnxruntime_inference,
    'tensorrt': tensorrt_inference,
    'vllm': vllm_inference,
    'vllm_fp16_int4': vllm_fp16_int4_inference,
    'welder': welder_inference,
    'ladder': ladder_inference,
    'ladder_fp16_int4': ladder_fp16_int4_inference,
    'ladder_fp16_nf4': ladder_fp16_nf4_inference,
    'ladder_fp8_fp8': ladder_fp8_fp8_inference,
    'ladder_mxfp8_mxfp8': ladder_mxfp8_mxfp8_inference,
    'ladder_int8_int1': ladder_int8_int1_inference
}

@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)

data = {}
path = './logs/{}_{}_{}_{}'.format(model, framework, batch_size, seq_len)
# create the directory if not exists
if not os.path.exists(path):
    os.makedirs(path)

memory_usage = 0
if os.path.exists(path):
    with pushd(path):
        print('Measure the memory for {} batch {} seq {} under {}'.format(model, batch_size, seq_len, framework))
        if os.path.exists('prepare_mem.sh'):
            os.system('bash prepare_mem.sh')
        # here start the inference process at the same time and
        # measure the memory at the same time
        inference_func = model_inference_mapping[framework]
        target_process = inference_func(model, batch_size, seq_len)
    sleep(30) # wait the memory to be steady (this large laguange model need more time to be steady)
    monitor_process = subprocess.Popen('bash nvidia_measure_memory.sh > run.log', shell=True)
    try:
        target_process.wait(timeout=10)
    except Exception as err:
        try:
            target_process.terminate()
            time.sleep(10)
        except Exception as err:
            print(err)
    monitor_process.terminate()
    memory_usage = analyze_log('run.log')
data['{}_{}_{}_{}'.format(model, framework, batch_size, seq_len)] = memory_usage
print(data)
with open(f'{args.model}_data.json', 'w') as f:
    json.dump(data, f)