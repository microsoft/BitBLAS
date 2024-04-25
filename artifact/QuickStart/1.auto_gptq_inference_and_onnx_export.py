import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONPATH"] = '/workspace/v-leiwang3/AutoGPTQ.tvm' + ":" + os.environ.get("PYTHONPATH", "")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from transformers import AutoTokenizer, TextGenerationPipeline, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch
import time

enable_quantize = True
export_nnfusion = True
use_triton = False

assert not (export_nnfusion and use_triton)

pretrained_model_dir = "facebook/opt-125m"
quantized_model_dir = f"qmodels/opt-125m-4bit"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)

examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

if enable_quantize:
    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize bits
        # desc_act=False,  # disable activation description
        # group_size=128,  # disable group quantization
        desc_act=True
    )

    # load un-quantized model, the model will always be force loaded into cpu
    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask" 
    # with value under torch.LongTensor type.
    model.quantize(examples, use_tvm=False if use_triton else True, use_triton=use_triton, export_nnfusion=export_nnfusion)

    # save quantized model
    model.save_quantized(quantized_model_dir)
# load quantized model to the first GPU
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_tvm=False if use_triton else True, use_triton=use_triton, export_nnfusion=export_nnfusion).half().cuda()
# or you can also use pipeline
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
print(pipeline("auto-gptq is")[0]["generated_text"])

# export 2 onnx
batch_size = 1
seq_length = 1
input_shape = (batch_size, seq_length)

input_ids = torch.ones(batch_size, seq_length, device="cuda", dtype=torch.int64)
with torch.no_grad():
    output = model(input_ids)
    print(output.logits)

onnx_name = f"qmodel_b{batch_size}s{seq_length}.onnx"
output_path = os.path.join(quantized_model_dir, f"qmodel_b{batch_size}s{seq_length}", onnx_name)
if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))
input_ids = torch.ones(input_shape, dtype=torch.long, device="cuda:0")
attention_mask = torch.ones(input_shape, dtype=torch.long, device="cuda:0")


def tofp16model(in_file_name, out_file_name):
    from onnx import checker, load_model, save_model
    from onnxconverter_common import convert_float_to_float16
    onnx_model = load_model(in_file_name)
    trans_model = convert_float_to_float16(onnx_model, keep_io_types=False)
    # checker.check_model(trans_model)
    save_model(trans_model, out_file_name)
    
if not export_nnfusion:
    start = time.time()
    for i in range(100):
        outputs = model(input_ids=input_ids)
    end = time.time()
    print("time: ", (end - start) / 100 * 1000, "ms")
    print(outputs.logits)
else:
    model = model.model.model
    model = model.half().cuda()
    model.eval()
    torch.onnx.export(      
        model,  
        input_ids,  
        f=output_path,  
        opset_version=13, 
    )
    # tofp16model(output_path, output_path.replace(".onnx", "_fp16.onnx"))
    import onnx
    from onnxsim import simplify
    
    model = onnx.load(output_path)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    output_path = output_path.replace(".onnx", "_simplified.onnx")
    onnx.save(model_simp, output_path, save_as_external_data=True)
    print("export onnx done")