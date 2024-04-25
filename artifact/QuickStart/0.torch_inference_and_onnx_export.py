import torch
from transformers import LlamaModel, LlamaConfig
import os
import argparse

# seed
torch.manual_seed(0)

# For simplicity inputs, we export the model with word embeddings and lm_head.
# But for the artifact scaling, we can export the model with a single layer without word embeddings and lm_head
# to make sure the model's performance can be scaled.
export_single_layer = False 
config_70b = LlamaConfig(
    vocab_size=32000,
    hidden_size=8192,
    intermediate_size=28672,
    num_hidden_layers=1,
    num_attention_heads=64,
    num_key_value_heads=8,
    hidden_act="silu",
    max_position_embeddings=4096,
    initializer_range=0.02,
    rms_norm_eps=1e-05,
    use_cache=True,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    tie_word_embeddings=False,
)

model = LlamaModel(config_70b).half().cuda()
print(model.dtype)
if export_single_layer:
    model = model.layers[0]
model.eval()
parser = argparse.ArgumentParser()
parser.add_argument("--seq_length", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
args = parser.parse_args()

batch_size = args.batch_size
seq_length = args.seq_length

if export_single_layer:
    input_ids = torch.ones(batch_size, seq_length, config_70b.hidden_size, device="cuda", dtype=torch.float16)
else:
    input_ids = torch.ones(batch_size, seq_length, device="cuda", dtype=torch.int64)

# # initialize the weights
# for name, module in model.named_modules():
#     if isinstance(module, torch.nn.Linear):
#         # int4_tensor = torch.randint(-1, 1, module.weight.data.size(), dtype=torch.int8)
#         int4_tensor = torch.zeros_like(module.weight.data)
#         module.weight.data = int4_tensor.half().cuda()

# do inference and print the output
with torch.no_grad():
    output = model(input_ids)
    print(output.last_hidden_state)

# make a directory to save the model -> {llama2_70b_layer1_seq1_bs16/model.onnx}
dir_name = f"llama2_70b_single_layer"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# Save model into ONNX
torch.onnx.export(
    model,
    input_ids,
    f"{dir_name}/model.onnx",
    export_params=True,
    opset_version=13
)
