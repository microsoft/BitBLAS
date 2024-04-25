import torch
from transformers import BloomConfig, BloomModel, AutoConfig
import os
import argparse

export_single_layer = True
config = AutoConfig.from_pretrained("bigscience/bloom")
config.n_layer = 1
model = BloomModel(config).half().cuda()
print(model.dtype)
if export_single_layer:
    model = model.h[0]
model.eval()
 
# Save an ONNX model with dynamic input shape

parser = argparse.ArgumentParser()
parser.add_argument("--seq_length", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

# batch_size, seq_length
batch_size = args.batch_size
seq_length = args.seq_length

if export_single_layer:
    input_ids = torch.ones(batch_size, seq_length, config.hidden_size, device="cuda", dtype=torch.float16)
    alibi = torch.ones(batch_size * config.n_head, 1, seq_length, device="cuda", dtype=torch.float16)
    if seq_length == 1 and batch_size > 1:
        attention_mask = torch.ones(1, seq_length, device="cuda", dtype=torch.bool)
    else:
        attention_mask = torch.ones(batch_size, seq_length, device="cuda", dtype=torch.bool)
else:
    input_ids = torch.ones(batch_size, seq_length, device="cuda", dtype=torch.int64)
# forward test
if not export_single_layer:
    outputs = model(input_ids)
# make a directory to save the model -> {bloom-176b_layer1_seq1_bs16/model.onnx}
dir_name = f"bloom-176b_layer1_seq{seq_length}_bs{batch_size}"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    
# Save model into ONNX
torch.onnx.export(
    model,
    (input_ids, alibi, attention_mask),
    f"{dir_name}/model.onnx",
    export_params=True,
    opset_version=13
)
