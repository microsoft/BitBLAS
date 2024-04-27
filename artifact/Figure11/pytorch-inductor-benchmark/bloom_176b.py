import torch
from transformers import BloomConfig, BloomModel, AutoConfig
import os
import time
import numpy as np
import argparse

export_single_layer = True
config = AutoConfig.from_pretrained("bigscience/bloom")
config.n_layer = 1
model = BloomModel(config).half().cuda()

if export_single_layer:
    model = model.h[0]
model.eval()
parser = argparse.ArgumentParser()
parser.add_argument("--seq_length", type=int, default=4096)
parser.add_argument("--batch_size", type=int, default=1)
args = parser.parse_args()
# Save an ONNX model with dynamic input shape

batch_size = args.batch_size
seq_length = args.seq_length

if export_single_layer:
    input_ids = torch.ones(
        batch_size, seq_length, config.hidden_size, device="cuda", dtype=torch.float16
    )
    alibi = torch.ones(
        batch_size * config.n_head, 1, seq_length, device="cuda", dtype=torch.float16
    )
    if seq_length == 1 and batch_size > 1:
        attention_mask = torch.ones(1, seq_length, device="cuda", dtype=torch.bool)
    else:
        attention_mask = torch.ones(
            batch_size, seq_length, device="cuda", dtype=torch.bool
        )
else:
    input_ids = torch.ones(batch_size, seq_length, device="cuda", dtype=torch.int64)

# torch inductor
with torch.no_grad(), torch.autocast("cuda"):
    compiled_model = torch.compile(model)


print("Compiling done, start inference")

while True:
    _ = compiled_model(input_ids)
