import torch
from transformers import LlamaModel, LlamaConfig
import time
import numpy as np
import argparse

export_single_layer = True
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
if export_single_layer:
    model = model.layers[0]
model.eval()
parser = argparse.ArgumentParser()
parser.add_argument("--seq_length", type=int, default=4096)
parser.add_argument("--batch_size", type=int, default=1)
args = parser.parse_args()

batch_size = args.batch_size
seq_length = args.seq_length

# Save an ONNX model with dynamic input shape

if export_single_layer:
    input_ids = torch.ones(
        batch_size,
        seq_length,
        config_70b.hidden_size,
        device="cuda",
        dtype=torch.float16,
    )
else:
    input_ids = torch.ones(batch_size, seq_length, device="cuda", dtype=torch.int64)

# torch inductor
with torch.no_grad(), torch.autocast("cuda"):
    compiled_model = torch.compile(model)

while True:
    _ = compiled_model(input_ids)
