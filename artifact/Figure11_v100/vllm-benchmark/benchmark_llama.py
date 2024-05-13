import torch
from transformers import LlamaModel, LlamaConfig, LlamaTokenizer
from transformers import AutoTokenizer
import os
import time
import numpy as np
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor import get_model, InputMetadata
import random
from vllm.model_executor.weight_utils import (get_quant_config,
                                              initialize_dummy_weights)
from vllm.model_executor.layers.quantization.awq import AWQLinearMethod, AWQConfig

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--seq_length", type=int, default=1)

args = parser.parse_args()
batch_size = args.batch_size
seq_length = args.seq_length

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
linear_method = None

run_single = True
enable_awq = True

if enable_awq:
    quant_config = AWQConfig(
        weight_bits=4,
        group_size=32,
        zero_point=False
    )
    linear_method = AWQLinearMethod(quant_config)
model = LlamaForCausalLM(config_70b, linear_method)
model = model.cuda().half()

if run_single:
    model = model.model.layers[0]

if run_single:
    hidden_states = torch.ones(batch_size, seq_length, config_70b.hidden_size, device="cuda", dtype=torch.float16)
    position_ids = torch.ones(batch_size, seq_length, device="cuda", dtype=torch.long)
    num_slots = 1 * 1
    slot_mapping = random.sample(range(num_slots), 1)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device="cuda")
    input_metadata = InputMetadata(
            prompt_lens=[1],
            slot_mapping=slot_mapping,
            max_context_len=None,
            context_lens=None,
            block_tables=None,
    )
    kv_caches = (None, None)
    cache_event = None
    residual = None
    args = (position_ids, hidden_states, kv_caches, input_metadata, cache_event, residual)
else:
    input_ids = torch.ones(batch_size, seq_length, device="cuda", dtype=torch.float16)
    position_ids = torch.ones(batch_size, seq_length, device="cuda", dtype=torch.long)
    input_metadata = InputMetadata(
            prompt_lens=[1],
            slot_mapping=1,
            max_context_len=None,
            context_lens=None,
            block_tables=None,
    )
    kv_caches = [(None, None)] * config_70b.num_hidden_layers
    cache_envents = None

while True:
    _ = model(*args)
