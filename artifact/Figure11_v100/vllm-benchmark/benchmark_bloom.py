import torch
from transformers import LlamaModel, LlamaConfig, LlamaTokenizer
from transformers import AutoTokenizer, AutoConfig
import os
import time
import numpy as np
from vllm.model_executor.models.bloom import BloomForCausalLM
from vllm.model_executor import get_model, InputMetadata
import random
from vllm.model_executor.weight_utils import (get_quant_config,
                                              initialize_dummy_weights)
from vllm.model_executor.layers.quantization.awq import AWQLinearMethod, AWQConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--seq_length", type=int, default=1)

args = parser.parse_args()
batch_size = args.batch_size
seq_length = args.seq_length

run_single = True
enable_awq = False

linear_method = None
if enable_awq:
    quant_config = AWQConfig(
        weight_bits=4,
        group_size=32,
        zero_point=False
    )
    linear_method = AWQLinearMethod(quant_config)
config = AutoConfig.from_pretrained("bigscience/bloom")
config.n_layer = 1
model = BloomForCausalLM(config, linear_method=linear_method)
model = model.cuda().half()
if run_single:
    model = model.transformer.h[0]
if run_single:
    hidden_states = torch.ones(batch_size, seq_length, config.hidden_size, device="cuda", dtype=torch.float16)
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
    args = (position_ids, hidden_states, kv_caches, input_metadata, cache_event)
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
    kv_caches = [(None, None)] * config.num_hidden_layers
    cache_envents = None

while True:
    _ = model(*args)
