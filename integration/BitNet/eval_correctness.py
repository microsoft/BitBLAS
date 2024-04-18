import math
import argparse
import torch
import random

from eval_utils import get_test_dataset
from modeling_bitnet import BitnetForCausalLM
from tokenization_bitnet import BitnetTokenizer 

from tqdm import tqdm
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--hf_path', default='1bitLLM/bitnet_b1_58-3B', type=str)
parser.add_argument('--seqlen', default=2048, type=int)


def calulate_loss(model, input, loss_fct):
    output = model(input,
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False)[0]
    shift_logits = output[:, :-1, :].contiguous()
    shift_labels = input[:, 1:]
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

def profile(model, input_data):
    import time

    import numpy as np
    model = model.cuda()
    model.eval()

    def get_runtime(num_repeats=1):
        tic = time.time()
        for _ in range(num_repeats):
            _ = model(input_data)
        torch.cuda.synchronize()
        return (time.time() - tic) * 1000 / num_repeats

    with torch.no_grad():
        # print("Warming up ...")
        st = time.time()
        while time.time() - st < 1.0:
            get_runtime()  # warmup
        warmup_runtime = get_runtime()
        num_repeats = max(1, int(1000 / warmup_runtime))
        times = get_runtime(num_repeats)
    return np.mean(times)

def main():
    model = BitnetForCausalLM.from_pretrained(
        '1bitLLM/bitnet_b1_58-3B',
        device_map='auto',
        low_cpu_mem_usage=True, 
        use_flash_attention_2=True,
        torch_dtype=torch.float16,
    ).half()
    with torch.no_grad():
        model._post_process_weights()
    input_id = torch.ones(1, 2048).long().cuda()
    # retrieve the model
    output = model(input_id)
    print(output)
    
    latency = profile(model, input_id)
    print("Latency: ", latency)

if __name__ == '__main__':
    main()