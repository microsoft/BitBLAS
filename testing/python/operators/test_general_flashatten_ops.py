# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import bitblas
import bitblas.testing
from bitblas import FlashAttenConfig, FlashAtten
import logging
from bitblas import set_log_level

set_log_level(logging.DEBUG)


# fmt: off
def flashatten_forward(batch, heads, seq_len, dim, Q_dtype, K_dtype, V_dtype, Accu_dtype, Out_dtype,
                       layout, is_causal):
    import torch
    torch.random.manual_seed(0)
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
    except ImportError:
        print("flash_attn is not installed, skipping test")
        return True

    type_convert_map = {"float16": torch.float16}

    flashatten_config = FlashAttenConfig(
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        dim=dim,
        Q_dtype=Q_dtype,
        K_dtype=K_dtype,
        V_dtype=V_dtype,
        Accu_dtype=Accu_dtype,
        Out_dtype=Out_dtype,
        layout=layout,
        is_causal=is_causal)
    flashatten = FlashAtten(config=flashatten_config, enable_tuning=False, backend="tl")

    Q_shape = [batch, seq_len, heads, dim]
    V_shape = [batch, seq_len, heads, dim]
    if layout == "ntn":
        K_shape = [batch, dim, heads, seq_len]
    else:
        K_shape = [batch, seq_len, heads, dim]
    Out_shape = [batch, seq_len, heads, dim]
    q = torch.rand(Q_shape, dtype=type_convert_map[Q_dtype]).cuda() - 0.5
    k = torch.rand(K_shape, dtype=type_convert_map[K_dtype]).cuda() - 0.5
    k_ref = k
    if layout == "ntn":
        k_ref = k.permute((0, 3, 2, 1))
    v = torch.rand(V_shape, dtype=type_convert_map[V_dtype]).cuda() - 0.5
    tl_output = torch.rand(Out_shape, dtype=type_convert_map[V_dtype]).cuda()

    ref_output = flash_attn_func(q, k_ref, v, causal=is_causal)
    flashatten(q, k, v, output=tl_output)
    print(ref_output)
    print(tl_output)
    torch.testing.assert_close(tl_output, ref_output, rtol=1e-1, atol=1e-1)


def test_flashatten_forward():
    flashatten_forward(1, 4, 256, 256, "float16", "float16", "float16", "float32", "float16", "nnn",
                       False)
    flashatten_forward(1, 4, 256, 256, "float16", "float16", "float16", "float32", "float16", "nnn",
                       True)
    flashatten_forward(1, 4, 256, 256, "float16", "float16", "float16", "float32", "float16", "ntn",
                       False)
    flashatten_forward(1, 4, 256, 256, "float16", "float16", "float16", "float32", "float16", "ntn",
                       True)


# fmt: on
if __name__ == "__main__":
    bitblas.testing.main()
