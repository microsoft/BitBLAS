# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import bitblas
import bitblas.testing
from bitblas import FlashAttenConfig, FlashAtten
import logging
from bitblas import set_log_level

set_log_level(logging.DEBUG)


def get_codegen_result(ops):
    code = ops.get_source()
    return code


# fmt: off
def flashatten_codegen_default(batch, heads, seq_len, dim, Q_dtype, K_dtype, V_dtype, Accu_dtype,
                               Out_dtype, layout, is_causal):

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
    assert get_codegen_result(flashatten)


def test_fa_codegen_default():
    flashatten_codegen_default(1, 4, 256, 256, "float16", "float16", "float16", "float32",
                               "float16", "nnn", False)
    flashatten_codegen_default(1, 4, 256, 256, "float16", "float16", "float16", "float32",
                               "float16", "ntn", False)

# fmt: on
if __name__ == "__main__":
    bitblas.testing.main()
