# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas import tvm as tvm
import bitblas.testing
from tvm.ir import structural_equal
from bitblas.ops.general_flashatten.tilelang.flashatten import FlashAttenScheduler


def assert_flashatten_scheduler_simplify(batch,
                                         heads,
                                         seq_len,
                                         dim,
                                         trans_K=False,
                                         dtype_QKV="float16",
                                         dtype_Out="float16",
                                         dtype_Accu="float32",
                                         is_causal=False):
    flashatten = FlashAttenScheduler(
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        dim=dim,
        trans_K=trans_K,
        dtype_QKV=dtype_QKV,
        dtype_Out=dtype_Out,
        dtype_Accu=dtype_Accu,
        is_causal=is_causal,
    ).deactivate_simplify().with_default_config()

    simplified_flashatten = FlashAttenScheduler.Simplify(flashatten)

    is_equal = structural_equal(flashatten, simplified_flashatten)

    assert is_equal is False, "Simplify should not return the same schedule"


def test_scheduler_simplify():
    assert_flashatten_scheduler_simplify(1, 4, 256, 256)


if __name__ == "__main__":
    bitblas.testing.main()
