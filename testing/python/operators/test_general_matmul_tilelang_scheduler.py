# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas import tvm as tvm
import bitblas.testing
from tvm.ir import structural_equal
from bitblas.ops.general_matmul.tilelang.dense.matmul_tile import (
    MatmulTileLibraryScheduler,)
from bitblas.ops.general_matmul.tilelang.dequantize import (MatmulDequantizeScheduler)
from bitblas.ops.general_matmul.tilelang.dense.gemv_simt import GemvFineGrainSIMTScheduler
from bitblas.ops.general_matmul.tilelang.dense import MatmulScheduler


def assert_gemv_scheduler_simplify(M,
                                   N,
                                   K,
                                   trans_A=False,
                                   trans_B=True,
                                   in_dtype="float16",
                                   out_dtype="float16",
                                   accum_dtype="float16"):
    matmul = GemvFineGrainSIMTScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
    ).deactivate_simplify().with_default_config()

    simplified = GemvFineGrainSIMTScheduler.Simplify(matmul)
    print(simplified)
    is_equal = structural_equal(matmul, simplified)
    if is_equal:
        print("Matmul is simplified")
    else:
        print("Matmul is not simplified")

    assert simplified is not None, "Simplify should return a schedule"


def assert_dense_scheduler_simplify(M,
                                    N,
                                    K,
                                    trans_A=False,
                                    trans_B=True,
                                    in_dtype="float16",
                                    out_dtype="float16",
                                    accum_dtype="float16"):
    matmul = MatmulTileLibraryScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
    ).deactivate_simplify().with_default_config()

    simplified = MatmulTileLibraryScheduler.Simplify(matmul)

    is_equal = structural_equal(matmul, simplified)
    if is_equal:
        print("Matmul is simplified")
    else:
        print("Matmul is not simplified")

    assert simplified is not None, "Simplify should return a schedule"


def assert_dequantize_scheduler_simplify(
    M,
    N,
    K,
    trans_A=False,
    trans_B=True,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    num_bits=4,
    storage_dtype="int8",
    source_format="uint",
    with_scaling=False,
    with_zeros=False,
    group_size=-1,
    fast_decoding=False,
    zeros_mode="original",
):
    matmul = MatmulDequantizeScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        num_bits=num_bits,
        storage_dtype=storage_dtype,
        source_format=source_format,
        with_scaling=with_scaling,
        with_zeros=with_zeros,
        group_size=group_size,
        fast_decoding=fast_decoding,
        zeros_mode=zeros_mode,
    ).deactivate_simplify().with_default_config()

    simplified = MatmulDequantizeScheduler.Simplify(matmul)
    print(simplified)
    is_equal = structural_equal(matmul, simplified)  # noqa: F841
    assert simplified is not None, "Simplify should return a schedule"


def assert_matmul_scheduler_with_default(M,
                                         N,
                                         K,
                                         trans_A=False,
                                         trans_B=True,
                                         in_dtype="float16",
                                         out_dtype="float16",
                                         accum_dtype="float16"):
    matmul = MatmulScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
    ).deactivate_simplify().with_default_config()
    print(matmul)
    assert matmul is not None, "with_default_config should return a schedule"


def assert_matmul_scheduler_get_hints(
    M,
    N,
    K,
    trans_A=False,
    trans_B=True,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
):
    scheduler = MatmulScheduler(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
    )
    hints = scheduler.get_hardware_aware_configs()
    for hint in hints:
        print(type(hint), hint)
    matmul = scheduler.apply_config(hint=hints[0])
    print(matmul)
    assert hints is not None, "with_default_config should return a schedule"


def test_scheduler_simplify():
    assert_dense_scheduler_simplify(128, 128, 128)


def test_dequantize_scheduler_simplify():
    assert_dequantize_scheduler_simplify(128, 128, 128)
    assert_dequantize_scheduler_simplify(128, 128, 128, with_scaling=True)
    assert_dequantize_scheduler_simplify(
        128, 128, 128, with_scaling=True, with_zeros=True, zeros_mode="original")
    assert_dequantize_scheduler_simplify(
        128, 128, 128, with_scaling=True, with_zeros=True, zeros_mode="rescale")
    assert_dequantize_scheduler_simplify(
        128, 128, 128, with_scaling=True, with_zeros=True, zeros_mode="quantized")


def test_matmul_scheduler_with_default():
    assert_matmul_scheduler_with_default(1, 128, 128)
    assert_matmul_scheduler_with_default(128, 128, 128)


def test_matmul_scheduler_get_hints():
    assert_matmul_scheduler_get_hints(1, 128, 128)
    assert_matmul_scheduler_get_hints(128, 128, 128)


if __name__ == "__main__":
    bitblas.testing.main()
