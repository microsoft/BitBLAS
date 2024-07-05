# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# pre-transformed tir expression of matmul
from bitblas import tvm
from tvm import te
from bitblas.ops.operator import TransformKind
from .base import TIRScriptEmitter, TIRScriptSelector


class BatchMatMulEmitter(TIRScriptEmitter):

    def __init__(
        self,
        batch,
        M,
        N,
        K,
        in_dtype="float16",
        out_dtype="float16",
        accum_dtype="float16",
        with_bias=False,
        layout="nt",
    ):
        self.batch = batch
        self.M = self._validate_dimension(M, "M")
        self.N = self._validate_dimension(N, "N")
        self.K = self._validate_dimension(K, "K")
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.accum_dtype = accum_dtype
        self.with_bias = with_bias
        self.layout = layout
        self._validate_layout()

    @staticmethod
    def _validate_dimension(dim, name):
        if not isinstance(dim, int):
            return tvm.te.var(name.lower())
        return dim

    def _validate_layout(self):
        if self.layout not in ["nn", "nt"]:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if self.layout == "nn":
            raise ValueError("Currently only support layout=nt")

    def _create_placeholders(self):
        A = te.placeholder((self.batch, self.M, self.K), name="A", dtype=self.in_dtype)
        B = te.placeholder((self.batch, self.N, self.K), name="B", dtype=self.in_dtype)
        Bias = te.placeholder(
            (self.N,), name="Bias", dtype=self.in_dtype) if self.with_bias else None
        return A, B, Bias

    def _compute_matmul(self, A, B):
        k = te.reduce_axis((0, self.K), name="k")
        C = te.compute(
            (self.batch, self.M, self.N),
            lambda b, i, j: te.sum(
                A[b, i, k].astype(self.accum_dtype) * B[b, j, k].astype(self.accum_dtype), axis=k),
            name="C",
        )
        return C

    def _apply_bias(self, C, Bias):
        if self.with_bias:
            return te.compute((self.batch, self.M, self.N),
                              lambda b, i, j: C[b, i, j] + Bias[j],
                              name="E")
        return C

    def _convert_dtype(self, tensor):
        if self.accum_dtype != self.out_dtype:
            return te.compute((self.batch, self.M, self.N),
                              lambda b, i, j: tensor[b, i, j].astype(self.out_dtype),
                              name="D")
        return tensor

    def emit(self):
        A, B, Bias = self._create_placeholders()
        C = self._compute_matmul(A, B)
        last_output = self._convert_dtype(C)
        if self.with_bias:
            last_output = self._apply_bias(last_output, Bias)

        args = [A, B, Bias, last_output] if self.with_bias else [A, B, last_output]
        func = te.create_prim_func(args)
        return tvm.IRModule.from_expr(func)


class BatchMatMulSelector(TIRScriptSelector):

    def __init__(self,
                 propagate_a: TransformKind = TransformKind.NonTransform,
                 propagate_b: TransformKind = TransformKind.NonTransform):
        self.propagate_a = propagate_a
        self.propagate_b = propagate_b

    def select(
        self,
        batch=1,
        M=None,
        N=16384,
        K=16384,
        in_dtype="float16",
        out_dtype="float16",
        accum_dtype="float16",
        with_bias=False,
        layout="nt",
    ):
        if layout == "nn":
            if self.propagate_a or self.propagate_b:
                raise ValueError(
                    "Currently only support propagate_a=False and propagate_b=False for layout=nn")
            return BatchMatMulEmitter(batch, M, N, K, in_dtype, out_dtype, accum_dtype, with_bias,
                                      layout).emit()
        elif layout == "nt":
            if self.propagate_a and self.propagate_b:
                raise ValueError("Currently only support propagate_a or propagate_b for layout=nt")
            elif self.propagate_a:
                raise ValueError("Currently only support propagate_a=False for layout=nt")
            elif self.propagate_b:
                raise ValueError("Currently only support propagate_b=False for layout=nt")
            else:
                return BatchMatMulEmitter(batch, M, N, K, in_dtype, out_dtype, accum_dtype,
                                          with_bias, layout).emit()
        else:
            raise ValueError(f"Unsupported layout: {layout}")


def select_implementation(
    Batch=1,
    M=None,
    N=16384,
    K=16384,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float16",
    with_bias=False,
    layout="nt",
    propagate_a: TransformKind = TransformKind.NonTransform,
    propagate_b: TransformKind = TransformKind.NonTransform,
):
    selector = BatchMatMulSelector(propagate_a, propagate_b)
    return selector.select(Batch, M, N, K, in_dtype, out_dtype, accum_dtype, with_bias, layout)
