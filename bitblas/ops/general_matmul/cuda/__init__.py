# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# TODO: Not Implemented Yet
from bitblas.ops.operator import TransformKind
from bitblas.base import TileDevice
from .template import i4_scale_template_source


class MatmulDequantizeCudaEmitter:

    def __init__(
        self,
        M,
        N,
        K,
        in_dtype="float16",
        out_dtype="float16",
        accum_dtype="float16",
        bit=4,
        storage_dtype="int8",
        source_format="uint",
        with_scaling=False,
        with_zeros=False,
        group_size=-1,
        fast_decoding=False,
        with_bias=False,
        zeros_mode="original",
        propagate_a: TransformKind = TransformKind.NonTransform,
        propagate_b: TransformKind = TransformKind.NonTransform,
    ):
        self.N = N
        self.K = K
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.accum_dtype = accum_dtype
        self.bit = bit
        self.storage_dtype = storage_dtype
        self.source_format = source_format
        self.with_scaling = with_scaling
        self.with_zeros = with_zeros
        self.group_size = group_size if group_size != -1 else K
        self.fast_decoding = fast_decoding
        self.with_bias = with_bias
        self.zeros_mode = zeros_mode
        self.propagate_a = self._legalize_transform_kind(propagate_a)
        self.propagate_b = self._legalize_transform_kind(propagate_b)

    def _legalize_group_size(self):
        if self.group_size == -1:
            self.group_size = self.K

    def _legalize_transform_kind(self, propagate):
        if propagate is None:
            return TransformKind.NonTransform
        if isinstance(propagate, bool):
            return (TransformKind.IntraWarpTransform if propagate else TransformKind.NonTransform)
        elif isinstance(propagate, int):
            return TransformKind(propagate)

    def is_available(self, arch: TileDevice):
        conditions = []
        # group size must be -1, 128, k
        conditions.append(self.group_size in [-1, 128, self.K])
        # source format must be int
        conditions.append(self.source_format == "int")
        # with scaling must be true
        conditions.append(self.with_scaling)
        # with zeros must be false
        conditions.append(not self.with_zeros)
        # bit must be 4
        conditions.append(self.bit == 4)
        # in_dtype must be float16
        conditions.append(self.in_dtype == "float16")
        # out_dtype must be float16
        conditions.append(self.out_dtype == "float16")
        # accum_dtype must be float32
        conditions.append(self.accum_dtype == "float32")
        # sm version must be 80 (A100)
        conditions.append(self.arch.sm_version == 80)
        return all(conditions)

    def get_weight_transform(self):
        raise NotImplementedError

    def get_scale_transform(self):
        raise NotImplementedError

    def get_wrapped_source(self):
        wrapped_source = f"""
        extern "C" void init() {{

        }}
        extern "C" void call(half* __restrict__ A, int8_t* __restrict__ B, half* __restrict__ Scale, half* __restrict__ C, int m, void* workspace, cudaStream_t stream=cudaStreamDefault) {{
            marlin_cuda(A, B, C, Scale, m, {self.N}, {self.K}, workspace, {self.group_size}, 0, -1, -1, 108, 16);
        }}
        """
        return i4_scale_template_source + wrapped_source
