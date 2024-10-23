# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm as tvm
from tvm import DataType
import tvm.tl.language as T
from typing import Optional
from bitblas.tl.utils import (
    get_mma_micro_size,  # noqa: F401
    make_swizzle_layout,  # noqa: F401
)
from .finegrained_primitive_tensorcore import MatmulDequantizeFineGrainedScheduler
from bitblas.tl.macro_generator import (
    TensorCoreIntrinEmitterWithLadderTransform,  # noqa: F401
)
from bitblas.ops.common import TransformKind  # noqa: F401
from dataclasses import dataclass
from bitblas.quantization import (
    _tir_packed_to_unsigned_convert,)
from bitblas.gpu.intrin.lop3 import get_lop3_intrin_group
from bitblas.gpu.matmul_analysis import (
    get_propagate_map,
    get_ladder_stage3_map,
)

# GPU warp configuration for NVIDIA GPUs
warp_size = 32


@dataclass
class MatmulDequantizeWeightPropagationScheduler(MatmulDequantizeFineGrainedScheduler):

    # Ladder Transform Config
    weight_transform_kind: TransformKind = TransformKind.LDMatrixTransform

    def apply_config(
        self,
        block_row_warps: Optional[int] = None,
        block_col_warps: Optional[int] = None,
        warp_row_tiles: Optional[int] = None,
        warp_col_tiles: Optional[int] = None,
        chunk: Optional[int] = None,
        num_stages: Optional[int] = None,
        enable_rasterization=False,
    ):
        assert block_row_warps is not None, "block_row_warps is required"
        assert block_col_warps is not None, "block_col_warps is required"
        assert warp_row_tiles is not None, "warp_row_tiles is required"
        assert warp_col_tiles is not None, "warp_col_tiles is required"
        assert chunk is not None, "chunk is required"
        assert num_stages is not None, "num_stages is required"

        M, N, K = self.M, self.N, self.K
        trans_A, trans_B = self.trans_A, self.trans_B
        weight_transform_kind = self.weight_transform_kind

        assert trans_A is False, "Dequantize only implement for trans_A=False currently"
        assert trans_B is True, "Dequantize only implement for trans_B=TRue currently"
        assert (weight_transform_kind == TransformKind.LDMatrixTransform
               ), "Dequantize only implement for LDMatrixTransform currently"

        in_dtype, out_dtype, accum_dtype = (
            self.in_dtype,
            self.out_dtype,
            self.accum_dtype,
        )
        # Calculate the micro size per warp using a helper function
        micro_size_x, micro_size_y, micro_size_k = get_mma_micro_size(in_dtype)

        block_M = block_row_warps * warp_row_tiles
        block_N = block_col_warps * warp_col_tiles
        block_K = chunk
        threads = warp_size * (block_row_warps * block_col_warps)

        fragement_size = (micro_size_x * micro_size_y) // warp_size
        warp_rows = warp_row_tiles // micro_size_x
        warp_cols = warp_col_tiles // micro_size_y

        fast_decoding = self.fast_decoding

        num_bits = self.num_bits
        storage_dtype = self.storage_dtype
        source_format = self.source_format
        storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
        num_elems_per_byte = self.num_elems_per_byte

        MAX_TRANSACTION_SIZE_IN_BITS = 128
        local_size = MAX_TRANSACTION_SIZE_IN_BITS // DataType(in_dtype).bits

        group_size = self.group_size
        if group_size == -1:
            group_size = K

        A_shape = (M, K)
        B_shape = (
            N // micro_size_y,
            K // micro_size_k,
            micro_size_y,
            micro_size_k // num_elems_per_byte,
        )
        LUT_shape = (group_size, K // num_elems_per_byte)
        Scale_shape = (N, K // group_size)
        Zeros_shape = (N, K // group_size)
        Qzeros_shape = ((K // group_size), N // storage_nbit * num_bits)
        Bias_shape = (N,)

        A_shared_shape = (block_M, block_K)
        B_shared_shape = (
            block_N // micro_size_y,
            block_K // micro_size_k,
            micro_size_y,
            micro_size_k // num_elems_per_byte,
        )

        C_shared_shape = (
            block_M // micro_size_x,
            block_N // micro_size_y,
            micro_size_x,
            micro_size_y,
        )

        import_source: Optional[str] = None
        func_name: str = ""
        if fast_decoding is True:
            lop3_intrin_info = get_lop3_intrin_group(
                out_dtype=out_dtype,
                source_format=source_format,
                source_bit=num_bits,
                storage_dtype=storage_dtype,
                with_scaling=self.with_scaling,
                with_zeros=self.with_zeros,
                storage_scope="warp",  # to get the ladder transform lop3 intrin
            )
            import_source = lop3_intrin_info["c_source"]
            func_name = lop3_intrin_info["func_name"]
            assert import_source is not None, "lop3_intrin_info is not found"
            assert func_name is not None, "lop3_intrin_info is not found"
            import_source = self.common_header + import_source

        # Configure the tensor core intrinsic emitter with ladder transform
        mma_emitter = TensorCoreIntrinEmitterWithLadderTransform(
            a_dtype=in_dtype,
            b_dtype=in_dtype,
            accum_dtype=accum_dtype,
            a_transposed=trans_A,
            b_transposed=trans_B,
            block_row_warps=block_row_warps,
            block_col_warps=block_col_warps,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=chunk,
            transform_kind_b=weight_transform_kind,
            num_elems_per_byte=num_elems_per_byte,
        )

        vec_load_qb = 16
        if block_N * block_K // num_elems_per_byte // threads < vec_load_qb:
            vec_load_qb = block_N * block_K // num_elems_per_byte // threads

        @T.prim_func
        def general_dequant_matmul(
                A: T.Buffer(A_shape, in_dtype),
                B: T.Buffer(B_shape, storage_dtype),
                LUT: T.Buffer(LUT_shape, in_dtype),
                Scale: T.Buffer(Scale_shape, in_dtype),
                Qzeros: T.Buffer(Qzeros_shape, storage_dtype),
                Zeros: T.Buffer(Zeros_shape, in_dtype),
                Bias: T.Buffer(Bias_shape, in_dtype),
                C: T.Buffer((M, N), out_dtype),
        ):
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared(A_shared_shape, in_dtype)
                B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
                C_shared = T.alloc_shared(C_shared_shape, out_dtype)

                A_frag = T.alloc_local((warp_rows * fragement_size), in_dtype)
                B_frag = T.alloc_local((warp_cols * fragement_size // num_elems_per_byte),
                                       storage_dtype)
                B_dequantize_frag = T.alloc_local((warp_cols * fragement_size), in_dtype)
                C_frag = T.alloc_local((warp_rows * warp_cols * fragement_size), accum_dtype)

                tx = T.thread_binding(0, threads, thread="threadIdx.x")

                T.annotate_layout({
                    A_shared: make_swizzle_layout(A_shared),
                })

                T.use_swizzle(10, enable=enable_rasterization)

                T.import_source(import_source)

                T.clear(C_frag)

                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):

                    T.copy(A[by * block_M, ko * block_K], A_shared)

                    for i in T.serial(block_N * block_K // num_elems_per_byte //
                                      (threads * vec_load_qb)):
                        for v in T.vectorized(0, vec_load_qb):
                            idx = i * threads * vec_load_qb + tx * vec_load_qb + v
                            vkk = idx % (micro_size_k // num_elems_per_byte)
                            vjj = (idx // (micro_size_k // num_elems_per_byte)) % micro_size_y
                            vk = (idx // (micro_size_k // num_elems_per_byte) // micro_size_y) % (
                                block_K // micro_size_k)
                            vj = (idx // (micro_size_k // num_elems_per_byte) // micro_size_y //
                                  (block_K // micro_size_k)) % (
                                      block_N // micro_size_y)
                            B_shared[vj, vk, vjj, vkk] = B[
                                bx * (block_N // micro_size_y) + vj,
                                ko * (block_K // micro_size_k) + vk,
                                vjj,
                                vkk,
                            ]

                    # Perform the matrix multiplication on tensor core fragments
                    for ki in T.serial(0, (block_K // micro_size_k)):

                        # Load A fragment
                        mma_emitter.ldmatrix_a(
                            A_frag,
                            A_shared,
                            ki,
                            thread_bindings=tx,
                        )

                        # Load B fragment
                        mma_emitter.ldmatrix_b(
                            B_frag,
                            B_shared,
                            ki,
                            thread_bindings=tx,
                        )

                        if fast_decoding is True:
                            self._normal_fast_dequant(
                                B_frag,
                                B_dequantize_frag,
                                Scale,
                                Zeros,
                                Qzeros,
                                func_name,
                                local_size,
                                warp_cols,
                                bx,
                                tx,
                                mma_emitter,
                                ko,
                                ki,
                                block_N,
                                block_K,
                            )
                        else:
                            self._normal_dequant(
                                B_frag,
                                B_dequantize_frag,
                                Scale,
                                Zeros,
                                Qzeros,
                                local_size,
                                warp_cols,
                                bx,
                                tx,
                                mma_emitter,
                                ko,
                                ki,
                                block_N,
                                block_K,
                            )

                        # Matrix multiplication on fragments
                        mma_emitter.mma(A_frag, B_dequantize_frag, C_frag)

                # Store the result back to C shared memory
                mma_emitter.stmatrix(
                    C_frag,
                    C_shared,
                    thread_bindings=tx,
                )

                # Store results from shared memory to global memory
                for i, j in T.Parallel(block_M, block_N):
                    C[by * block_M + i, bx * block_N + j] = C_shared[
                        i // micro_size_x,
                        j // micro_size_y,
                        i % micro_size_x,
                        j % micro_size_y,
                    ]

        return self.maybe_simplify(general_dequant_matmul)

    def _normal_dequant(
        self,
        compressed_weight_local: T.Buffer,
        dequant_weight_local: T.Buffer,
        scale_buffer: T.Buffer,
        zeros_buffer: T.Buffer,
        qzeros_buffer: T.Buffer,
        local_size: int,
        warp_cols: int,
        pid_n: T.Var,
        thread_bindings: T.Var,
        mma_emitter: TensorCoreIntrinEmitterWithLadderTransform,
        ko: T.Var,
        ki: T.Var,
        stride_n: int,
        stride_k: int,
    ):
        num_elems_per_byte = self.num_elems_per_byte
        with_scaling = self.with_scaling
        with_zeros = self.with_zeros
        zeros_mode = self.zeros_mode
        num_bits = self.num_bits
        in_dtype = self.in_dtype
        group_size = self.group_size
        storage_dtype = self.storage_dtype
        storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
        storage_type = str("".join(c for c in storage_dtype if not c.isdigit()))
        micro_size_k = mma_emitter.micro_size_k
        k_inner_stride = micro_size_k // local_size

        @T.macro
        def _normal_dequant_impl(
            compressed_weight_local: T.Buffer,
            dequant_weight_local: T.Buffer,
            scale_buffer: T.Buffer,
            zeros_buffer: T.Buffer,
            qzeros_buffer: T.Buffer,
        ):
            for j in T.serial(warp_cols):
                for v in T.serial(0, local_size):
                    tx = thread_bindings % mma_emitter.WARP_SIZE
                    tz = (thread_bindings // (mma_emitter.WARP_SIZE * mma_emitter.block_row_warps)
                         ) % mma_emitter.block_col_warps
                    vi = (
                        tz * (warp_cols * mma_emitter.WARP_SIZE // k_inner_stride) + j *
                        (mma_emitter.WARP_SIZE // k_inner_stride) + (tx // k_inner_stride))
                    vj = ki * micro_size_k + (tx % k_inner_stride) * local_size + v
                    remaped_i, remaped_j = self.get_param_indices(
                        pid_n * stride_n + vi,
                        ko * stride_k + vj,
                        transform_kind=TransformKind.LDMatrixTransform,
                        in_dtype=in_dtype,
                        matrix_name="B",
                        group_size=group_size,
                    )
                    if not with_scaling:
                        dequant_weight_local[j * local_size + v] = self._decode_func(
                            num_bits,
                            compressed_weight_local[j * local_size // num_elems_per_byte +
                                                    v // num_elems_per_byte],
                            v % num_elems_per_byte,
                            dtype=in_dtype,
                        )
                    elif not with_zeros:
                        dequant_weight_local[j * local_size + v] = (
                            self._decode_func(
                                num_bits,
                                compressed_weight_local[j * local_size // num_elems_per_byte +
                                                        v // num_elems_per_byte],
                                v % num_elems_per_byte,
                                dtype=in_dtype,
                            ) * scale_buffer[remaped_i, remaped_j])
                    elif zeros_mode == "original":
                        dequant_weight_local[j * local_size + v] = (self._decode_func(
                            num_bits,
                            compressed_weight_local[j * local_size // num_elems_per_byte +
                                                    v // num_elems_per_byte],
                            v % num_elems_per_byte,
                            dtype=in_dtype,
                        ) - zeros_buffer[remaped_i, remaped_j]) * scale_buffer[remaped_i, remaped_j]
                    elif zeros_mode == "rescale":
                        dequant_weight_local[j * local_size + v] = (
                            self._decode_func(
                                num_bits,
                                compressed_weight_local[j * local_size // num_elems_per_byte +
                                                        v // num_elems_per_byte],
                                v % num_elems_per_byte,
                                dtype=in_dtype,
                            ) * scale_buffer[remaped_i, remaped_j] -
                            zeros_buffer[remaped_i, remaped_j])
                    elif zeros_mode == "quantized":
                        dequant_qzeros = _tir_packed_to_unsigned_convert(
                            storage_type, storage_nbit)(
                                num_bits,
                                qzeros_buffer[
                                    remaped_i,
                                    remaped_j // num_elems_per_byte,
                                ],
                                (pid_n * stride_n + vi) % num_elems_per_byte,
                                dtype=storage_dtype,
                            )

                        dequant_weight_local[j * local_size + v] = (self._decode_func(
                            num_bits,
                            compressed_weight_local[j * local_size // num_elems_per_byte +
                                                    v // num_elems_per_byte],
                            v % num_elems_per_byte,
                            zero=dequant_qzeros,
                            dtype=in_dtype,
                        )) * scale_buffer[remaped_i, remaped_j]

        return _normal_dequant_impl(
            compressed_weight_local,
            dequant_weight_local,
            scale_buffer,
            zeros_buffer,
            qzeros_buffer,
        )

    def _normal_fast_dequant(
        self,
        compressed_weight_local: T.Buffer,
        dequant_weight_local: T.Buffer,
        scale_buffer: T.Buffer,
        zeros_buffer: T.Buffer,
        qzeros_buffer: T.Buffer,
        func_name: str,
        local_size: int,
        warp_cols: int,
        pid_n: T.Var,
        thread_bindings: T.Var,
        mma_emitter: TensorCoreIntrinEmitterWithLadderTransform,
        ko: T.Var,
        ki: T.Var,
        stride_n: int,
        stride_k: int,
    ):
        num_elems_per_byte = self.num_elems_per_byte
        with_scaling = self.with_scaling
        with_zeros = self.with_zeros
        zeros_mode = self.zeros_mode
        in_dtype = self.in_dtype
        group_size = self.group_size
        micro_size_k = mma_emitter.micro_size_k
        k_inner_stride = micro_size_k // local_size
        grouped_k = scale_buffer.shape[-1]

        @T.macro
        def _normal_fast_dequant_impl(
            compressed_weight_local: T.Buffer,
            dequant_weight_local: T.Buffer,
            scale_buffer: T.Buffer,
            zeros_buffer: T.Buffer,
            qzeros_buffer: T.Buffer,
        ):
            for j in T.serial(warp_cols):
                tx = thread_bindings % mma_emitter.WARP_SIZE
                tz = (thread_bindings // (mma_emitter.WARP_SIZE * mma_emitter.block_row_warps)
                     ) % mma_emitter.block_col_warps
                vi = (
                    tz * (warp_cols * mma_emitter.WARP_SIZE // k_inner_stride) + j *
                    (mma_emitter.WARP_SIZE // k_inner_stride) + (tx // k_inner_stride))
                vj = ki * micro_size_k + (tx % k_inner_stride) * local_size
                remapped_i, remapped_j = self.get_param_indices(
                    pid_n * stride_n + vi,
                    ko * stride_k + vj,
                    transform_kind=TransformKind.LDMatrixTransform,
                    in_dtype=in_dtype,
                    matrix_name="B",
                    group_size=group_size,
                )
                if not with_scaling:
                    T.call_extern(
                        func_name,
                        T.address_of(compressed_weight_local[j * local_size // num_elems_per_byte]),
                        T.address_of(dequant_weight_local[j * local_size]),
                        dtype=in_dtype,
                    )
                elif not with_zeros:
                    # Scaling only
                    T.call_extern(
                        func_name,
                        T.address_of(compressed_weight_local[j * local_size // num_elems_per_byte]),
                        T.address_of(dequant_weight_local[j * local_size]),
                        T.address_of(scale_buffer[remapped_i, remapped_j]),
                        local_size * grouped_k,
                        local_size,
                        dtype=in_dtype,
                    )
                elif zeros_mode in ["original", "rescale"]:
                    T.call_extern(
                        func_name,
                        T.address_of(compressed_weight_local[j * local_size // num_elems_per_byte]),
                        T.address_of(dequant_weight_local[j * local_size]),
                        T.address_of(scale_buffer[remapped_i, remapped_j]),
                        T.address_of(zeros_buffer[remapped_i, remapped_j]),
                        local_size * grouped_k,
                        local_size,
                        dtype=in_dtype,
                    )
                # TODO: Implement quantized zeros

        return _normal_fast_dequant_impl(
            compressed_weight_local,
            dequant_weight_local,
            scale_buffer,
            zeros_buffer,
            qzeros_buffer,
        )

    def get_param_indices(
        self,
        rl,
        rr,
        l=16,
        r=16,
        transform_kind=TransformKind.LDMatrixTransform,  # noqa: E741
        trans=True,
        in_dtype="float16",
        matrix_name="B",
        group_size=1,
    ):  # noqa: E741
        intra_index_map, _ = get_propagate_map(trans=trans, dtype=in_dtype, matrix_name=matrix_name)

        ladder_stage3_index_map, ladder_stage3_inverse_index_map = (
            get_ladder_stage3_map(dtype=in_dtype))

        # assume the param layout is n, k

        warp_i, warp_j = rl % l, rr % r

        spatial_i, spatial_j = rl // l, rr // r

        # If is stage3 ladder transform
        if transform_kind > 2:
            warp_i, warp_j = ladder_stage3_inverse_index_map.map_indices([warp_i, warp_j])

        warp_i, warp_j = intra_index_map.map_indices([warp_i, warp_j])
        new_indices = (
            spatial_i * l + warp_i,
            (spatial_j * r + warp_j) // group_size,
        )

        return new_indices

    def __post_init__(self):
        # Legalize group_size
        if self.with_scaling and self.group_size == -1:
            object.__setattr__(self, "group_size", self.K)
