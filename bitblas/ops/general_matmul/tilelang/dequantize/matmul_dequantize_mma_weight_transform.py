# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm as tvm
from bitblas import tilelang as tilelang
from tvm import DataType
import tilelang.language as T
from typing import Optional, List
from tilelang.intrinsics.utils import (
    get_mma_micro_size,
    make_mma_swizzle_layout as make_swizzle_layout,
    index_to_coordinates,
)
from bitblas.base.arch import TileDevice
from bitblas.base.roller.hint import Hint
from .matmul_dequantize_mma import MatmulDequantizeMMAScheduler
from bitblas.tl.mma_macro_generator import (
    TensorCoreIntrinEmitterWithLadderTransform,
    INT4TensorCoreIntrinEmitterWithLadderTransform,
)
from bitblas.base.operator_common import TransformKind  # noqa: F401
from dataclasses import dataclass
from bitblas.base.utils import get_roller_hints_from_func
from bitblas.ops.general_matmul.tirscript import (
    matmul_dequantize_select_implementation,)
from bitblas.quantization import (
    _tir_packed_to_unsigned_convert,)
from bitblas.gpu.matmul_analysis import (
    get_propagate_map,
    get_ladder_stage3_map,
)

# GPU warp configuration for NVIDIA GPUs
warp_size = 32


@dataclass
class MatmulDequantizeMMAWeightPropagationScheduler(MatmulDequantizeMMAScheduler):

    # force set default weight transform kind to LDMatrixTransform
    weight_transform_kind: TransformKind = TransformKind.LDMatrixTransform

    class TLHint(MatmulDequantizeMMAScheduler.TLHint):
        hint_type: str = "MatmulDequantizeMMAWeightPropagationScheduler"

    def apply_config(
        self,
        block_row_warps: Optional[int] = None,
        block_col_warps: Optional[int] = None,
        warp_row_tiles: Optional[int] = None,
        warp_col_tiles: Optional[int] = None,
        chunk: Optional[int] = None,
        num_stages: Optional[int] = None,
        enable_rasterization: bool = False,
        split_k_factor: Optional[int] = None,
    ):
        assert block_row_warps is not None, "block_row_warps is required"
        assert block_col_warps is not None, "block_col_warps is required"
        assert warp_row_tiles is not None, "warp_row_tiles is required"
        assert warp_col_tiles is not None, "warp_col_tiles is required"
        assert chunk is not None, "chunk is required"
        assert num_stages is not None, "num_stages is required"

        M = self.maybe_dynamic(self.M, "m")
        N, K = self.N, self.K
        assert isinstance(N, int) and isinstance(K, int), "Do not support dynamic N and K Currently"
        trans_A, trans_B = self.trans_A, self.trans_B
        input_transform_kind = self.input_transform_kind
        weight_transform_kind = self.weight_transform_kind

        assert trans_A is False, "Dequantize only implement for trans_A=False currently"
        assert trans_B is True, "Dequantize only implement for trans_B=TRue currently"
        assert (
            weight_transform_kind == TransformKind.LDMatrixTransform
        ), f"Dequantize only implement for LDMatrixTransform currently, got {weight_transform_kind}"

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

        fragement_size_a = (micro_size_x * micro_size_k) // warp_size
        fragement_size_b = (micro_size_y * micro_size_k) // warp_size
        fragement_size_c = (micro_size_x * micro_size_y) // warp_size
        warp_rows = warp_row_tiles // micro_size_x
        warp_cols = warp_col_tiles // micro_size_y

        fast_decoding = self.fast_decoding
        with_bias = self.with_bias

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

        B_shape = (
            N // micro_size_y,
            K // micro_size_k,
            micro_size_y,
            micro_size_k // num_elems_per_byte,
        )
        LUT_shape = (1 << num_bits,)
        Scale_shape = (N, K // group_size)
        Zeros_shape = (N, K // group_size)
        Qzeros_shape = ((K // group_size), N // storage_nbit * num_bits)
        C_shape = (M, N)
        Bias_shape = (N,)

        is_a_smooth = self.is_a_smooth
        is_b_smooth = self.is_b_smooth

        if is_a_smooth:
            A_shape = (M // micro_size_x, K // micro_size_k, micro_size_x, micro_size_k)
            A_shared_shape = (
                block_M // micro_size_x,
                block_K // micro_size_k,
                micro_size_x,
                micro_size_k,
            )
        else:
            A_shape = (M, K)
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

        shared_scope = "shared.dyn"

        import_source: Optional[str] = None
        func_name: str = ""
        if fast_decoding is True:
            # Lazy import to save the startup time
            # as intrin registry may take a while to load
            from bitblas.gpu.intrin.lop3 import get_lop3_intrin_group
            lop3_intrin_info = get_lop3_intrin_group(
                out_dtype=in_dtype,
                source_format=source_format,
                source_bit=num_bits,
                storage_dtype=storage_dtype,
                with_scaling=self.with_scaling,
                with_zeros=self.with_zeros,
                zeros_mode=self.zeros_mode,
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
            transform_kind_a=input_transform_kind,
            transform_kind_b=weight_transform_kind,
            num_elems_per_byte=num_elems_per_byte,
        )

        splitK = K // split_k_factor
        enable_split_k = split_k_factor > 1

        def check_require_cache():
            conditions = [False]
            conditions.append(self.check_require_cache())
            conditions.append(enable_split_k)
            return any(conditions)

        cache_write_required = check_require_cache()

        @T.prim_func
        def general_dequant_matmul(
                A: T.Buffer(A_shape, in_dtype),
                B: T.Buffer(B_shape, storage_dtype),
                LUT: T.Buffer(LUT_shape, in_dtype),
                Scale: T.Buffer(Scale_shape, in_dtype),
                Qzeros: T.Buffer(Qzeros_shape, storage_dtype),
                Zeros: T.Buffer(Zeros_shape, in_dtype),
                Bias: T.Buffer(Bias_shape, in_dtype),
                C: T.Buffer(C_shape, out_dtype),
        ):
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k_factor,
                    threads=threads) as (bx, by, bz):
                A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
                B_shared = T.alloc_shared(B_shared_shape, storage_dtype, scope=shared_scope)
                C_shared = T.alloc_shared(C_shared_shape, out_dtype, scope=shared_scope)

                A_frag = T.alloc_local((warp_rows * fragement_size_a), in_dtype)
                B_frag = T.alloc_local((warp_cols * fragement_size_b // num_elems_per_byte),
                                       storage_dtype)
                B_dequantize_frag = T.alloc_local((warp_cols * fragement_size_b), in_dtype)
                C_frag = T.alloc_local((warp_rows * warp_cols * fragement_size_c), accum_dtype)

                tx = T.thread_binding(0, threads, thread="threadIdx.x")

                T.annotate_layout({
                    A_shared: make_swizzle_layout(A_shared, is_a_smooth),
                    B_shared: make_swizzle_layout(B_shared, is_b_smooth),
                })

                T.use_swizzle(10, enable=enable_rasterization)

                T.import_source(import_source)

                T.clear(C_frag)

                if enable_split_k:  # noqa: SIM102
                    if bz == 0:
                        for i, j in T.Parallel(block_M, block_N):
                            m, n = by * block_M + i, bx * block_N + j
                            C[m, n] = T.cast(0, out_dtype)

                for ko in T.Pipelined(T.ceildiv(splitK, block_K), num_stages=num_stages):

                    if is_a_smooth:
                        for i, k, ii, kk in T.Parallel(
                                block_M // micro_size_x,
                                block_K // micro_size_k,
                                micro_size_x,
                                micro_size_k,
                        ):
                            A_shared[i, k, ii,
                                     kk] = A[by * (block_M // micro_size_x) + i,
                                             bz * splitK + ko * (block_K // micro_size_k) + k, ii,
                                             kk]
                    else:
                        T.copy(A[by * block_M, bz * splitK + ko * block_K], A_shared)

                    for j, k, jj, kk in T.Parallel(
                            block_N // micro_size_y,
                            block_K // micro_size_k,
                            micro_size_y,
                        (micro_size_k // num_elems_per_byte),
                    ):
                        B_shared[j, k, jj, kk] = B[
                            bx * (block_N // micro_size_y) + j,
                            bz * (splitK // micro_size_k) + ko * (block_K // micro_size_k) + k,
                            jj,
                            kk,
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
                                bz * T.ceildiv(splitK, block_K) + ko,
                                ki,
                                block_N,
                                block_K,
                            )
                        else:
                            self._normal_dequant(
                                B_frag,
                                B_dequantize_frag,
                                LUT,
                                Scale,
                                Zeros,
                                Qzeros,
                                local_size,
                                warp_cols,
                                bx,
                                tx,
                                mma_emitter,
                                bz * T.ceildiv(splitK, block_K) + ko,
                                ki,
                                block_N,
                                block_K,
                            )

                        # Matrix multiplication on fragments
                        mma_emitter.mma(A_frag, B_dequantize_frag, C_frag)

                if cache_write_required:
                    # Store the result back to C shared memory
                    mma_emitter.stmatrix(
                        C_frag,
                        C_shared,
                        thread_bindings=tx,
                    )

                    if with_bias:  # noqa: SIM102
                        if bz == 0:  # as bz is the k-dim, otherwise, bias will be added multiple times
                            for i, j in T.Parallel(block_M, block_N):
                                C_shared[
                                    i // micro_size_x,
                                    j // micro_size_y,
                                    i % micro_size_x,
                                    j % micro_size_y,
                                ] += Bias[bx * block_N + j]

                # Store results from shared memory to global memory
                    if enable_split_k:
                        # only for fp16
                        if DataType(out_dtype).bits == 16:
                            for i, j in T.Parallel(block_M, block_N // 2):
                                m, n = by * block_M + i, bx * block_N + j * 2
                                T.atomic_addx2(
                                    C[m, n], C_shared[
                                        i // micro_size_x,
                                        (j * 2) // micro_size_y,
                                        i % micro_size_x,
                                        (j * 2) % micro_size_y,
                                    ])
                        else:
                            for i, j in T.Parallel(block_M, block_N):
                                m, n = by * block_M + i, bx * block_N + j
                                C[m, n] = C_shared[
                                    i // micro_size_x,
                                    j // micro_size_y,
                                    i % micro_size_x,
                                    j % micro_size_y,
                                ]
                    else:
                        for i, j in T.Parallel(block_M, block_N):
                            m, n = by * block_M + i, bx * block_N + j
                            C[m, n] = C_shared[
                                i // micro_size_x,
                                j // micro_size_y,
                                i % micro_size_x,
                                j % micro_size_y,
                            ]

                else:
                    mma_emitter.stmatrix(
                        C_frag,
                        C,
                        thread_bindings=tx,
                        pid_m=by,
                        pid_n=bx,
                    )

        return self.post_process(general_dequant_matmul)

    def _normal_dequant(
        self,
        compressed_weight_local: T.Buffer,
        dequant_weight_local: T.Buffer,
        lut_buffer: T.Buffer,
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
        source_format = self.source_format
        is_lut = source_format == "nf"
        storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
        storage_type = str("".join(c for c in storage_dtype if not c.isdigit()))
        micro_size_k = mma_emitter.micro_size_k
        k_inner_stride = micro_size_k // local_size

        @T.macro
        def _normal_dequant_impl(
            compressed_weight_local: T.Buffer,
            dequant_weight_local: T.Buffer,
            lut_buffer: T.Buffer,
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
                    if is_lut:
                        index = _tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
                            num_bits,
                            compressed_weight_local[j * local_size // num_elems_per_byte +
                                                    v // num_elems_per_byte],
                            v % num_elems_per_byte,
                            "int32"  # default index dtype
                        )
                        dequant_weight_local[j * local_size + v] = lut_buffer[index]
                    else:
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
                            ) - zeros_buffer[remaped_i, remaped_j]) * scale_buffer[remaped_i,
                                                                                   remaped_j]
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
            lut_buffer,
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
                qzeros_remapped_i, qzeros_remapped_j = remapped_j, remapped_i

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
                else:
                    T.call_extern(
                        func_name,
                        T.address_of(compressed_weight_local[j * local_size // num_elems_per_byte]),
                        T.address_of(dequant_weight_local[j * local_size]),
                        T.address_of(scale_buffer[remapped_i, remapped_j]),
                        T.address_of(qzeros_buffer[
                            qzeros_remapped_i,
                            (qzeros_remapped_j // num_elems_per_byte),
                        ]),
                        local_size * grouped_k,
                        local_size // num_elems_per_byte,
                        qzeros_remapped_j % num_elems_per_byte,
                        local_size,
                        dtype=in_dtype,
                    )

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

    @property
    def is_a_smooth(self):
        return self.input_transform_kind > TransformKind.NonTransform

    @property
    def is_b_smooth(self):
        return self.weight_transform_kind > TransformKind.NonTransform


@dataclass
class MatmulINT4DequantizeMMAWeightPropagationScheduler(
        MatmulDequantizeMMAWeightPropagationScheduler):

    class TLHint(MatmulDequantizeMMAWeightPropagationScheduler.TLHint):
        hint_type: str = "MatmulINT4DequantizeMMAWeightPropagationScheduler"

    def get_roller_configs(self, arch: TileDevice = None, topk: int = 10):
        layout = f"{'t' if self.trans_A else 'n'}{'t' if self.trans_B else 'n'}"
        M = self.M
        K = self.K // 2  # 2xint4 should be packed into one single int8
        storage_dtype = "int8"
        num_bits = self.num_bits * 2

        # This is a hack to utilize tensor core
        if isinstance(M, int) and M < 16:
            M = 16

        # INT4XINT2 is equal to int8xint4 with reduced shape
        # Simple TIR Compute Expression
        ir_module = matmul_dequantize_select_implementation(
            M=M,
            N=self.N,
            K=K,
            in_dtype=storage_dtype,
            out_dtype=self.out_dtype,
            accum_dtype=self.accum_dtype,
            layout=layout,
            bit=num_bits,
            storage_dtype=self.storage_dtype,
            source_format=self.source_format,
            with_scaling=self.with_scaling,
            with_zeros=self.with_zeros,
            group_size=self.group_size,
            fast_decoding=self.fast_decoding,
            with_bias=self.with_bias,
            zeros_mode=self.zeros_mode)

        roller_hints = get_roller_hints_from_func(
            ir_module,
            arch,
            topk,
            tensorcore_only=True,
            allow_gemv=True,
        )

        if roller_hints is None:
            raise ValueError("No Roller Hints Found for TensorCore Scheduling")

        for hint in roller_hints:
            print(hint)

        def serialize_hints_to_configs(hints: List[Hint]):
            configs = []
            for hint in hints:
                config = self.TLHint.from_roller_hint(hint)
                configs.append(config)
            return configs

        return serialize_hints_to_configs(roller_hints)

    def apply_config(
        self,
        block_row_warps: Optional[int] = None,
        block_col_warps: Optional[int] = None,
        warp_row_tiles: Optional[int] = None,
        warp_col_tiles: Optional[int] = None,
        chunk: Optional[int] = None,
        num_stages: Optional[int] = None,
        enable_rasterization: bool = False,
        split_k_factor: Optional[int] = None,
    ):
        assert block_row_warps is not None, "block_row_warps is required"
        assert block_col_warps is not None, "block_col_warps is required"
        assert warp_row_tiles is not None, "warp_row_tiles is required"
        assert warp_col_tiles is not None, "warp_col_tiles is required"
        assert chunk is not None, "chunk is required"
        assert num_stages is not None, "num_stages is required"
        # unused variable
        split_k_factor = split_k_factor

        M = self.maybe_dynamic(self.M, "m")
        N, K = self.N, self.K
        assert isinstance(N, int) and isinstance(K, int), "Do not support dynamic N and K Currently"
        K = K // 2  # 2xint4 should be packed into one single int8

        trans_A, trans_B = self.trans_A, self.trans_B
        weight_transform_kind = self.weight_transform_kind

        assert trans_A is False, "Dequantize only implement for trans_A=False currently"
        assert trans_B is True, "Dequantize only implement for trans_B=TRue currently"
        assert (
            weight_transform_kind == TransformKind.LDMatrixTransform
        ), f"Dequantize only implement for LDMatrixTransform currently, got {weight_transform_kind}"

        in_dtype, out_dtype, accum_dtype = (
            self.in_dtype,
            self.out_dtype,
            self.accum_dtype,
        )
        assert in_dtype == "int4", "Only support int4 input"
        assert accum_dtype == "int32", "Only support int32 accumulation"
        storage_dtype = self.storage_dtype

        # Calculate the micro size per warp using a helper function
        micro_size_x, micro_size_y, micro_size_k = get_mma_micro_size(storage_dtype)

        block_M = block_row_warps * warp_row_tiles
        block_N = block_col_warps * warp_col_tiles
        block_K = chunk
        threads = warp_size * (block_row_warps * block_col_warps)

        fragement_size_a = (micro_size_x * micro_size_k) // warp_size
        fragement_size_b = (micro_size_y * micro_size_k) // warp_size
        fragement_size_c = (micro_size_x * micro_size_y) // warp_size
        warp_rows = warp_row_tiles // micro_size_x
        warp_cols = warp_col_tiles // micro_size_y

        fast_decoding = self.fast_decoding

        num_bits = self.num_bits
        source_format = self.source_format
        num_elems_per_byte = self.num_elems_per_byte

        MAX_TRANSACTION_SIZE_IN_BITS = 128
        local_size = MAX_TRANSACTION_SIZE_IN_BITS // DataType(storage_dtype).bits
        local_size_compressed = local_size // num_elems_per_byte

        group_size = self.group_size
        if group_size == -1:
            group_size = K

        is_a_smooth = self.is_a_smooth
        is_b_smooth = self.is_b_smooth

        if is_a_smooth:
            A_shape = (M // micro_size_x, K // micro_size_k, micro_size_x, micro_size_k)
            A_shared_shape = (
                block_M // micro_size_x,
                block_K // micro_size_k,
                micro_size_x,
                micro_size_k,
            )
        else:
            A_shape = (M, K)
            A_shared_shape = (block_M, block_K)

        B_shape = (
            N // micro_size_y,
            K // micro_size_k,
            micro_size_y,
            micro_size_k // num_elems_per_byte,
        )
        B_dequantize_shared_shape = (
            block_N // micro_size_y,
            block_K // micro_size_k,
            micro_size_y,
            micro_size_k,
        )
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
            # Lazy import to save the startup time
            # as intrin registry may take a while to load
            from bitblas.gpu.intrin.lop3 import get_lop3_intrin_group

            lop3_intrin_info = get_lop3_intrin_group(
                out_dtype=in_dtype,
                source_format=source_format,
                source_bit=num_bits,
                storage_dtype=storage_dtype,
                with_scaling=self.with_scaling,
                with_zeros=self.with_zeros,
                zeros_mode=self.zeros_mode,
                storage_scope="warp",  # to get the ladder transform lop3 intrin
            )
            import_source = lop3_intrin_info["c_source"]
            func_name = lop3_intrin_info["func_name"]
            assert import_source is not None, "lop3_intrin_info is not found"
            assert func_name is not None, "lop3_intrin_info is not found"
            import_source = self.common_header + import_source

        # Configure the tensor core intrinsic emitter with ladder transform
        mma_emitter = INT4TensorCoreIntrinEmitterWithLadderTransform(
            a_dtype=storage_dtype,
            b_dtype=storage_dtype,
            accum_dtype=accum_dtype,
            a_transposed=trans_A,
            b_transposed=trans_B,
            block_row_warps=block_row_warps,
            block_col_warps=block_col_warps,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=chunk,
            transform_kind_b=weight_transform_kind,
        )

        vec_load_qb = 16
        if block_N * block_K // num_elems_per_byte // threads < vec_load_qb:
            vec_load_qb = block_N * block_K // num_elems_per_byte // threads

        @T.prim_func
        def general_dequant_matmul(
                A: T.Buffer(A_shape, storage_dtype),
                B: T.Buffer(B_shape, storage_dtype),
                C: T.Buffer((M, N), out_dtype),
        ):
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared(A_shared_shape, storage_dtype)
                B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
                B_dequantize_shared = T.alloc_shared(B_dequantize_shared_shape, storage_dtype)
                C_shared = T.alloc_shared(C_shared_shape, out_dtype)

                A_frag = T.alloc_local((warp_rows * fragement_size_a), storage_dtype)
                B_frag = T.alloc_local((warp_cols * fragement_size_b), storage_dtype)
                C_frag = T.alloc_local((warp_rows * warp_cols * fragement_size_c), accum_dtype)
                B_local = T.alloc_local([local_size_compressed], storage_dtype)
                B_dequantize_local = T.alloc_local([local_size], storage_dtype)

                tx = T.thread_binding(0, threads, thread="threadIdx.x")

                T.annotate_layout({
                    A_shared: make_swizzle_layout(A_shared, is_a_smooth),
                    B_shared: make_swizzle_layout(B_shared, is_b_smooth),
                })

                T.use_swizzle(10, enable=enable_rasterization)

                T.import_source(import_source)

                T.clear(C_frag)

                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):

                    if is_a_smooth:
                        for i, k, ii, kk in T.Parallel(
                                block_M // micro_size_x,
                                block_K // micro_size_k,
                                micro_size_x,
                                micro_size_k,
                        ):
                            A_shared[i, k, ii, kk] = A[by * (block_M // micro_size_x) + i,
                                                       ko * (block_K // micro_size_k) + k, ii, kk]
                    else:
                        T.copy(A[by * block_M, ko * block_K], A_shared)

                    # Load B into shared memory
                    # TODO(lei): Layout Inference Pass is not efficient to handle the four dims int8 load
                    for i in T.serial(block_N * block_K // num_elems_per_byte //
                                      (threads * vec_load_qb)):
                        for v in T.vectorized(0, vec_load_qb):
                            idx = i * threads * vec_load_qb + threads * vec_load_qb + tx * vec_load_qb + v
                            vj, vk, vjj, vkk = index_to_coordinates(idx, B_shared_shape)
                            B_shared[vj, vk, vjj,
                                     vkk] = B[bx * (block_N // micro_size_y) + vj,
                                              ko * (block_K // micro_size_k) + vk, vjj, vkk]

                    for i in T.serial(block_N * block_K // num_elems_per_byte //
                                      (threads * local_size_compressed)):
                        for v in T.vectorized(0, local_size_compressed):
                            index = (
                                i * threads * local_size_compressed + tx * local_size_compressed +
                                v)
                            vi, vj, vii, vjj = index_to_coordinates(index, B_shared_shape)
                            B_local[v] = B_shared[vi, vj, vii, vjj]

                        if fast_decoding:
                            # Simulated dequantization
                            T.call_extern('handle', func_name, T.address_of(B_local[0]),
                                          T.address_of(B_dequantize_local[0]), 32)
                        else:
                            for v in T.serial(0, local_size):
                                int2x2_value = (B_local[v // 2] >> ((v % 2) * 4)) & 0x0F

                                int4_0 = (int2x2_value >> 0) & 0x03
                                int4_1 = (int2x2_value >> 2) & 0x03

                                B_dequantize_local[v] = (int4_1 << 4) | int4_0

                        for v in T.vectorized(0, local_size):
                            index = i * threads * local_size + tx * local_size + v
                            vi, vj, vii, vjj = index_to_coordinates(index,
                                                                    B_dequantize_shared_shape)
                            B_dequantize_shared[vi, vj, vii, vjj] = B_dequantize_local[v]

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
                            B_dequantize_shared,
                            ki,
                            thread_bindings=tx,
                        )

                        # Matrix multiplication on fragments
                        mma_emitter.mma(A_frag, B_frag, C_frag)

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

        return self.post_process(general_dequant_matmul)

    @property
    def num_elems_per_byte(self):
        # force value for int4
        storage_nbit = 4
        num_bits = self.num_bits
        return storage_nbit // num_bits
