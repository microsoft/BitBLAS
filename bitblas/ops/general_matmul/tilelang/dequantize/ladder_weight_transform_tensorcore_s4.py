# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from bitblas import tvm as tvm
from tvm import DataType
import tvm.tl.language as T
from typing import Optional, List
from bitblas.tl.utils import (
    get_mma_micro_size,  # noqa: F401
    make_swizzle_layout,  # noqa: F401
    index_to_coordinates,  # noqa: F401
)
from bitblas.base.arch import TileDevice
from bitblas.base.roller.hint import Hint
from bitblas.tl.macro_generator import (
    INT4TensorCoreIntrinEmitterWithLadderTransform,  # noqa: F401
)
from bitblas.ops.common import TransformKind  # noqa: F401
from dataclasses import dataclass
from bitblas.base.utils import get_roller_hints_from_func
from bitblas.gpu.intrin.lop3 import get_lop3_intrin_group
from bitblas.ops.general_matmul.tirscript import (
    matmul_dequantize_select_implementation,)
from bitblas.ops.general_matmul.tilelang.dequantize.ladder_weight_transform_tensorcore import (
    MatmulDequantizeWeightPropagationScheduler,)

# GPU warp configuration for NVIDIA GPUs
warp_size = 32


@dataclass
class MatmulINT4DequantizeWeightPropagationScheduler(MatmulDequantizeWeightPropagationScheduler):

    def get_roller_configs(self, arch: TileDevice = None, topk: int = 10):
        layout = f"{'t' if self.trans_A else 'n'}{'t' if self.trans_B else 'n'}"
        K = self.K // 2  # 2xint4 should be packed into one single int8
        storage_dtype = "int8"
        num_bits = self.num_bits * 2
        # INT4XINT2 is equal to int8xint4 with reduced shape
        # Simple TIR Compute Expression
        ir_module = matmul_dequantize_select_implementation(
            M=self.M,
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

        def serialze_hints_to_configs(hints: List[Hint]):
            configs = []
            for hint in hints:
                config = self.TLHint.from_roller_hint(hint)
                configs.append(config)
            return configs

        return serialze_hints_to_configs(roller_hints)

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
        K = K // 2  # 2xint4 should be packed into one single int8

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

        A_shape = (M, K)
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
                out_dtype=in_dtype,
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
                    A_shared: make_swizzle_layout(A_shared),
                })

                T.use_swizzle(10, enable=enable_rasterization)

                T.import_source(import_source)

                T.clear(C_frag)

                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):

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

        return self.maybe_simplify(general_dequant_matmul)

    @property
    def num_elems_per_byte(self):
        # force value for int4
        storage_nbit = 4
        num_bits = self.num_bits
        return storage_nbit // num_bits

    def __post_init__(self):
        # Legalize group_size
        if self.with_scaling and self.group_size == -1:
            object.__setattr__(self, "group_size", self.K)
