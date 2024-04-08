# Operator Configuration Documentation

The Operator Configuration Documentation provides detailed information about the configurations used for matrix multiplication (matmul) operations in a specific environment, such as deep learning frameworks or custom hardware accelerators. This document outlines two data classes, MatmulWeightOnlyDequantizeConfig and MatmulConfig, which are used to configure the matmul operations with and without weight dequantization, respectively.

## Table of Contents
- [Matrix Multiplication Configuration (`MatmulConfig`)](#matrix-multiplication-configuration-matmulconfig)
- [Matrix Multiplication Weight Only Dequantize Configuration (`MatmulWeightOnlyDequantizeConfig`)](#matrix-multiplication-weight-only-dequantize-configuration-matmulweightonlydequantizeconfig)

## Matrix Multiplication Configuration (`MatmulConfig`)

This configuration class defines the general settings for matrix multiplication operations.

### Fields

The fields in `MatmulConfig` largely overlap with those in `MatmulWeightOnlyDequantizeConfig`, excluding those specific to dequantization.

## Common Fields

- **M, N, K**: Matrix dimensions.
- **in_dtype, out_dtype, accum_dtype**: Data types for the operation.
- **with_bias**: Enables the inclusion of a bias vector in the operation.
- **layout**: Determines the layout of the matrices.
- **propagate_a, propagate_b**: Controls weight transformation.

## Matrix Multiplication Weight Only Dequantize Configuration (`MatmulWeightOnlyDequantizeConfig`)

This configuration class is designed for specifying the parameters related to the dequantization of weights in matrix multiplication operations.

### Fields

- **M, N, K**: Dimensions of the matrices involved in the operation. `M` can be an integer or a tuple, specifying the dimensions of the output matrix, while `N` and `K` represent the common dimension in matrix multiplication and the dimension of the input matrix, respectively.
- **in_dtype, out_dtype, accum_dtype**: Data types for input, output, and accumulation during the operation. Defaults to `"float16"`.
- **bit**: The bit width used for quantization. Default is `4`.
- **storage_dtype**: Data type used for storing the quantized weights. Default is `"int8"`.
- **source_format**: Specifies the format of the source data for dequantization, with options including `"int"`, `"uint"`, `"fp"`, and `"af"`. Default is `"int"`.
- **with_scaling, with_zeros, fast_decoding, with_bias**: Boolean flags to enable specific features in the dequantization process.
- **group_size**: Specifies the group size for grouped operations. Default is `-1`.
- **propagate_a, propagate_b**: Specifies whether and how weight transformation is applied to matrix A and matrix B, using the `TransformKind` enumeration.
- **layout**: The layout of matrices involved in the operation. Default is `"nt"`.
- **zeros_mode**: Specifies the zero-point adjustment method during dequantization. Options are `"original"`, `"rescale"`, and `"quantized"`. Default is `"original"`.
    - **`"original"`**: In this mode, the dequantization formula is adjusted to subtract the zero-point before scaling. The formula used is `target = (dequantize_weight - zero_point) * scale`. This method is straightforward and aligns closely with many hardware implementations.

    - **`"rescale"`**: This option modifies the dequantization process by applying the scale factor directly to the dequantized weight and then subtracting the zero-point. The formula becomes `target = dequantize_weight * scale - zero_point`. This can be useful for operations where scaling prior to zero-point correction aligns better with the computational workflow or hardware optimizations.

    - **`"quantized"`**: In the quantized mode, the zero-point adjustment is applied after both the dequantization and an additional dequantization of the zero values themselves. The formula here is `target = (dequantize_weight - dequantize_zeros) * scale`, where `dequantize_zeros` represents the dequantized representation of zero values. This approach is particularly useful for aligning the dequantization process with specific computational frameworks or hardware that prefers explicit handling of zero values.
