## Matmul

`Matmul` is an operator class that performs matrix multiplication, supporting various optimizations and quantization strategies.
### MatmulConfig:

`MatmulConfig` is a configuration class for the `Matmul` operator, specifying the matrix multiplication operation's parameters and behaviors.

### Parameters:

- **M** *(Union[int, Tuple[int]])*: The size of the first dimension of the matrix A, or a range of sizes if dynamic shape support is needed. Can be an integer or a tuple representing the dynamic range.
    - If `int`, the bitblas matmul will generate a static shape kernel, which can only be used for the input shape of the specified value.
    - If `List[int]`, the bitblas matmul will generate a dynamic shape kernel, which can be used for the input shape of the specified values. While the input shape represents the target optimized range.
    - If `None`, the bitblas matmul will use a default value [1, 16, 32, 64, 128, 256, 512, 1024].
- **N** *(int)*: The size of the second dimension of matrix B and the output matrix.
- **K** *(int)*: The common dimension of matrices A and B.
- **A_dtype** *(str, default='float16')*: The data type of matrix A.
    - Choices: `'float16'`, `'int8'`.
- **W_dtype** *(str, default='float16')*: The data type of matrix B. Also acts as a wrapper for source_format and bit.
    - Choices: `'float16'`, `'int8'`, `'int4'`, `'int2'`, `'int1'`, `'fp4_e2m1'`, `'af4'`.
- **out_dtype** *(str, default='float16')*: The data type of the output matrix.
    - Choices: `'float16'`, `'int8'`, `'int32'`.
- **accum_dtype** *(str, default='float16')*: The data type used for accumulation during the matrix multiplication.
    - Choices: `'float16'`, `'int32'`.
- **layout** *(Literal['nn', 'nt', 'tn', 'tt'], default='nt')*: The layout of the matrix multiplication operation.
    - `'nn'`: Both matrices are non-transposed.
    - `'nt'`: Matrix A is non-transposed, and matrix B is transposed.
    - `'tn'`: Matrix A is transposed, and matrix B is non-transposed.
    - `'tt'`: Both matrices are transposed.
- **with_bias** *(bool, default=False)*: Indicates whether a bias vector is added to the output.
- **group_size** *(int, default=-1)*: The group size for quantization, -1 indicates no grouping.
- **with_scaling** *(bool, default=False)*: Indicates whether scaling is applied during quantization.
- **with_zeros** *(bool, default=False)*: Indicates whether zero optimization is applied.
- **zeros_mode** *(Literal['original', 'rescale', 'quantized'], default='original')*: The mode of zero optimization.
    - Choices: `None`, `'original'`, `'rescale'`, `'quantized'`.
        - `'original'`: Subtract zero-point before scaling. Formula: `target = (dequantize_weight - zero_point) * scale`. where `zero_point` has the same datatype with scale.
        - `'rescale'`: Apply scale factor directly to dequantized weight and then subtract zero-point. Formula: `target = dequantize_weight * scale - zero_point`.
        - `'quantized'`: Apply zero-point adjustment after dequantization and additional dequantization of zero values. Formula: `target = (dequantize_weight - dequantize_qzeros) * scale`, where `dequantize_zeros` represents the dequantized representation of zero values, which can be adapted to qzeros params.
- **storage_dtype** *(str, default='int8')*: The storage data type for quantized weights.
    - Choices: `'int32'`, `'int8'`. (we prefer using `'int8'` as the storage data type)
- **fast_decoding** *(bool, default=True)*: Enables fast decoding by default.
    - If `True`, the module will use fast decoding for INT4/2/1 quantization, which will introduce a weight transformation, and this transformation will be applied to the weight automalically when you call `transform_weight`. More details about this transformation is coming soon.
- **propagate_a** *(TransformKind, default=NonTransform)*: Transformation kind for matrix A. Which comes from Ladder, detailed document is coming soon. By default, it is `NonTransform`, which means do not enable transformation.
    - Choices: `NonTransform`, `InterWarpTransform`, `IntraWarpTransform`.
    - Choices: `False`, `True`. Where `False` is equivalent to `NonTransform`. `True` is equivalent to `IntraWarpTransform`.
    - Choices: `0`, `1`, `2`. Where `0` is equivalent to `NonTransform`. `1` is equivalent to `InterWarpTransform`. `2` is equivalent to `IntraWarpTransform`.
- **propagate_b** *(TransformKind, default=NonTransform)*: Transformation kind for matrix B. Which comes from Ladder, detailed document is coming soon. By default, it is `NonTransform`, which means do not enable transformation.
    - Choices: `NonTransform`, `InterWarpTransform`, `IntraWarpTransform`.
    - Choices: `False`, `True`. Where `False` is equivalent to `NonTransform`. `True` is equivalent to `IntraWarpTransform`.
    - Choices: `0`, `1`, `2`. Where `0` is equivalent to `NonTransform`. `1` is equivalent to `InterWarpTransform`. `2` is equivalent to `IntraWarpTransform`.

###  Initialization:

```python
Matmul(config: MatmulConfig, name: str = "matmul", target: Optional[Union[str, Target]] = None, enable_tuning: bool = True)
```

- **config** *(MatmulConfig)*: The configuration for the matrix multiplication operation.
- **name** *(str, default='matmul')*: The name of the operator. Currently this parameter is not used.
- **target** *(Optional[Union[str, Target]], default=None)*: The compilation target. If `None`, the target is auto-detected.
- **enable_tuning** *(bool, default=True)*: If `True`, enables hardware-aware tuning.

###  Methods:

#### `forward(*args) -> Any`

Executes the matrix multiplication operation with the given arguments.

- **args**: Input tensors for the operation.

if arguemnts do not contain the output tensor, the method will create a new tensor for the output. Finally, the method returns the output tensor.

#### `transform_weight(weight, scale=None, zeros=None, bias=None)`

Transforms the given weight tensor based on the specified quantization parameters.

- **weight** *(Tensor)*: The input weight tensor to be transformed.
- **scale** *(Optional[Tensor], default=None)*: Scaling factor for the weight tensor.
- **zeros** *(Optional[Tensor], default=None)*: Zero-point adjustment for the weight tensor.
- **bias** *(Optional[Tensor], default=None)*: Bias to be added to the weight tensor.

#### `__call__(*args: Any) -> Any`

Allows the object to be called like a function, forwarding the call to the `forward` method.

### Properties:

- **M**, **N**, **K**, **A_dtype**, **W_dtype**, **out_dtype**, **accum_dtype**, **storage_dtype**, **with_scaling**, **with_zeros**, **group_size**, **fast_decoding**, **with_bias**, **propagate_a**, **propagate_b**, **layout**, **zeros_mode**: These properties correspond to the parameters defined in `MatmulConfig`, providing easy access to the configuration details.


## Linear

`Linear(infeatures: int, outfeatures: int, bias: bool = False, A_dtype: str = 'float16', W_dtype: str = 'float16', accum_dtype: str = 'float16', out_dtype: str = 'float16', group_size: int = -1, with_scaling: bool = None, with_zeros: bool = False, zeros_mode: str = None, opt_features: Union[int, List[int]] = [1, 16, 32, 64, 128, 256, 512], enable_tuning: bool = True, fast_decoding: bool = True, propagate_b: bool = False)`

Applies a linear transformation to the incoming data: `y = Ax + b`. This module supports quantization and optimization for NVIDIA GPUs using the BitBLAS library.

### Parameters:

- **infeatures** *(int)*: size of each input sample.
- **outfeatures** *(int)*: size of each output sample.
- **bias** *(bool, optional)*: If set to `False`, the layer will not learn an additive bias. Default: `False`.
- **A_dtype** *(str, optional)*: Data type of the input tensor. Default: `'float16'`.
    - Choices: `'float16'`, `'int8'`.
- **W_dtype** *(str, optional)*: Data type of the weights. Default: `'float16'`.
    - Choices: `'float16'`, `'int8'`, `'int4'`, `'int2'`, `'int1'`, `'fp4_e2m1'`, `'af4'`.
- **accum_dtype** *(str, optional)*: Data type for accumulation. Default: `'float16'`.
    - Choices: `'float16'`, `'int32'`.
- **out_dtype** *(str, optional)*: Data type of the output tensor. Default: `'float16'`.
    - Choices: `'float16'`, `'int8'`, `'int32'`.
- **group_size** *(int, optional)*: Group size for quantization. Default: `-1` (no grouping).
- **with_scaling** *(bool, optional)*: Whether to use scaling during quantization. Default: `False`.
- **with_zeros** *(bool, optional)*: Whether to use zero optimization. Default: `False`.
- **zeros_mode** *(str, optional)*: Mode for zero optimization. Default: `None`.
    - Choices: `None`, `'original'`, `'rescale'`, `'quantized'`.
        - `'original'`: Subtract zero-point before scaling. Formula: `target = (dequantize_weight - zero_point) * scale`. where `zero_point` has the same datatype with scale.
        - `'rescale'`: Apply scale factor directly to dequantized weight and then subtract zero-point. Formula: `target = dequantize_weight * scale - zero_point`.
        - `'quantized'`: Apply zero-point adjustment after dequantization and additional dequantization of zero values. Formula: `target = (dequantize_weight - dequantize_qzeros) * scale`, where `dequantize_zeros` represents the dequantized representation of zero values, which can be adapted to qzeros params.
- **opt_features** *(Union[int, List[int]], optional)*: Optimize range of the input shape for dynamic symbolic. Default: `[1, 16, 32, 64, 128, 256, 512]`.
    - If `int`, the bitblas matmul will generate a static shape kernel, which can only be used for the input shape of the specified value.
    - If `List[int]`, the bitblas matmul will generate a dynamic shape kernel, which can be used for the input shape of the specified values. While the input shape represents the target optimized range.
- **enable_tuning** *(bool, optional)*: Whether to enable hardware-aware tuning. Default: `True`.
    - If `True`, the module will perform hardware-aware tuning, and you can also specify the `BITBLAS_DATABASE` value to load pre-optimized kernel from database.
    - If `False`, the module use a default schedule (the performan may not be optimal).
- **fast_decoding** *(bool, optional)*: Whether to enable fast decoding. Default: `True`.
    - If `True`, the module will use fast decoding for INT4/2/1 quantization, which will introduce a weight transformation, and this transformation will be applied to the weight automalically when you call `load_and_transform_weight`.
- **propagate_b** *(bool, optional)*: Whether to propagate bias. Default: `False`.
    - If `Ture`, the module will use a further optimization for layout propagation(OSDI'24 Ladder), which will introduce another transformation to get better performance, and this transformation will be applied to the weight automalically when you call `load_and_transform_weight`. The default value is `False` because we currently do not implement the gemv case, which will be addressed in the future.

### Methods:

#### `forward(A, Output=None)`

Defines the computation performed at every call.

- **A** *(Tensor)*: Input tensor.
- **Output** *(Tensor, optional)*: Pre-allocated output tensor. Default: `None`.
    - If `None`, the module will allocate a new tensor for the output.
    - If not `None`, the module will use the pre-allocated tensor for the output.

Returns: The output tensor.

#### `init_params()`

Initializes parameters handles (convert constant params into ctypes void pointer) for the computation. We currently put this fuction in the forward function, so you do not need to call it manually. But if you lift this function out of the forward function, you can call it manually to aoid the transformation.

#### `load_and_transform_weight(weight, scales=None, zeros=None, bias=None)`

This method is designed to load and optionally transform the weight matrix along with scales, zeros, and bias for use in quantized computations. It is particularly useful when transitioning from a floating-point model to a quantized model, allowing for the adjustment of model parameters to fit the requirements of quantization and optimization processes.

- **Parameters:**
  - **weight** *(Tensor)*: The weight tensor to be loaded into the layer. This tensor should have dimensions that match the expected input features and output features of the layer. The method will also apply any necessary transformations to the weight tensor to align with the quantization and optimization configurations of the layer.
  - **scales** *(Tensor, optional)*: A tensor containing scale factors for quantization. These scales are used to adjust the weight values during the quantization process, ensuring that the dynamic range of the weights is appropriately represented in the quantized format. If not provided, the method assumes that either scaling is not required or has already been applied to the weights.
  - **zeros** *(Tensor, optional)*: A tensor indicating the optimized representation of zeros, particularly useful in sparse models where zero values can be efficiently encoded. This parameter is only relevant if zero optimization (`with_zeros`) is enabled for the layer. Providing this tensor allows for further memory and computation optimizations during the forward pass.
  - **bias** *(Tensor, optional)*: The bias tensor to be loaded into the layer. If the layer is configured to use a bias (`bias=True` during initialization), this tensor provides the bias values for each output feature. If `None`, it is assumed that the layer does not use a bias or that the bias is already incorporated into another parameter.
`load_and_transform_weight(weight, scales=None, zeros=None, bias=None)`

Loads and transforms the weight matrix and optional scales, zeros, and bias for quantized computation.

- **weight** *(Tensor)*: Weight tensor.
- **scales** *(Tensor, optional)*: Scales tensor for quantization. Default: `None`.
- **zeros** *(Tensor, optional)*: Zeros tensor for zero optimization. Default: `None`.
- **bias** *(Tensor, optional)*: Bias tensor. Default: `None`.

### `repack_from_gptq(gptq_module)`

This method facilitates the integration of parameters from a module that has undergone Generalized Post Training Quantization (GPTQ), repacking and transforming these parameters as necessary for compatibility with the BitBLAS-optimized `Linear` layer. The `gptq_module` must have its parameters in a format that is compatible with the expectations of the `Linear` layer's quantization and optimization configurations. This includes the shape and data type of the quantized weights, scales, and zeros. The method automatically handles the transformation and repacking of these parameters, including transposing weights if necessary, converting quantized zeros into the expected format, and adjusting scales and biases for direct use in the optimized forward pass of the `Linear` layer.

- **Parameters:**
  - **gptq_module** *(Module)*: A module that contains quantized parameters following the GPTQ process. This module should have attributes corresponding to quantized weights (`qweight`), scales (`scales`), optimized zeros (`qzeros`), and optionally biases (`bias`). The method extracts these parameters, applies any necessary transformations for compatibility with the BitBLAS optimizations, and loads them into the `Linear` layer.
