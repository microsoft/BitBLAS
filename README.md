# BitBLAS

BitBLAS is a light weight framework to generate high performance CUDA/HIP code for BLAS operators with swizzling and layout propagation.

## Feature

- Auto Tensorization.
- Dynamic symbolic support, generate kernel with dynamic shape.
- Auto Layout Propagation.
- Aanalysis based on the DSL TensorIR, can be easily extended.

## Requirements
To manually install BitBLAS, please checkout `maint/scripts/installation.sh`.

Also Make sure you already have the cuda toolkit (version >= 11) installed in the system.

Finally, add ./python and tvm/python to PYTHONPATH.

## Quick Start
- 
