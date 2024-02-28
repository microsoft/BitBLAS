# Transparency Responsible FAQ for BitBLAS

## What is BitBLAS?

BitBLAS is a software framework designed to generate high-performance CUDA/HIP code for BLAS operators with optimizing swizzling and layout propagation. BitBLAS makes it easier and more efficient to perform complex mathematical operations, especially in the fields of machine learning and high-performance computing.

## What can BitBLAS do?

BitBLAS offers several functionalities aimed at improving the performance and flexibility of linear algebra computations. It supports auto tensorization to optimize operations across different data types and compute patterns and dynamic symbolic support for generating kernels that can handle dyanmic input shape. Its design targets high-performance computing needs.

## What are BitBLAS's intended uses?

The primary intended uses of BitBLAS are in the domains of machine learning, deep learning, scientific computing, and any other field that requires efficient linear algebra computations. It is designed to accelerate training and inference processes, and improve computational efficiency in high-performance computing applications.

## How was BitBLAS evaluated?

BitBLAS was evaluated based on its performance and correctness in generating and executing optimized code for various linear algebra operations. Performance metrics include the speed of execution, computational efficiency. The evaluation involved comparing BitBLAS's performance with other existing frameworks, such as cuBLAS/CUTLASS/rocBLAS/AMOS/TensorIR and manual implementation of some operators, across a range of computational tasks and hardware configurations. The accuracy verification through simulations with PyTorch operators/

## What are the limitations of BitBLAS?

While BitBLAS is designed to be highly efficient, there are limitations to its applicability and performance. It is optimized for CUDA/HIP environments, which means its use is restricted to systems with compatible GPUs.

Moreover, due to the complexity of dynamic shape support, we currently only support only one dim is dynamic.

## Operational factors and settings for effective and responsible use

BitBLAS is expected to perform reliably within the operational factors and settings of CUDA/HIP-enabled hardware architectures. Users can influence the system's behavior through customization of the DSL scripts, selection of data types and precision and operator configurations, and tuning of performance parameters to suit their specific computational needs. These choices impact the efficiency and accuracy of the computations, making it essential for users to understand the trade-offs involved.

## Plugins and Extensibility

BitBLAS doesn't allow for plug ins or extensibility.
