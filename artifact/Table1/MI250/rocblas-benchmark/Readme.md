# rocBLAS Benchmark  
  
This project contains an example of using rocBLAS for benchmarking matrix multiplication with various data types.  
  
## Requirements  
  
- ROCm  
- rocBLAS  
- rocPRIM  
- rocThrust  
- hipRAND  
  
## Building  
  
To build the project, follow these steps:  
  
1. Create a build directory:  
    ```sh  
    mkdir build  
    cd build  
    ``` 

2. Run CMake:
    ```sh  
    cmake ..   
    ``` 

3. Compile the project:
    ```sh  
    make  
    ```

Running
 
To run the benchmark, execute the following command in the build directory:
`./rocblas_benchmark`
 
This will run the benchmark with the specified problem sizes and configurations in inference_server_set. The output will show the time in milliseconds for each problem size and configuration for different data types (FP32, FP16-F32, FP16-F16, and INT8-INT32).

License
 
This project is open-source and free to use, modify, and distribute.