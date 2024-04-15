cmake_minimum_required(VERSION 3.8)  
project(rocblas_benchmark LANGUAGES CXX)  

# set cxx to hipcc
set(CMAKE_CXX_COMPILER hipcc)
# set c++ to 14

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(rocblas REQUIRED)  
find_package(rocprim REQUIRED)  
find_package(rocthrust  REQUIRED)  
find_package(hiprand REQUIRED)  

add_executable(rocblas_benchmark  
    rocblas_benchmark.cpp  
)  
  
target_include_directories(rocblas_benchmark PRIVATE ${HIP_INCLUDE_DIRS})  
target_include_directories(rocblas_benchmark PRIVATE ${ROCBLAS_INCLUDE_DIRS})  
target_include_directories(rocblas_benchmark PRIVATE ${HIPRAND_INCLUDE_DIRS})  



target_link_libraries(rocblas_benchmark PRIVATE roc::rocblas roc::rocprim roc::rocthrust hip::hiprand roc::rocprim_hip)  
