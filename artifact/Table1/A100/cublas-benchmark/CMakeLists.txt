cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(Cuda_Gemm C CXX CUDA)

# Sets the possible install locations
set(CUBLAS_HINTS
        ${CUDA_ROOT}
        $ENV{CUDA_ROOT}
        $ENV{CUDA_TOOLKIT_ROOT_DIR}
        )
set(CUBLAS_PATHS
        /usr
        /usr/local
        /usr/local/cuda
        )

# Finds the include directories
find_path(CUBLAS_INCLUDE_DIRS
        NAMES cublas_v2.h cuda.h
        HINTS ${CUBLAS_HINTS}
        PATH_SUFFIXES include inc include/x86_64 include/x64
        PATHS ${CUBLAS_PATHS}
        DOC "cuBLAS include header cublas_v2.h"
        )
mark_as_advanced(CUBLAS_INCLUDE_DIRS)

# Finds the libraries
find_library(CUDA_LIBRARIES
        NAMES cudart
        HINTS ${CUBLAS_HINTS}
        PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86 lib/Win32 lib/import lib64/import
        PATHS ${CUBLAS_PATHS}
        DOC "CUDA library"
        )
mark_as_advanced(CUDA_LIBRARIES)
find_library(CUBLAS_LIBRARIES
        NAMES cublas
        HINTS ${CUBLAS_HINTS}
        PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86 lib/Win32 lib/import lib64/import
        PATHS ${CUBLAS_PATHS}
        DOC "cuBLAS library"
        )
mark_as_advanced(CUBLAS_LIBRARIES)

# ==================================================================================================

# Notification messages
if(NOT CUBLAS_INCLUDE_DIRS)
    message(STATUS "Could NOT find 'cuBLAS.h', install CUDA/cuBLAS or set CUDA_ROOT")
endif()
if(NOT CUDA_LIBRARIES)
    message(STATUS "Could NOT find CUDA library, install it or set CUDA_ROOT")
endif()
if(NOT CUBLAS_LIBRARIES)
    message(STATUS "Could NOT find cuBLAS library, install it or set CUDA_ROOT")
endif()

# Determines whether or not cuBLAS was found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuBLAS DEFAULT_MSG CUBLAS_INCLUDE_DIRS CUDA_LIBRARIES CUBLAS_LIBRARIES)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(cublas_benchmark cublas_benchmark.cu)
set_target_properties(cublas_benchmark PROPERTIES CUDA_ARCHITECTURES "86")
set_target_properties(cublas_benchmark PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cublas_benchmark PRIVATE ${CUDA_LIBRARIES} -lcublas -lcurand)
