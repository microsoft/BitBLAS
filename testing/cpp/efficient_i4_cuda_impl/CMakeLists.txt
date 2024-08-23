# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
function (ADD_CUDA_TEST_EXECUTABLE name)
    add_executable(${name} ${name}.cu)
    set_target_properties(${name} PROPERTIES CUDA_ARCHITECTURES 80)
    # add flags
    target_compile_options(${name} PRIVATE --expt-relaxed-constexpr)
    set_target_properties(${name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${name} gtest gtest_main)
endfunction(ADD_CUDA_TEST_EXECUTABLE)

ADD_CUDA_TEST_EXECUTABLE(efficient_i4)

function (ADD_CPP_TEST_EXECUTABLE name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} gtest gtest_main pthread)
endfunction(ADD_CPP_TEST_EXECUTABLE)

ADD_CPP_TEST_EXECUTABLE(param_permutate)
