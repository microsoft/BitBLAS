function (ADD_CUDA_TEST_EXECUTABLE name)
    add_executable(${name} ${name}.cu)
    set_target_properties(${name} PROPERTIES CUDA_ARCHITECTURES 60)
    set_target_properties(${name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${name} gtest gtest_main)
endfunction(ADD_CUDA_TEST_EXECUTABLE)

ADD_CUDA_TEST_EXECUTABLE(lowprecision_to_float16)
