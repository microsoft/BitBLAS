cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(YourProjectTests LANGUAGES CXX CUDA)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)

# Find GTest
find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})

add_subdirectory(lop3_type_conversion)
