# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Learn a lot from the MLC-LLM Project 
# https://github.com/mlc-ai/mlc-llm/blob/main/CMakeLists.txt

cmake_minimum_required(VERSION 3.18)
project(TILE_LANG C CXX)


if(EXISTS ${CMAKE_BINARY_DIR}/config.cmake)
  include(${CMAKE_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_SOURCE_DIR}/config.cmake)
    include(${CMAKE_SOURCE_DIR}/config.cmake)
  endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(
    CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE
  )
  message(STATUS "Setting default build type to " ${CMAKE_BUILD_TYPE})
endif(NOT CMAKE_BUILD_TYPE)


if (TILE_LANG_INSTALL_STATIC_LIB)
  set(BUILD_STATIC_RUNTIME ON)
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT DEFINED TVM_SOURCE_DIR)
  if(DEFINED ENV{TVM_SOURCE_DIR})
    set(TVM_SOURCE_DIR "$ENV{TVM_SOURCE_DIR}")
  else()
    set(TVM_SOURCE_DIR ${PROJECT_SOURCE_DIR}/3rdparty/tvm)
  endif(DEFINED ENV{TVM_SOURCE_DIR})
endif (NOT DEFINED TVM_SOURCE_DIR)

message(STATUS "TVM_SOURCE_DIR: ${TVM_SOURCE_DIR}")
add_subdirectory(${TVM_SOURCE_DIR} tvm EXCLUDE_FROM_ALL)

tvm_file_glob(GLOB_RECURSE TILE_LANG_SRCS src/*.cc)
message(STATUS "TILE_LANG_SRCS: ${TILE_LANG_SRCS}")
add_library(tilelang_objs OBJECT ${TILE_LANG_SRCS})

set(
  TILE_LANG_INCLUDES
  ${TVM_SOURCE_DIR}/include
  ${TVM_SOURCE_DIR}/src
  ${TVM_SOURCE_DIR}/3rdparty/dlpack/include
  ${TVM_SOURCE_DIR}/3rdparty/dmlc-core/include
)

# Find CUDA
find_package(CUDA REQUIRED)
# CUDA Dependencies
set(TILE_LANG_INCLUDES ${TILE_LANG_INCLUDES} ${CUDA_INCLUDE_DIRS})

set(TILE_LANG_COMPILE_DEFS ${TILE_LANG_COMPILE_DEFS} DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
set(TILE_LANG_COMPILE_DEFS ${TILE_LANG_COMPILE_DEFS} __STDC_FORMAT_MACROS=1)
set(TILE_LANG_COMPILE_DEFS ${TILE_LANG_COMPILE_DEFS} PICOJSON_USE_INT64)

target_include_directories(tilelang_objs PRIVATE ${TILE_LANG_INCLUDES})
target_compile_definitions(tilelang_objs PRIVATE ${TILE_LANG_COMPILE_DEFS})
target_include_directories(tilelang_objs PRIVATE ${TOKENZIER_CPP_PATH}/include)
target_compile_definitions(tilelang_objs PRIVATE -DTILE_LANG_EXPORTS)


add_library(tilelang SHARED $<TARGET_OBJECTS:tilelang_objs>)
add_library(tilelang_static STATIC $<TARGET_OBJECTS:tilelang_objs>)
add_dependencies(tilelang_static tvm_runtime)
set_target_properties(tilelang_static PROPERTIES OUTPUT_NAME tilelang)

target_link_libraries(tilelang PUBLIC tvm_runtime)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(tilelang PRIVATE "TVM_LOG_DEBUG")
    target_compile_definitions(tilelang_objs PRIVATE "TVM_LOG_DEBUG")
    target_compile_definitions(tilelang_static PRIVATE "TVM_LOG_DEBUG")
endif()


add_library(tilelang_module SHARED $<TARGET_OBJECTS:tilelang_objs>)
target_link_libraries(tilelang_module PUBLIC tvm)


set_property(TARGET tilelang_module APPEND PROPERTY LINK_OPTIONS "${MLC_VISIBILITY_FLAG}")
set_property(TARGET tilelang APPEND PROPERTY LINK_OPTIONS "${MLC_VISIBILITY_FLAG}")


# when this option is on,
# we install all static lib deps into lib
if (TILE_LANG_INSTALL_STATIC_LIB)
  install(TARGETS
    tilelang_static
    tvm_runtime
    LIBRARY DESTINATION lib${LIB_SUFFIX}
    )
else()
  install(TARGETS tvm_runtime tilelang tilelang_module
    tilelang_static
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib${LIB_SUFFIX}
  )
endif()
