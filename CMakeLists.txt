# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Learn a lot from the MLC-LLM Project 
# https://github.com/mlc-ai/mlc-llm/blob/main/CMakeLists.txt

cmake_minimum_required(VERSION 3.18)
project(TILE_LANG C CXX)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0")
  macro(tilelang_file_glob glob variable)
    file(${glob} ${variable} CONFIGURE_DEPENDS ${ARGN})
  endmacro()
else()
  macro(tilelang_file_glob)
    file(${glob} ${variable} ${ARGN})
  endmacro()
endif()

if (DEFINED TVM_PREBUILD_PATH)
  message(STATUS "TVM_PREBUILD_PATH: ${TVM_PREBUILD_PATH}")
  if(EXISTS ${TVM_PREBUILD_PATH}/config.cmake)
    include(${TVM_PREBUILD_PATH}/config.cmake)
  endif()
else()
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

endif()


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

if (DEFINED TVM_PREBUILD_PATH)
  message(STATUS "TVM_PREBUILD_PATH: ${TVM_PREBUILD_PATH}")
  # add library path
  add_library(tvm SHARED IMPORTED)
  set_target_properties(tvm PROPERTIES
    IMPORTED_LOCATION "${TVM_PREBUILD_PATH}/libtvm.so"
    INTERFACE_INCLUDE_DIRECTORIES "${TVM_PREBUILD_PATH}/../include"
  )
  add_library(tvm_runtime SHARED IMPORTED)
  set_target_properties(tvm_runtime PROPERTIES
    IMPORTED_LOCATION "${TVM_PREBUILD_PATH}/libtvm_runtime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${TVM_PREBUILD_PATH}/../include"
  )
else()
  message(STATUS "TVM_PREBUILD_PATH NOT SET, will build TVM from source")
  message(STATUS "TVM_SOURCE_DIR: ${TVM_SOURCE_DIR}")
  add_subdirectory(${TVM_SOURCE_DIR} tvm EXCLUDE_FROM_ALL)
endif()


tilelang_file_glob(GLOB_RECURSE TILE_LANG_SRCS src/*.cc)
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
find_package(CUDAToolkit REQUIRED)
# CUDA Dependencies
message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
set(TILE_LANG_INCLUDES ${TILE_LANG_INCLUDES} ${CUDAToolkit_INCLUDE_DIRS})

set(TILE_LANG_COMPILE_DEFS ${TILE_LANG_COMPILE_DEFS} DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
set(TILE_LANG_COMPILE_DEFS ${TILE_LANG_COMPILE_DEFS} __STDC_FORMAT_MACROS=1)
set(TILE_LANG_COMPILE_DEFS ${TILE_LANG_COMPILE_DEFS} PICOJSON_USE_INT64)

target_include_directories(tilelang_objs PRIVATE ${TILE_LANG_INCLUDES})
target_compile_definitions(tilelang_objs PRIVATE ${TILE_LANG_COMPILE_DEFS})
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

# when this option is on,
# we install all static lib deps into lib
if (TILE_LANG_INSTALL_STATIC_LIB)
  install(TARGETS
    tilelang_static
    tvm_runtime
    LIBRARY DESTINATION lib${LIB_SUFFIX}
    )
else()
  if (DEFINED TVM_PREBUILD_PATH)
    install(TARGETS tilelang tilelang_module
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib${LIB_SUFFIX}
    )
  else()
    install(TARGETS tvm_runtime tilelang tilelang_module
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib${LIB_SUFFIX}
    )
  endif()
endif()
