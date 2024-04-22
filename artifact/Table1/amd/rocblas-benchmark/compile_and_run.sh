#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
mkdir -p build
cd build
cmake ..
make -j
./rocblas-benchmark
