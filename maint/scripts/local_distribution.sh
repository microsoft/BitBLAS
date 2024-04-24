#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# if dist and build directories exist, remove them
if [ -d dist ]; then
    rm -r dist
fi

if [ -d build ]; then
    rm -r build
fi

python setup.py bdist_wheel
