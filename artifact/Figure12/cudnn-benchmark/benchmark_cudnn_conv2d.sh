#!/bin/sh

python conv2d.py --target cuda --enable_cudnn --number 5 --repeats 5 --begin 0 --num 10 --dtype FP16
