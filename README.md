# Welder

Welder is an end-to-end deep-learning compiler for CUDA GPUs. Welder features automatic fusion and tiling on the whole tensor computation graph. Welder can also generate efficient kernels for Volta/Ampere TensorCores (utilizing CUTLASS block/warp level templates).

## Requirements
To use Welder, you need to go through the following script to setup TVM and CUTLASS first.

Also Make sure you already have the cuda toolkit (version >= 11) installed in the system.

```bash
git clone https://github.com/nox-410/tvm --recursive -b develop
# Fill in USE_CUDA and USE_LLVM in tvm/cmake/config.cmake, like this:
# echo "set(USE_LLVM ON)" >> config.cmake && echo "set(USE_CUDA ON)" >> config.cmake
# You need to install LLVM first if you don't have one, like this:
# apt-get install llvm-dev
mkdir -p tvm/build && cd tvm/build && cp ../cmake/config.cmake . && cmake .. && make -j && cd -
export PYTHONPATH="$PYTHONPATH:$PWD/tvm/python"

git clone https://github.com/nox-410/cutlass -b welder
export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:$PWD/cutlass/include"

pip install torch onnx attrs cloudpickle decorator psutil synr tornado xgboost

# (optional) for some performance test or model
pip install torchvision timm onnxruntime-gpu
```

Finally, add ./python to PYTHONPATH.

## Usage
### Prepare onnx model

Supporting opset11, use ./testing/torch2onnx.py to get some supported models.

```bash
python3 ./testing/torch2onnx.py bert --fp16 --bs 64 --prefix workdir
```

### Run the compiler

```bash
python3 ./testing/relay_test.py workdir
```
Note: It is recommended to add --nhwc flag for convolution-based models under fp16-tensorcore case.

### run test

```bash
# check the model inference latency
python3 ./testing/test_welder_perf.py workdir
# check the accuracy, this requires onnxruntime-gpu to do the cross validation
python3 ./testing/test_welder_acc.py workdir
```

## Citation
Please cite [the paper](https://www.usenix.org/system/files/osdi23-shi.pdf) in your publications if Welder helps your research.
```
@inproceedings{shi2023welder,
  title={Welder: Scheduling Deep Learning Memory Access via Tile-graph},
  author={Shi, Yining and Yang, Zhi and Xue, Jilong and Ma, Lingxiao and Xia, Yuqing and Miao, Ziming and Guo, Yuxiao and Yang, Fan and Zhou, Lidong},
  booktitle={17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23)},
  pages={701--718},
  year={2023}
}
```
