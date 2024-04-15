mkdir -p ./baseline_framework

apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev

apt-get install -y llvm-10

git clone --branch v0.14.0 https://github.com/apache/tvm ./baseline_framework/tvm_14.0 --recursive

cd ./baseline_framework/tvm_14.0
mkdir build
cp cmake/config.cmake build
cd build
echo "set(USE_LLVM llvm-config-10)" >> config.cmake && echo "set(USE_CUDA ON)" >> config.cmake

cmake .. && make -j && cd ../../..
