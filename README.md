# Ladder

Installing Ladder:

```bash
git clone --recursive https://github.com/LeiWang1999/Ladder Ladder

cd 3rdparty/tvm

mkdir build; cd build; cp ../cmake/config.cmake .; 
echo set\(USE_LLVM ON\) >> config.cmake;
echo set\(USE_CUDA ON\) >> config.cmake; 

cmake ..; make -j
# mv out of tvm folder
cd ..


# append the following envs to ~/.bashrc
export PYTHONPATH=/your/path/to/Ladder/python
export PYTHONPATH=$PYTHONPATH:/your/path/to/Ladder/3rdparty/tvm/python
export CPLUS_INCLUDE_PATH=/home/t-leiwang/ladder_workspace/ladder_cutlass/include

```