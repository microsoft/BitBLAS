#pragma once

#include <vector>
#include <numeric>
#include <memory>
#include <cassert>

#include <curand.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/copy.h>

template <typename InputType, typename OutputType>
__global__ void convertType(const InputType *input, OutputType *output, int numElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
        output[idx] = static_cast<OutputType>(input[idx]);
    }
}

template <typename InputType, typename OutputType>
__global__ void scaleType(const InputType *input, OutputType *output, int numElements, float ScaleFactor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
        output[idx] = static_cast<OutputType>(input[idx] * ScaleFactor);
    }
}

template <typename T>
class Tensor
{
    std::vector<int> dims_;
    int size_;

    struct deleteCudaPtr
    {
        void operator()(T *p) const
        {
            cudaFree(p);
        }
    };

    std::shared_ptr<T> ptr_;

public:
    Tensor() {}

    Tensor(std::vector<int> dims) : dims_(dims)
    {
        T *tmp_ptr;
        size_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
        cudaMalloc(&tmp_ptr, sizeof(T) * size_);

        ptr_.reset(tmp_ptr, deleteCudaPtr());
    }

    T *begin() const { return ptr_.get(); }
    T *end() const { return ptr_.get() + size_; }
    int size() const { return size_; }
    std::vector<int> dims() const { return dims_; }
};

template <typename T>
Tensor<T> fill(std::vector<int> dims, float val)
{
    Tensor<T> tensor(dims);
    thrust::fill(thrust::device_ptr<T>(tensor.begin()),
                 thrust::device_ptr<T>(tensor.end()), val);
    return tensor;
}

template <typename T>
Tensor<T> zeros(std::vector<int> dims)
{
    Tensor<T> tensor(dims);
    thrust::fill(thrust::device_ptr<T>(tensor.begin()),
                 thrust::device_ptr<T>(tensor.end()), 0.f);
    return tensor;
}

template <typename T>
typename std::enable_if<(std::is_same<T, float>::value), Tensor<T>>::type
rand(std::vector<int> dims, curandGenerator_t curand_gen)
{
    Tensor<T> tensor(dims);
    curandGenerateUniform(curand_gen, tensor.begin(), tensor.size());
    return tensor;
}

template <typename T>
typename std::enable_if<!(std::is_same<T, float>::value), Tensor<T>>::type
rand(std::vector<int> dims, curandGenerator_t curand_gen)
{
    Tensor<float> temp_tensor(dims);
    curandGenerateUniform(curand_gen, temp_tensor.begin(), temp_tensor.size());

    Tensor<T> tensor(dims);
    auto size = tensor.size();
    // Convert Tensor With Efficient Cuda Kernel
    convertType<<<(size + 255) / 256, 256>>>(temp_tensor.begin(), tensor.begin(), size);
    return tensor;
}

void pad_dim(int &dim, int pad_v)
{
    assert(pad_v > 0);
    if (dim % pad_v)
    {
        int pad = pad_v - dim % pad_v;
        dim += pad;
    }
}