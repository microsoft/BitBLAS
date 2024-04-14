#pragma once

#include <cassert>
#include <memory>
#include <numeric>
#include <vector>

#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <thrust/copy.h>

template <typename T> class Tensor {
  std::vector<int> dims_;
  int size_;

  struct deleteHipPtr {
    void operator()(T *p) const { hipFree(p); }
  };

  std::shared_ptr<T> ptr_;

public:
  Tensor() {}

  Tensor(std::vector<int> dims) : dims_(dims) {
    T *tmp_ptr;
    size_ =
        std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
    hipMalloc(&tmp_ptr, sizeof(T) * size_);

    ptr_.reset(tmp_ptr, deleteHipPtr());
  }

  T *begin() const { return ptr_.get(); }
  T *end() const { return ptr_.get() + size_; }
  int size() const { return size_; }
  std::vector<int> dims() const { return dims_; }
};

template <typename T> Tensor<T> fill(std::vector<int> dims, float val) {
  Tensor<T> tensor(dims);
  thrust::fill(tensor.begin(), tensor.end(), val);
  return tensor;
}

template <typename T> Tensor<T> zeros(std::vector<int> dims) {
  Tensor<T> tensor(dims);
  thrust::fill(tensor.begin(), tensor.end(), 0.f);
  return tensor;
}

template <typename T>
typename std::enable_if<(std::is_same<T, float>::value), Tensor<T>>::type
rand(std::vector<int> dims, hiprandGenerator_t hiprand_gen) {
  Tensor<T> tensor(dims);
  hiprandGenerateUniform(hiprand_gen, tensor.begin(), tensor.size());
  return tensor;
}

template <typename T>
typename std::enable_if<(std::is_same<T, half>::value), Tensor<T>>::type
rand(std::vector<int> dims, hiprandGenerator_t hiprand_gen) {
  Tensor<T> tensor(dims);
  hiprandGenerateUniformHalf(hiprand_gen, tensor.begin(), tensor.size());
  return tensor;
}

template <typename T>
typename std::enable_if<(std::is_same<T, uint8_t>::value), Tensor<T>>::type
rand(std::vector<int> dims, hiprandGenerator_t hiprand_gen)
{
  Tensor<T> tensor(dims);
  hiprandGenerateChar(hiprand_gen, tensor.begin(), tensor.size());
  return tensor;
}

void pad_dim(int &dim, int pad_v) {
  assert(pad_v > 0);
  if (dim % pad_v) {
    int pad = pad_v - dim % pad_v;
    dim += pad;
  }
}
