/**
 *    Author:  Vishnu Satish
 *    Created: Dec 25, 2025
 **/

#pragma once

#include <memory>
#include <vector>

#include "private/shape.h"
#include "private/tensor_impl.h"

// Lightweight object that is a wrapper around a `TensorImpl`.
class Tensor {
  std::shared_ptr<TensorImpl> m_impl;

 public:
  Tensor();
  Tensor(const Shape& shape, bool requiresGrad = false);
  Tensor(const Tensor& other);
  Tensor(std::shared_ptr<TensorImpl> otherImpl);
  Tensor(const Shape& shape, const std::vector<float>& data,
         bool requiresGrad = false);

  bool isValid() const;

  void print(std::ostream& os);
  void dumpTensor(std::ostream& os);
  Tensor getGrad() const;
  Shape getShape() const;

  // Operations that create new Tensors and also create DAG nodes.
  Tensor operator+(const Tensor& other);
  Tensor operator+(float other);
  Tensor operator-(const Tensor& other);
  Tensor operator*(const Tensor& other);
  Tensor operator*(float other);
  Tensor operator/(const Tensor& other);
  Tensor operator/(float other);
  Tensor pow(float exp);
  Tensor tanh();
  Tensor exp();
  Tensor log();
  Tensor matmul(const Tensor& other);
  Tensor permute(const std::vector<size_t>& dims);
  Tensor broadcast(const Shape& target);
  Tensor reduceSum(const std::vector<int>& dims, bool keep_dims = false);
  Tensor reduceMax(int dim, bool keep_dim = false);
  Tensor reshape(const Shape& shape);
  Tensor makeContiguous();
  Tensor slice(const std::vector<int>& start, const Shape& size);
  Tensor relu();
  Tensor softmax(int dim);

  // In-place operations.
  Tensor& operator-=(const Tensor& other);

  void backward();

  // F should be callable.
  // TODO: change name, this should not be called fillRandom.
  template <typename F>
  void fillRandom(F&& initFn) {
    m_impl->fillRandom(initFn);
  }
};
