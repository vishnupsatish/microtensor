/**
 *    Author:  Vishnu Satish
 *    Created: Dec 25, 2025
 **/

#pragma once

#include <memory>
#include <vector>

#include "private/shape.h"

class TensorImpl;

// Lightweight object that is a wrapper around a `TensorImpl`.
class Tensor {
  std::shared_ptr<TensorImpl> m_impl;

 public:
  Tensor(const Shape& shape, bool requiresGrad = false);
  Tensor(const Tensor& other);
  Tensor(std::shared_ptr<TensorImpl> otherImpl);
  Tensor(const Shape& shape, const std::vector<float>& data,
         bool requiresGrad = false);

  bool isValid() const;
  void fillRandom();
  void print(std::ostream& os);
  void dumpTensor(std::ostream& os);
  Tensor getGrad() const;
  Shape getShape() const;

  Tensor operator+(const Tensor& other);
  Tensor operator-(const Tensor& other);
  Tensor operator*(const Tensor& other);
  Tensor operator/(const Tensor& other);
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

  void backward();
};
