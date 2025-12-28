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
  Tensor(const Shape& shape);
  Tensor(const Tensor& other);
  Tensor(std::shared_ptr<TensorImpl> otherImpl);
  Tensor(const Shape& shape, const std::vector<float>& data);

  void fillRandom();
  void print(std::ostream& os);
  Tensor getGrad() const;
  Shape getShape() const;

  void dumpTensor(std::ostream& os);

  Tensor operator+(const Tensor& other);
  Tensor operator*(const Tensor& other);

  Tensor tanh();

  // Populating gradients backwards.
  void backward();
};
