/**
 *    Author:  Vishnu Satish
 *    Created: Dec 25, 2025
 **/

#pragma once

#include <memory>
#include <vector>

class TensorImpl;

// Lightweight object that is a wrapper around a `TensorImpl`.
class Tensor {
  std::shared_ptr<TensorImpl> m_impl;

 public:
  Tensor(const std::vector<size_t>& shape);
  Tensor(const Tensor& other);
  Tensor(std::shared_ptr<TensorImpl> otherImpl);

  void fill_random();
  void print(std::ostream& os);
  Tensor getGrad() const;

  // Performs operation and stores state to be able to find gradients
  // automatically.
  Tensor operator+(const Tensor& other);
  Tensor operator*(const Tensor& other);

  // Populating gradients backwards.
  void backward();
};
