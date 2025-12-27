/**
 *    Author:  Vishnu Satish
 *    Created: Dec 25, 2025
 **/

#pragma once

#include <initializer_list>
#include <memory>
#include <vector>

#include "operation.h"

// Not to be used in a polymorphic setting!
struct Storage : std::vector<float> {
  using std::vector<float>::vector;
};

class Tensor {
  std::shared_ptr<Storage> m_data;
  std::vector<size_t> m_strides;
  std::vector<size_t> m_shape;
  std::shared_ptr<Tensor> m_grad;
  // Automatically require gradient for now.
  // const bool requires_grad = false;

  void initialize_grad();
  void accumulate_grad(std::shared_ptr<Tensor> new_grad);

 public:
  std::unique_ptr<Operation> m_creator;

  Tensor(const std::vector<size_t>& shape);

  void fill_random();
  void print(std::ostream& os);
  std::shared_ptr<Tensor> getGrad() const;

  std::shared_ptr<Tensor> operator*(const Tensor& other);
  std::shared_ptr<Tensor> operator+(const Tensor& other);

  void backward();
};
