/**
 *    Author:  Vishnu Satish
 *    Created: Dec 26, 2025
 **/

// MUST ONLY be included by the current library.

#pragma once

#include <memory>
#include <vector>

class Operation;

// Not to be used in a polymorphic setting!
struct Storage : std::vector<float> {
  using std::vector<float>::vector;
};

struct TensorImpl {
  std::shared_ptr<Storage> m_data;
  std::vector<size_t> m_strides;
  std::vector<size_t> m_shape;
  std::shared_ptr<TensorImpl> m_grad;
  // Automatically require gradient for now.
  // const bool requires_grad = false;

  void initialize_grad();
  void accumulate_grad(std::shared_ptr<TensorImpl> new_grad);

 public:
  std::unique_ptr<Operation> m_creator;

  TensorImpl(const std::vector<size_t>& shape);

  void fill_random();
  void print(std::ostream& os);
  std::shared_ptr<TensorImpl> getGrad() const;

  void backward();
};
