/**
 *    Author:  Vishnu Satish
 *    Created: Dec 26, 2025
 **/

// MUST ONLY be included by the current library.

#pragma once

#include <memory>
#include <vector>

#include "shape.h"

class Operation;

// Not to be used in a polymorphic setting!
struct Storage : std::vector<float> {
  using std::vector<float>::vector;
};

struct TensorImpl {
  std::shared_ptr<Storage> m_data;
  std::vector<size_t> m_strides;
  Shape m_shape;
  std::shared_ptr<TensorImpl> m_grad;
  size_t m_offset = 0;
  // Automatically require gradient for now.
  // const bool requires_grad = false;

  void initialize_grad();
  void accumulate_grad(std::shared_ptr<TensorImpl> new_grad);

 public:
  std::unique_ptr<Operation> m_creator;

  // Uses default strides for the shape.
  TensorImpl(const Shape& shape);
  TensorImpl(const Shape& shape, const std::vector<size_t>& strides,
             std::shared_ptr<Storage> data);

  void fill_random();
  void print(std::ostream& os);
  std::shared_ptr<TensorImpl> getGrad() const;

  void backward();
};

// TODO: in the future, this could be part of a tensor iterator class.
size_t get_physical_offset(const std::vector<size_t>& coords,
                           const std::vector<size_t>& strides,
                           size_t offset = 0);

void increment_coords(std::vector<size_t>& coords, const Shape& shape);