/**
 *    Author:  Vishnu Satish
 *    Created: Dec 26, 2025
 **/

#include "operation.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

#include "tensor_impl.h"

namespace {

// TODO: This can probably become compile-time polymorphism, unless we want to
// allow users to write Tensor functions / deep learning models at runtime.
class AddOp : public Operation {
 public:
  using Operation::Operation;

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    // If shapes are not the same, handled by BroadcastOp.
    assert(m_parents[0]->m_shape == m_parents[1]->m_shape);
    return {grad_output, grad_output};
  }
};

class MulOp : public Operation {
 public:
  using Operation::Operation;

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    assert(m_parents[0]->m_shape == m_parents[1]->m_shape);
    // Note: we must not track creator when multiplying as a result of finding
    // the gradient. Consider the following example: c = a + b, d = a * c,
    // backward(d). We see that grad_a = grad_output * c, and if tracking the
    // creator is enabled, grad_a's creator will be a MulOp with inputs
    // grad_output and c. But, c's creator was an AddOp with inputs a and b, and
    // grad_a is owned by a. This forms a cycle, which is a -> grad_a -> c -> a,
    // and thus will cause a memory leak. Certain ML algorithms require
    // double-backprop, so we'll implement if/when that time comes.
    auto grad_op1 = multiply(grad_output, m_parents[1], false);
    auto grad_op2 = multiply(grad_output, m_parents[0], false);

    return {grad_op1, grad_op2};
  }
};

class BroadcastOp : public Operation {
 public:
  using Operation::Operation;

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    auto input = m_parents[0];

    auto grad_input = std::make_shared<TensorImpl>(input->m_shape);
    std::fill(grad_input->m_data->begin(), grad_input->m_data->end(), 0.0f);
    auto broadcast_strides =
        get_broadcast_strides(grad_input, grad_output->m_shape);

    size_t total_elements = sizeFromShape(grad_output->m_shape);
    std::vector<size_t> coords(grad_output->m_shape.size(), 0);

    for (size_t i = 0; i < total_elements; ++i) {
      // I think grad_input->m_offset will always be 0...
      size_t offset_view =
          get_physical_offset(coords, broadcast_strides, grad_input->m_offset);
      size_t offset_out = get_physical_offset(coords, grad_output->m_strides,
                                              grad_output->m_offset);

      (*grad_input->m_data)[offset_view] += (*grad_output->m_data)[offset_out];

      increment_coords(coords, grad_output->m_shape);
    }

    return {grad_input};
  }
};

}  // namespace

////////////////////////////////////////////////////////////////////////////////

// Kernels that perform computations and populate data for `backward`.

std::shared_ptr<TensorImpl> multiply(std::shared_ptr<TensorImpl> a,
                                     std::shared_ptr<TensorImpl> b,
                                     bool track_creator) {
  auto target_shape_opt = getBroadcastShape(a->m_shape, b->m_shape);
  if (!target_shape_opt) {
    throw std::runtime_error("Tensors are not broadcast-compatible");
  }
  Shape& target_shape = *target_shape_opt;
  // Note: if track_creator is false, broadcasting should not need to occur,
  // since the tensor shapes should be the same. This can be asserted.
  auto a_bc = (a->m_shape == target_shape)
                  ? a
                  : broadcast(a, target_shape, track_creator);
  auto b_bc = (b->m_shape == target_shape)
                  ? b
                  : broadcast(b, target_shape, track_creator);

  auto out = std::make_shared<TensorImpl>(target_shape);
  size_t total_elements = sizeFromShape(target_shape);
  std::vector<size_t> coords(target_shape.size(), 0);

  for (size_t i = 0; i < total_elements; ++i) {
    size_t offset_a =
        get_physical_offset(coords, a_bc->m_strides, a_bc->m_offset);
    size_t offset_b =
        get_physical_offset(coords, b_bc->m_strides, b_bc->m_offset);
    (*out->m_data)[i] = (*a_bc->m_data)[offset_a] * (*b_bc->m_data)[offset_b];
    increment_coords(coords, target_shape);
  }
  if (track_creator) {
    out->m_creator = std::make_unique<MulOp>(std::vector{a_bc, b_bc}, out);
  }
  return out;
}

std::shared_ptr<TensorImpl> add(std::shared_ptr<TensorImpl> a,
                                std::shared_ptr<TensorImpl> b,
                                bool track_creator) {
  auto target_shape_opt = getBroadcastShape(a->m_shape, b->m_shape);
  if (!target_shape_opt) {
    throw std::runtime_error("Tensors are not broadcast-compatible");
  }
  Shape& target_shape = *target_shape_opt;
  // Note: if track_creator is false, broadcasting should not need to occur,
  // since the tensor shapes should be the same. This can be asserted.
  auto a_bc = (a->m_shape == target_shape)
                  ? a
                  : broadcast(a, target_shape, track_creator);
  auto b_bc = (b->m_shape == target_shape)
                  ? b
                  : broadcast(b, target_shape, track_creator);

  auto out = std::make_shared<TensorImpl>(target_shape);
  size_t total_elements = sizeFromShape(target_shape);
  std::vector<size_t> coords(target_shape.size(), 0);

  for (size_t i = 0; i < total_elements; ++i) {
    size_t offset_a =
        get_physical_offset(coords, a_bc->m_strides, a_bc->m_offset);
    size_t offset_b =
        get_physical_offset(coords, b_bc->m_strides, b_bc->m_offset);
    (*out->m_data)[i] = (*a_bc->m_data)[offset_a] + (*b_bc->m_data)[offset_b];
    increment_coords(coords, target_shape);
  }
  if (track_creator) {
    out->m_creator = std::make_unique<AddOp>(std::vector{a_bc, b_bc}, out);
  }
  return out;
}

std::shared_ptr<TensorImpl> broadcast(std::shared_ptr<TensorImpl> a,
                                      const Shape& target, bool track_creator) {
  // Assumes broadcastable, not intended to be a user-facing function for now.
  // only used for implicit broadcasting.
  auto strides = get_broadcast_strides(a, target);
  auto out = std::make_shared<TensorImpl>(target, strides, a->m_data);
  out->m_offset = a->m_offset;
  if (track_creator) {
    out->m_creator = std::make_unique<BroadcastOp>(std::vector{a}, out);
  }
  return out;
}
