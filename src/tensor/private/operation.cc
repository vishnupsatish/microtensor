/**
 *    Author:  Vishnu Satish
 *    Created: Dec 26, 2025
 **/

#include "operation.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "tensor_impl.h"

namespace {

template <typename F>
std::shared_ptr<TensorImpl> elementwiseBinaryKernel(
    std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b, F&& math_fn) {
  auto out = std::make_shared<TensorImpl>(a->m_shape);
  size_t total_elements = sizeFromShape(a->m_shape);
  std::vector<size_t> coords(a->m_shape.size(), 0);

  for (size_t i = 0; i < total_elements; ++i) {
    size_t offset_a = getPhysicalOffset(coords, a->m_strides, a->m_offset);
    size_t offset_b = getPhysicalOffset(coords, b->m_strides, b->m_offset);
    (*out->m_data)[i] = math_fn((*a->m_data)[offset_a], (*b->m_data)[offset_b]);
    incrementCoords(coords, a->m_shape);
  }
  return out;
}

template <typename F>
std::shared_ptr<TensorImpl> elementwiseUnaryKernel(
    std::shared_ptr<TensorImpl> a, F&& math_fn) {
  auto out = std::make_shared<TensorImpl>(a->m_shape);
  size_t total_elements = sizeFromShape(a->m_shape);
  std::vector<size_t> coords(a->m_shape.size(), 0);

  for (size_t i = 0; i < total_elements; ++i) {
    size_t offset_a = getPhysicalOffset(coords, a->m_strides, a->m_offset);
    (*out->m_data)[i] = math_fn((*a->m_data)[offset_a]);
    incrementCoords(coords, a->m_shape);
  }
  return out;
}

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

class TanhOp : public Operation {
 public:
  using Operation::Operation;

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    auto out = m_output.lock();
    // grad_input = grad_output * (1 - out^2)
    return {elementwiseBinaryKernel(
        grad_output, out, [](float g, float y) { return g * (1.0f - y * y); })};
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
        getBroadcastStrides(grad_input, grad_output->m_shape);

    size_t total_elements = sizeFromShape(grad_output->m_shape);
    std::vector<size_t> coords(grad_output->m_shape.size(), 0);

    for (size_t i = 0; i < total_elements; ++i) {
      // grad_input's offset will always be 0, since it is a new tensor.
      size_t offset_view =
          getPhysicalOffset(coords, broadcast_strides, grad_input->m_offset);
      size_t offset_out = getPhysicalOffset(coords, grad_output->m_strides,
                                            grad_output->m_offset);

      (*grad_input->m_data)[offset_view] += (*grad_output->m_data)[offset_out];

      incrementCoords(coords, grad_output->m_shape);
    }

    return {grad_input};
  }
};

// Helper to align two tensors to a common broadcasted shape.
std::pair<std::shared_ptr<TensorImpl>, std::shared_ptr<TensorImpl>> alignInputs(
    std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b,
    bool track_creator) {
  auto target_shape_opt = getBroadcastShape(a->m_shape, b->m_shape);
  if (!target_shape_opt) {
    throw std::runtime_error("Tensors are not broadcast-compatible");
  }
  Shape& target_shape = *target_shape_opt;

  auto a_bc = (a->m_shape == target_shape)
                  ? a
                  : broadcast(a, target_shape, track_creator);
  auto b_bc = (b->m_shape == target_shape)
                  ? b
                  : broadcast(b, target_shape, track_creator);
  return {a_bc, b_bc};
}

}  // namespace

////////////////////////////////////////////////////////////////////////////////

// Kernels that perform computations and populate data for `backward`.

std::shared_ptr<TensorImpl> multiply(std::shared_ptr<TensorImpl> a,
                                     std::shared_ptr<TensorImpl> b,
                                     bool track_creator) {
  auto [a_bc, b_bc] = alignInputs(a, b, track_creator);
  auto out = elementwiseBinaryKernel(a_bc, b_bc, std::multiplies<float>());

  if (track_creator) {
    out->m_creator = std::make_unique<MulOp>(std::vector{a_bc, b_bc}, out);
  }
  return out;
}

std::shared_ptr<TensorImpl> add(std::shared_ptr<TensorImpl> a,
                                std::shared_ptr<TensorImpl> b,
                                bool track_creator) {
  auto [a_bc, b_bc] = alignInputs(a, b, track_creator);
  auto out = elementwiseBinaryKernel(a_bc, b_bc, std::plus<float>());

  if (track_creator) {
    out->m_creator = std::make_unique<AddOp>(std::vector{a_bc, b_bc}, out);
  }
  return out;
}

std::shared_ptr<TensorImpl> tanh(std::shared_ptr<TensorImpl> a,
                                 bool track_creator) {
  auto out = elementwiseUnaryKernel(a, std::tanh<float>);

  if (track_creator) {
    out->m_creator = std::make_unique<TanhOp>(std::vector{a}, out);
  }
  return out;
}

std::shared_ptr<TensorImpl> broadcast(std::shared_ptr<TensorImpl> a,
                                      const Shape& target, bool track_creator) {
  // Assumes broadcastable, not intended to be a user-facing function for now.
  // only used for implicit broadcasting.
  auto strides = getBroadcastStrides(a, target);
  auto out = std::make_shared<TensorImpl>(target, strides, a->m_data);
  out->m_offset = a->m_offset;
  if (track_creator) {
    out->m_creator = std::make_unique<BroadcastOp>(std::vector{a}, out);
  }
  return out;
}
