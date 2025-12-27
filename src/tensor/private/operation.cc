/**
 *    Author:  Vishnu Satish
 *    Created: Dec 26, 2025
 **/

#include "operation.h"

#include <memory>
#include <vector>

#include "tensor_impl.h"

// TODO: This can probably become compile-time polymorphism, unless we want to
// allow users to write Tensor functions / deep learning models at runtime.
class AddOp : public Operation {
 public:
  using Operation::Operation;

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    return {grad_output, grad_output};
  }
};

class MulOp : public Operation {
 public:
  using Operation::Operation;

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
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

////////////////////////////////////////////////////////////////////////////////

// Kernels that perform computations and populate data for `backward`.
// TODO: support broadcasting and store the state required for broadcasting in
// the Operation subclasses.

std::shared_ptr<TensorImpl> multiply(std::shared_ptr<TensorImpl> a,
                                     std::shared_ptr<TensorImpl> b,
                                     bool track_creator) {
  auto out = std::make_shared<TensorImpl>(a->m_shape);

  for (size_t i = 0; i < out->m_data->size(); ++i)
    (*out->m_data)[i] = (*a->m_data)[i] * (*b->m_data)[i];

  if (track_creator) {
    out->m_creator = std::make_unique<MulOp>(std::vector{a, b}, out);
  }

  return out;
}

std::shared_ptr<TensorImpl> add(std::shared_ptr<TensorImpl> a,
                                std::shared_ptr<TensorImpl> b,
                                bool track_creator) {
  auto out = std::make_shared<TensorImpl>(a->m_shape);

  for (size_t i = 0; i < out->m_data->size(); ++i)
    (*out->m_data)[i] = (*a->m_data)[i] + (*b->m_data)[i];

  if (track_creator) {
    out->m_creator = std::make_unique<AddOp>(std::vector{a, b}, out);
  }

  return out;
}
