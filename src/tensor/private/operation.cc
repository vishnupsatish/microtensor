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

#include "shape.h"
#include "tensor_impl.h"

// The code in this file is very coupled to the internals of TensorImpl (which,
// for now, is a struct, which makes everything public by default). The reason
// is, for much of these functions, we require a shared_ptr<TensorImpl> rather
// than just a TensorImpl. One potential solution is to inherit from
// enable_shared_from_this, but for now, I think this is an okay solution since
// users shouldn't be interfacing with TensorImpl directly anyways.

namespace {

template <typename F>
std::shared_ptr<TensorImpl> elementwiseBinaryKernel(
    std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b,
    F&& binary_fn) {
  auto out = std::make_shared<TensorImpl>(a->m_shape);
  size_t total_elements = sizeFromShape(a->m_shape);
  std::vector<size_t> coords(a->m_shape.size(), 0);

  for (size_t i = 0; i < total_elements; ++i) {
    size_t offset_a = getPhysicalOffset(coords, a->m_strides, a->m_offset);
    size_t offset_b = getPhysicalOffset(coords, b->m_strides, b->m_offset);
    (*out->m_data)[i] =
        binary_fn((*a->m_data)[offset_a], (*b->m_data)[offset_b]);
    incrementCoords(coords, a->m_shape);
  }
  return out;
}

template <typename F>
std::shared_ptr<TensorImpl> elementwiseUnaryKernel(
    std::shared_ptr<TensorImpl> a, F&& unary_fn) {
  auto out = std::make_shared<TensorImpl>(a->m_shape);
  size_t total_elements = sizeFromShape(a->m_shape);
  std::vector<size_t> coords(a->m_shape.size(), 0);

  for (size_t i = 0; i < total_elements; ++i) {
    size_t offset_a = getPhysicalOffset(coords, a->m_strides, a->m_offset);
    (*out->m_data)[i] = unary_fn((*a->m_data)[offset_a]);
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
    // TODO: is it ever possible for this to fail?
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
      // Somewhat of a hack.
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

class SqueezeOp : public Operation {
  int m_dim;

 public:
  SqueezeOp(std::vector<std::shared_ptr<TensorImpl>> parents,
            std::shared_ptr<TensorImpl> output, int dim)
      : Operation(std::move(parents), output), m_dim(dim) {}

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    auto grad_input = unsqueeze(grad_output, m_dim, false);
    return {grad_input};
  }
};

class UnsqueezeOp : public Operation {
  int m_dim;

 public:
  UnsqueezeOp(std::vector<std::shared_ptr<TensorImpl>> parents,
              std::shared_ptr<TensorImpl> output, int dim)
      : Operation(std::move(parents), output), m_dim(dim) {}

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    auto grad_input = squeeze(grad_output, m_dim, false);
    return {grad_input};
  }
};

class MatmulOp : public Operation {
 public:
  using Operation::Operation;

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    // TODO: implement.
    return {};
  }
};

////////////////////////////////////////////////////////////////////////////////

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

// Performs C = A * B for a single matrix slice
// Assumes pointers are already offset to the correct batch location
void matmul_2d_kernel(float* ptr_c, const float* ptr_a, const float* ptr_b,
                      size_t m, size_t k, size_t n, size_t stride_a_m,
                      size_t stride_a_k,  // Strides for A's last 2 dims
                      size_t stride_b_k,
                      size_t stride_b_n,  // Strides for B's last 2 dims
                      size_t stride_c_m,
                      size_t stride_c_n  // Strides for C's last 2 dims
) {
  for (size_t i = 0; i < m; ++i) {    // rows of A
    for (size_t j = 0; j < n; ++j) {  // columns of B

      float sum = 0.0f;

      for (size_t p = 0; p < k; ++p) {
        float val_a = ptr_a[i * stride_a_m + p * stride_a_k];
        float val_b = ptr_b[p * stride_b_k + j * stride_b_n];
        sum += val_a * val_b;
      }

      ptr_c[i * stride_c_m + j * stride_c_n] = sum;
    }
  }
}

void matmul_batched(std::shared_ptr<TensorImpl> c,
                    std::shared_ptr<TensorImpl> a,
                    std::shared_ptr<TensorImpl> b) {
  size_t rank = c->getRank();

  size_t batch_rank = rank - 2;
  size_t m = c->m_shape[rank - 2];
  size_t n = c->m_shape[rank - 1];

  size_t k = a->m_shape[rank - 1];

  // Tracks where we are in the batch dims.
  std::vector<size_t> current_coords(batch_rank, 0);
  Shape batch_shape(c->m_shape.begin(), c->m_shape.end() - 2);

  // Calculate total number of batches (e.g., 10 * 5 = 50)
  size_t total_batches = 1;
  for (size_t i = 0; i < batch_rank; ++i) total_batches *= batch_shape[i];

  // 3. The Main Loop
  for (size_t batch = 0; batch < total_batches; ++batch) {
    size_t offset_a = a->m_offset;
    size_t offset_b = b->m_offset;
    size_t offset_c = c->m_offset;

    // Use the current coordinates to find the start of the data
    for (size_t i = 0; i < batch_rank; ++i) {
      offset_a += current_coords[i] * a->m_strides[i];
      offset_b += current_coords[i] * b->m_strides[i];
      offset_c += current_coords[i] * c->m_strides[i];
    }

    float* ptr_a = a->m_data->data() + offset_a;
    float* ptr_b = b->m_data->data() + offset_b;
    float* ptr_c = c->m_data->data() + offset_c;
    matmul_2d_kernel(ptr_c, ptr_a, ptr_b, m, k, n, a->m_strides[rank - 2],
                     a->m_strides[rank - 1], b->m_strides[rank - 2],
                     b->m_strides[rank - 1], c->m_strides[rank - 2],
                     c->m_strides[rank - 1]);

    incrementCoords(current_coords, batch_shape);  // [0,0] -> [0,1]
  }
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
  // Share the data.
  auto out = std::make_shared<TensorImpl>(target, strides, a->m_data);
  out->m_offset = a->m_offset;
  if (track_creator) {
    out->m_creator = std::make_unique<BroadcastOp>(std::vector{a}, out);
  }
  return out;
}

std::shared_ptr<TensorImpl> squeeze(std::shared_ptr<TensorImpl> a, int dimInput,
                                    bool track_creator) {
  size_t dim = dimInput >= 0 ? dimInput : a->getRank() + dimInput;
  auto shape = a->m_shape;
  auto strides = a->m_strides;
  if (shape[dim] != 1) {
    return a;
  }
  shape.erase(shape.begin() + dim);
  strides.erase(shape.begin() + dim);
  auto out = std::make_shared<TensorImpl>(shape, strides, a->m_data);
  if (track_creator) {
    out->m_creator = std::make_unique<SqueezeOp>(std::vector{a}, out, dim);
  }
  return out;
}

std::shared_ptr<TensorImpl> unsqueeze(std::shared_ptr<TensorImpl> a,
                                      int dimInput, bool track_creator) {
  size_t dim = dimInput >= 0 ? dimInput : a->getRank() + dimInput + 1;
  auto shape = a->m_shape;
  auto strides = a->m_strides;
  shape.insert(shape.begin() + dim, 1);
  strides.insert(strides.begin() + dim, 0);
  auto out = std::make_shared<TensorImpl>(shape, strides, a->m_data);
  if (track_creator) {
    out->m_creator = std::make_unique<UnsqueezeOp>(std::vector{a}, out, dim);
  }
  return out;
}

std::shared_ptr<TensorImpl> matmul(std::shared_ptr<TensorImpl> a,
                                   std::shared_ptr<TensorImpl> b,
                                   bool track_creator) {
  if (a->getRank() == 0 || b->getRank() == 0) {
    throw std::runtime_error("Both tensors must be at least rank-1");
  }

  // Following the rules provided in
  // https://docs.pytorch.org/docs/stable/generated/torch.matmul.html.
  auto a_2d = a->getRank() == 1 ? unsqueeze(a, -2, track_creator) : a;
  auto b_2d = b->getRank() == 1 ? unsqueeze(b, -1, track_creator) : b;

  // Broadcast the first rank-2 dimensions
  auto shapesOpt = getBroadcastShapesForMatmul(a_2d->m_shape, b_2d->m_shape);
  if (!shapesOpt) {
    throw std::runtime_error("Incompatible shapes for matrix multiplication");
  }
  auto [shape_a_2d, shape_b_2d] = *shapesOpt;
  // Note: shares the same data as `a` and `b`.
  auto a_bc = shape_a_2d != a_2d->m_shape
                  ? broadcast(a_2d, shape_a_2d, track_creator)
                  : a_2d;
  auto b_bc = shape_b_2d != b_2d->m_shape
                  ? broadcast(b_2d, shape_b_2d, track_creator)
                  : b_2d;

  // It holds that a_bc and b_bc are both at least two dimensions, and the
  // shapes are matmul-compatible.
  auto outShape = Shape(a_bc->m_shape.begin(), a_bc->m_shape.end() - 2);
  outShape.push_back(a_bc->m_shape[a_bc->m_shape.size() - 2]);
  outShape.push_back(b_bc->m_shape[b_bc->m_shape.size() - 1]);
  auto matmulRes = std::make_shared<TensorImpl>(outShape);

  matmul_batched(matmulRes, a_bc, b_bc);
  if (track_creator) {
    matmulRes->m_creator =
        std::make_unique<MatmulOp>(std::vector{a_bc, b_bc}, matmulRes);
  }

  if (a->getRank() == 1) {
    // Note: the old matmulRes still exists if track_creator is enabled, as a
    // shared pointer.
    matmulRes = squeeze(matmulRes, -2, track_creator);
  }
  if (b->getRank() == 1) {
    matmulRes = squeeze(matmulRes, -1, track_creator);
  }
  return matmulRes;
}
