/**
 *    Author:  Vishnu Satish
 *    Created: Dec 26, 2025
 **/

#include "operation.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "grad_mode.h"
#include "kernels.h"
#include "shape.h"
#include "tensor_impl.h"

// The code in this file is very coupled to the internals of TensorImpl (which,
// for now, is a struct, which makes everything public by default). The reason
// is, for much of these functions, we require a shared_ptr<TensorImpl> rather
// than just a TensorImpl. One potential solution is to inherit from
// enable_shared_from_this, but for now, I think this is an okay solution since
// users shouldn't be interfacing with TensorImpl directly anyways.

namespace {

class AddOp : public Operation {
 public:
  using Operation::Operation;

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    NoGrad guard;
    // If shapes are not the same, handled by BroadcastOp.
    assert(m_parents[0]->m_shape == m_parents[1]->m_shape);
    return {grad_output, grad_output};
  }
};

class SubtractOp : public Operation {
 public:
  using Operation::Operation;

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    NoGrad guard;
    // If shapes are not the same, handled by BroadcastOp.
    assert(m_parents[0]->m_shape == m_parents[1]->m_shape);
    auto negGrad = std::make_shared<TensorImpl>(m_parents[0]->m_shape);
    elementwiseUnaryKernel(negGrad, grad_output, [](float g) { return -g; });
    return {grad_output, negGrad};
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
    NoGrad guard;
    auto grad_op1 = multiply(grad_output, m_parents[1]);
    auto grad_op2 = multiply(grad_output, m_parents[0]);
    return {grad_op1, grad_op2};
  }
};

class DivOp : public Operation {
 public:
  using Operation::Operation;

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    NoGrad guard;
    assert(m_parents[0]->m_shape == m_parents[1]->m_shape);
    // dividing => a/b
    auto num = m_parents[0];
    auto denom = m_parents[1];
    // grad * 1/b
    auto grad_op1 = multiply(grad_output, divide(1.0f, denom));
    // grad * -(a/b^2)
    auto div = divide(num, pow(denom, 2));
    auto neg = std::make_shared<TensorImpl>(div->m_shape);
    elementwiseUnaryKernel(neg, div, [](float x) { return -x; });
    auto grad_op2 = multiply(grad_output, neg);
    return {grad_op1, grad_op2};
  }
};

class PowOp : public Operation {
  float m_exp;

 public:
  PowOp(std::vector<std::shared_ptr<TensorImpl>> parents,
        std::shared_ptr<TensorImpl> output, float exp)
      : Operation(std::move(parents), output), m_exp(exp) {}

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    NoGrad guard;
    auto inp = m_parents[0];
    // TODO: use elementwiseBinaryKernel for efficiency.
    auto gradS1 = multiply(pow(inp, m_exp - 1), m_exp);
    auto gradRes = multiply(grad_output, gradS1);
    return {gradRes};
  }
};

class TanhOp : public Operation {
 public:
  using Operation::Operation;

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    NoGrad guard;
    // TODO: is it ever possible for this to fail?
    auto out = m_output.lock();
    // grad_input = grad_output * (1 - out^2)
    auto res = std::make_shared<TensorImpl>(grad_output->m_shape);
    elementwiseBinaryKernel(res, grad_output, out, [](float g, float y) {
      return g * (1.0f - y * y);
    });
    return {res};
  }
};

class ExpOp : public Operation {
 public:
  using Operation::Operation;

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    NoGrad guard;
    // TODO: is it ever possible for this to fail?
    // Don't recalculate e^x again.
    auto out = m_output.lock();
    auto ret = multiply(out, grad_output);
    return {ret};
  }
};

class LogOp : public Operation {
 public:
  using Operation::Operation;

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    NoGrad guard;
    auto inp = m_parents[0];
    return {divide(grad_output, inp)};
  }
};

class BroadcastOp : public Operation {
 public:
  using Operation::Operation;

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    NoGrad guard;
    // For any prepended dims, we need to reduce and not keep dims. For
    // non-prepended dims that were broadcasted, reduce and keep dims.
    auto input = m_parents[0];
    int diff = grad_output->getRank() - input->getRank();
    std::vector<int> prependedDims;
    for (int i = 0; i < diff; ++i) {
      prependedDims.push_back(i);
    }
    auto rmPrepended = reduceSum(grad_output, prependedDims, false);
    std::vector<int> dimsToReduce;
    for (size_t i = 0; i < input->getRank(); ++i) {
      if (input->m_shape[i] == 1 && rmPrepended->m_shape[i] > 1) {
        dimsToReduce.push_back(i);
      }
    }
    auto grad_input = reduceSum(rmPrepended, dimsToReduce, true);
    return {grad_input};
  }
};

class SqueezeOp : public Operation {
  std::vector<int> m_dims;

 public:
  SqueezeOp(std::vector<std::shared_ptr<TensorImpl>> parents,
            std::shared_ptr<TensorImpl> output, std::vector<int> dims)
      : Operation(std::move(parents), output), m_dims(std::move(dims)) {}

  std::vector<std::shared_ptr<TensorImpl>> backward(

      std::shared_ptr<TensorImpl> grad_output) override {
    NoGrad guard;
    auto grad_input = unsqueeze(grad_output, m_dims);
    return {grad_input};
  }
};

class UnsqueezeOp : public Operation {
  std::vector<int> m_dims;

 public:
  UnsqueezeOp(std::vector<std::shared_ptr<TensorImpl>> parents,
              std::shared_ptr<TensorImpl> output, std::vector<int> dims)
      : Operation(std::move(parents), output), m_dims(std::move(dims)) {}

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    NoGrad guard;
    auto grad_input = squeeze(grad_output, m_dims);
    return {grad_input};
  }
};
class MatmulOp : public Operation {
 public:
  using Operation::Operation;

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    NoGrad guard;
    // Invariant: inputs are matmul-compatible without the need for any
    // broadcasting/squeezing/unsqueezing.
    size_t rank = grad_output->getRank();
    auto a = m_parents[0];
    auto b = m_parents[1];
    assert(rank == a->getRank());
    assert(rank == b->getRank());
    // Transpose the matrix; only concerns the last two dimensions.
    std::vector<size_t> permuteDims;
    permuteDims.reserve(rank);
    for (size_t i = 0; i < rank - 2; ++i) {
      permuteDims.push_back(i);
    }
    permuteDims.push_back(rank - 1);
    permuteDims.push_back(rank - 2);
    auto aTranspose = permute(a, permuteDims);
    auto bTranspose = permute(b, permuteDims);
    auto dA = matmul(grad_output, bTranspose);
    auto dB = matmul(aTranspose, grad_output);
    return {dA, dB};
  }
};

class PermuteOp : public Operation {
  std::vector<size_t> m_dims;

 public:
  PermuteOp(std::vector<std::shared_ptr<TensorImpl>> parents,
            std::shared_ptr<TensorImpl> output, std::vector<size_t> dims)
      : Operation(std::move(parents), output), m_dims(std::move(dims)) {}

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    NoGrad guard;
    std::vector<size_t> reverseDims(m_dims.size());
    for (size_t i = 0; i < m_dims.size(); ++i) {
      reverseDims[m_dims[i]] = i;
    }
    auto grad_input = permute(grad_output, reverseDims);
    return {grad_input};
  }
};

class ReduceSumOp : public Operation {
 public:
  using Operation::Operation;

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    NoGrad guard;
    auto input = m_parents[0];
    return {broadcast(grad_output, input->m_shape)};
  }
};

// Important note: PyTorch implements the backward pass of reduce max in a
// different way. If there are equal values along a particular dimension, it
// distributes the gradient equally (and divides by the count) rather than just
// sending it to the first one like we do here. Therefore, when differential
// testing, we cannot test the backward pass against PyTorch if there are
// several values that are the same in the dimension that is being reduced.
class ReduceMaxOp : public Operation {
  size_t m_dim;
  std::shared_ptr<TensorImpl> m_argmax;

 public:
  ReduceMaxOp(std::vector<std::shared_ptr<TensorImpl>> parents,
              std::shared_ptr<TensorImpl> output, size_t dim,
              std::shared_ptr<TensorImpl> argmax)
      : Operation(std::move(parents), output),
        m_dim{dim},
        m_argmax(std::move(argmax)) {}

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    NoGrad guard;
    auto input = m_parents[0];
    auto grad = std::make_shared<TensorImpl>(input->m_shape);
    auto gradStrides = grad->m_strides;
    size_t reducedStride = gradStrides[m_dim];
    // Set stride of reduced dim to 0, we will add it later based on the argmax
    // value (represents the coord).
    gradStrides[m_dim] = 0;
    std::vector<size_t> coords(m_argmax->getRank());

    // Idea: argmax contains the "missing coordinate", which represents where
    // the max item is, and therefore where to place the downstream gradient
    // value. Loop argmax coordinates, use the value to increment the physical
    // offset of the calculated offset based on gradStrides. We're just finding
    // the place where to put the gradient.
    size_t totalElements = sizeFromShape(m_argmax->m_shape);
    for (size_t i = 0; i < totalElements; ++i) {
      size_t argmaxOffset =
          getPhysicalOffset(coords, m_argmax->m_strides, m_argmax->m_offset);
      size_t downstreamGradOffset = getPhysicalOffset(
          coords, grad_output->m_strides, grad_output->m_offset);
      // Which coord does the max element come from, in the input?
      size_t reducedCoordVal = (*m_argmax->m_data)[argmaxOffset];
      // grad: the return value. downstream grad: gradient we get as function
      // param. downstream grad has the same shape as the reduced value.

      // Determines where the current downstream grad value should go
      size_t gradOffset =
          getPhysicalOffset(coords, gradStrides, grad->m_offset) +
          reducedStride * reducedCoordVal;
      (*grad->m_data)[gradOffset] =
          (*grad_output->m_data)[downstreamGradOffset];
      incrementCoords(coords, m_argmax->m_shape);
    }
    return {grad};
  }
};

class MakeContiguousOp : public Operation {
 public:
  using Operation::Operation;

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    NoGrad guard;
    return {grad_output};
  }
};

class ReshapeOp : public Operation {
 public:
  using Operation::Operation;

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    NoGrad guard;
    auto ret = reshape(grad_output, m_parents[0]->m_shape);
    return {ret};
  }
};

class SliceOp : public Operation {
  std::vector<int> m_start;
  Shape m_size;

 public:
  SliceOp(std::vector<std::shared_ptr<TensorImpl>> parents,
          std::shared_ptr<TensorImpl> output, std::vector<int> start,
          Shape size)
      : Operation(std::move(parents), output),
        m_start{std::move(start)},
        m_size(std::move(size)) {}

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    NoGrad guard;
    auto input = m_parents[0];
    Shape& inputShape = input->m_shape;
    auto ret = std::make_shared<TensorImpl>(inputShape);
    std::fill(ret->m_data->begin(), ret->m_data->end(), 0.0f);
    // should be 0
    size_t offset = ret->m_offset;
    for (size_t i = 0; i < m_start.size(); ++i) {
      offset += m_start[i] * ret->m_strides[i];
    }
    assert(m_size == grad_output->m_shape);
    // We've artifically created a "view" of ret that includes only the sliced
    // part; just consider offset, ret's strides, and m_size as the shape.
    size_t totalElements = sizeFromShape(m_size);
    std::vector<size_t> coords(input->getRank());
    for (size_t i = 0; i < totalElements; ++i) {
      auto gradOffset = getPhysicalOffset(coords, grad_output->m_strides,
                                          grad_output->m_offset);
      // Where to place the current gradient value
      auto retOffset = getPhysicalOffset(coords, ret->m_strides, offset);
      (*ret->m_data)[retOffset] = (*grad_output->m_data)[gradOffset];
      incrementCoords(coords, m_size);
    }
    return {ret};
  }
};

class ReLUOp : public Operation {
 public:
  using Operation::Operation;

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    NoGrad guard;
    auto ret = std::make_shared<TensorImpl>(m_parents[0]->m_shape);
    elementwiseBinaryKernel(
        ret, m_parents[0], grad_output,
        [](float par, float grad) { return par >= 0 ? grad : 0; });
    return {ret};
  }
};

class TriuOp : public Operation {
  int m_diagonal;

 public:
  TriuOp(std::vector<std::shared_ptr<TensorImpl>> parents,
         std::shared_ptr<TensorImpl> output, int diagonal)
      : Operation(std::move(parents), output), m_diagonal{diagonal} {}

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    NoGrad guard;
    return {triu(grad_output, m_diagonal)};
  }
};

class GeluOp : public Operation {
 public:
  using Operation::Operation;

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    NoGrad guard;
    auto input = m_parents[0];
    auto res = std::make_shared<TensorImpl>(input->m_shape);
    const float sqrt2Pi = 0.7978845608f;
    const float coeff = 0.044715f;
    elementwiseBinaryKernel(res, input, grad_output, [&](float x, float g_out) {
      float x2 = x * x;
      float x3 = x2 * x;
      float inner = sqrt2Pi * (x + coeff * x3);
      float t = std::tanh(inner);
      float g_prime = sqrt2Pi * (1.0f + 3.0f * coeff * x2);
      float sech2 = 1.0f - t * t;
      float derivative = 0.5f * (1.0f + t) + 0.5f * x * sech2 * g_prime;
      return g_out * derivative;
    });
    return {res};
  }
};

class GatherOp : public Operation {
  int m_dim;

 public:
  GatherOp(std::vector<std::shared_ptr<TensorImpl>> parents,
           std::shared_ptr<TensorImpl> output, int dim)
      : Operation(std::move(parents), output), m_dim{dim} {}

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    NoGrad guard;
    auto& inp = m_parents[0];
    auto& idx = m_parents[1];
    auto out = std::make_shared<TensorImpl>(inp->m_shape);
    out->fillRandom([]() { return 0; });
    std::vector<size_t> coords(inp->getRank());
    size_t totalElements = sizeFromShape(idx->m_shape);
    for (int i = 0; i < totalElements; ++i) {
      size_t idxOffset =
          getPhysicalOffset(coords, idx->m_strides, idx->m_offset);
      size_t gradOffset = getPhysicalOffset(coords, grad_output->m_strides,
                                            grad_output->m_offset);
      // Where the value comes from (and gradient should go).
      size_t val = static_cast<size_t>((*idx->m_data)[idxOffset]);
      size_t outOffset =
          getPhysicalOffset(coords, out->m_strides, out->m_offset);
      outOffset -= coords[m_dim] * out->m_strides[m_dim];
      outOffset += val * out->m_strides[m_dim];
      (*out->m_data)[outOffset] += (*grad_output->m_data)[gradOffset];
      incrementCoords(coords, idx->m_shape);
    }
    return {out, nullptr};
  }
};

class MaskedFillOp : public Operation {
  float m_val;

 public:
  MaskedFillOp(std::vector<std::shared_ptr<TensorImpl>> parents,
               std::shared_ptr<TensorImpl> output, float val)
      : Operation(std::move(parents), output), m_val{val} {}

  std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) override {
    NoGrad guard;
    // Note: in the forward implementation we assert that the mask does not
    // require a grad, but the autograd asserts that the number of inputs (as
    // provided in m_parents) is equal to the number of gradients we return from
    // this function. Therefore, we provide a nullptr as the gradient associated
    // with the mask, as it will never be used.
    return {maskedFill(grad_output, m_parents[1], 0), nullptr};
  }
};

////////////////////////////////////////////////////////////////////////////////

// Helper to align two tensors to a common broadcasted shape.
std::pair<std::shared_ptr<TensorImpl>, std::shared_ptr<TensorImpl>> alignInputs(
    std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b) {
  auto target_shape_opt = getBroadcastShape(a->m_shape, b->m_shape);
  if (!target_shape_opt) {
    throw std::runtime_error("Tensors are not broadcast-compatible");
  }
  Shape& target_shape = *target_shape_opt;

  auto a_bc = (a->m_shape == target_shape) ? a : broadcast(a, target_shape);
  auto b_bc = (b->m_shape == target_shape) ? b : broadcast(b, target_shape);
  return {a_bc, b_bc};
}

// Performs C = A * B for a single matrix slice
// Assumes pointers are already offset to the correct batch location
void matmulKernel(float* ptr_c, const float* ptr_a, const float* ptr_b,
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

void matmulBatched(std::shared_ptr<TensorImpl> c, std::shared_ptr<TensorImpl> a,
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
    matmulKernel(ptr_c, ptr_a, ptr_b, m, k, n, a->m_strides[rank - 2],
                 a->m_strides[rank - 1], b->m_strides[rank - 2],
                 b->m_strides[rank - 1], c->m_strides[rank - 2],
                 c->m_strides[rank - 1]);

    incrementCoords(current_coords, batch_shape);
  }
}

bool isPermutation(const std::vector<size_t>& v) {
  int n = v.size();
  std::vector<bool> seen(n, false);

  for (int x : v) {
    if (x < 0 || x >= n) return false;
    if (seen[x]) return false;
    seen[x] = true;
  }
  return true;
}

}  // namespace

////////////////////////////////////////////////////////////////////////////////

// Kernels that perform computations and populate data for `backward`.

std::shared_ptr<TensorImpl> multiply(std::shared_ptr<TensorImpl> a,
                                     std::shared_ptr<TensorImpl> b) {
  auto [a_bc, b_bc] = alignInputs(a, b);
  auto out = std::make_shared<TensorImpl>(a_bc->m_shape);
  elementwiseBinaryKernel(out, a_bc, b_bc, std::multiplies<float>());
  out->m_requiresGrad =
      GradMode::enabled && (a->m_requiresGrad || b->m_requiresGrad);
  if (out->m_requiresGrad) {
    out->m_creator = std::make_unique<MulOp>(std::vector{a_bc, b_bc}, out);
  }
  return out;
}

std::shared_ptr<TensorImpl> multiply(std::shared_ptr<TensorImpl> a, float cst) {
  auto b = std::make_shared<TensorImpl>(Shape{}, std::vector<float>{cst});
  return multiply(a, b);
}

std::shared_ptr<TensorImpl> divide(std::shared_ptr<TensorImpl> a,
                                   std::shared_ptr<TensorImpl> b) {
  auto [a_bc, b_bc] = alignInputs(a, b);
  auto out = std::make_shared<TensorImpl>(a_bc->m_shape);
  elementwiseBinaryKernel(out, a_bc, b_bc, std::divides<float>());
  out->m_requiresGrad =
      GradMode::enabled && (a->m_requiresGrad || b->m_requiresGrad);
  if (out->m_requiresGrad) {
    out->m_creator = std::make_unique<DivOp>(std::vector{a_bc, b_bc}, out);
  }
  return out;
}

std::shared_ptr<TensorImpl> divide(std::shared_ptr<TensorImpl> a, float denom) {
  auto cst = make_shared<TensorImpl>(Shape{}, std::vector<float>{denom});
  return divide(a, cst);
}

std::shared_ptr<TensorImpl> divide(float num, std::shared_ptr<TensorImpl> b) {
  auto cst = make_shared<TensorImpl>(Shape{}, std::vector<float>{num});
  return divide(cst, b);
}

std::shared_ptr<TensorImpl> pow(std::shared_ptr<TensorImpl> a, float exp) {
  auto out = std::make_shared<TensorImpl>(a->m_shape);
  elementwiseUnaryKernel(out, a, [&](float elt) { return std::pow(elt, exp); });
  out->m_requiresGrad = GradMode::enabled && (a->m_requiresGrad);
  if (out->m_requiresGrad) {
    out->m_creator = std::make_unique<PowOp>(std::vector{a}, out, exp);
  }
  return out;
}

std::shared_ptr<TensorImpl> add(std::shared_ptr<TensorImpl> a,
                                std::shared_ptr<TensorImpl> b) {
  auto [a_bc, b_bc] = alignInputs(a, b);
  auto out = std::make_shared<TensorImpl>(a_bc->m_shape);
  elementwiseBinaryKernel(out, a_bc, b_bc, std::plus<float>());
  out->m_requiresGrad =
      GradMode::enabled && (a->m_requiresGrad || b->m_requiresGrad);
  if (out->m_requiresGrad) {
    out->m_creator = std::make_unique<AddOp>(std::vector{a_bc, b_bc}, out);
  }
  return out;
}

std::shared_ptr<TensorImpl> add(std::shared_ptr<TensorImpl> a, float b) {
  auto cst = make_shared<TensorImpl>(Shape{}, std::vector<float>{b});
  return add(a, cst);
}

std::shared_ptr<TensorImpl> subtract(std::shared_ptr<TensorImpl> a,
                                     std::shared_ptr<TensorImpl> b) {
  auto [a_bc, b_bc] = alignInputs(a, b);
  auto out = std::make_shared<TensorImpl>(a_bc->m_shape);
  elementwiseBinaryKernel(out, a_bc, b_bc, std::minus<float>());
  out->m_requiresGrad =
      GradMode::enabled && (a->m_requiresGrad || b->m_requiresGrad);
  if (out->m_requiresGrad) {
    out->m_creator = std::make_unique<SubtractOp>(std::vector{a_bc, b_bc}, out);
  }
  return out;
}

std::shared_ptr<TensorImpl> tanh(std::shared_ptr<TensorImpl> a) {
  auto out = std::make_shared<TensorImpl>(a->m_shape);
  elementwiseUnaryKernel(out, a, std::tanh<float>);
  out->m_requiresGrad = GradMode::enabled && (a->m_requiresGrad);
  if (out->m_requiresGrad) {
    out->m_creator = std::make_unique<TanhOp>(std::vector{a}, out);
  }
  return out;
}

std::shared_ptr<TensorImpl> exp(std::shared_ptr<TensorImpl> a) {
  auto out = std::make_shared<TensorImpl>(a->m_shape);
  elementwiseUnaryKernel(out, a, std::exp<float>);
  out->m_requiresGrad = GradMode::enabled && (a->m_requiresGrad);
  if (out->m_requiresGrad) {
    out->m_creator = std::make_unique<ExpOp>(std::vector{a}, out);
  }
  return out;
}

std::shared_ptr<TensorImpl> log(std::shared_ptr<TensorImpl> a) {
  auto out = std::make_shared<TensorImpl>(a->m_shape);
  elementwiseUnaryKernel(out, a, std::log<float>);
  out->m_requiresGrad = GradMode::enabled && (a->m_requiresGrad);
  if (out->m_requiresGrad) {
    out->m_creator = std::make_unique<LogOp>(std::vector{a}, out);
  }
  return out;
}

std::shared_ptr<TensorImpl> broadcast(std::shared_ptr<TensorImpl> a,
                                      const Shape& target) {
  // Assumes broadcastable, not intended to be a user-facing function for now.
  // only used for implicit broadcasting.
  auto strides = getBroadcastStrides(a, target);
  auto out = std::make_shared<TensorImpl>(target, strides, a->m_data);
  out->m_offset = a->m_offset;
  out->m_requiresGrad = GradMode::enabled && (a->m_requiresGrad);
  if (out->m_requiresGrad) {
    out->m_creator = std::make_unique<BroadcastOp>(std::vector{a}, out);
  }
  return out;
}

std::shared_ptr<TensorImpl> squeeze(std::shared_ptr<TensorImpl> a,
                                    std::vector<int> dimInputs) {
  auto shape = a->m_shape;
  auto strides = a->m_strides;
  std::vector<int> normalizedDims;
  for (auto dimInput : dimInputs) {
    int dim = dimInput >= 0 ? dimInput : a->getRank() + dimInput;
    if (shape[dim] != 1) {
      // PyTorch semantics; does not error, just continues and does not squeeze
      // that dim.
      continue;
    }
    normalizedDims.push_back(dim);
  }
  sort(normalizedDims.rbegin(), normalizedDims.rend());
  for (auto dim : normalizedDims) {
    shape.erase(shape.begin() + dim);
    strides.erase(strides.begin() + dim);
  }
  auto out = std::make_shared<TensorImpl>(shape, strides, a->m_data);
  out->m_requiresGrad = GradMode::enabled && (a->m_requiresGrad);
  if (out->m_requiresGrad) {
    out->m_creator =
        std::make_unique<SqueezeOp>(std::vector{a}, out, normalizedDims);
  }
  return out;
}

std::shared_ptr<TensorImpl> unsqueeze(std::shared_ptr<TensorImpl> a,
                                      std::vector<int> dimInputs) {
  auto shape = a->m_shape;
  auto strides = a->m_strides;
  std::vector<int> normalizedDims;
  for (auto dimInput : dimInputs) {
    int dim = dimInput >= 0 ? dimInput : a->getRank() + dimInput + 1;
    normalizedDims.push_back(dim);
  }
  sort(normalizedDims.begin(), normalizedDims.end());
  for (auto dim : normalizedDims) {
    shape.insert(shape.begin() + dim, 1);
    strides.insert(strides.begin() + dim, 0);
  }
  auto out = std::make_shared<TensorImpl>(shape, strides, a->m_data);
  out->m_requiresGrad = GradMode::enabled && (a->m_requiresGrad);
  if (out->m_requiresGrad) {
    out->m_creator =
        std::make_unique<UnsqueezeOp>(std::vector{a}, out, normalizedDims);
  }
  return out;
}

std::shared_ptr<TensorImpl> matmul(std::shared_ptr<TensorImpl> a,
                                   std::shared_ptr<TensorImpl> b) {
  if (a->getRank() == 0 || b->getRank() == 0) {
    throw std::runtime_error("Both tensors must be at least rank-1");
  }

  // Following the rules provided in
  // https://docs.pytorch.org/docs/stable/generated/torch.matmul.html.
  auto a_2d = a->getRank() == 1 ? unsqueeze(a, {-2}) : a;
  auto b_2d = b->getRank() == 1 ? unsqueeze(b, {-1}) : b;

  // Broadcast the first rank-2 dimensions
  auto shapesOpt = getBroadcastShapesForMatmul(a_2d->m_shape, b_2d->m_shape);
  if (!shapesOpt) {
    throw std::runtime_error("Incompatible shapes for matrix multiplication");
  }
  auto [shape_a_2d, shape_b_2d] = *shapesOpt;
  // Note: shares the same data as `a` and `b`.
  auto a_bc = shape_a_2d != a_2d->m_shape ? broadcast(a_2d, shape_a_2d) : a_2d;
  auto b_bc = shape_b_2d != b_2d->m_shape ? broadcast(b_2d, shape_b_2d) : b_2d;

  // It holds that a_bc and b_bc are both at least two dimensions, and the
  // shapes are matmul-compatible.
  auto outShape = Shape(a_bc->m_shape.begin(), a_bc->m_shape.end() - 2);
  outShape.push_back(a_bc->m_shape[a_bc->m_shape.size() - 2]);
  outShape.push_back(b_bc->m_shape[b_bc->m_shape.size() - 1]);
  auto matmulRes = std::make_shared<TensorImpl>(outShape);

  matmulBatched(matmulRes, a_bc, b_bc);

  matmulRes->m_requiresGrad =
      GradMode::enabled && (a_bc->m_requiresGrad || b_bc->m_requiresGrad);
  if (matmulRes->m_requiresGrad) {
    matmulRes->m_creator =
        std::make_unique<MatmulOp>(std::vector{a_bc, b_bc}, matmulRes);
  }

  std::vector<int> squeezeDims;
  if (a->getRank() == 1) {
    squeezeDims.push_back(-2);
  }
  if (b->getRank() == 1) {
    squeezeDims.push_back(-1);
  }
  if (!squeezeDims.empty()) {
    matmulRes = squeeze(matmulRes, squeezeDims);
  }
  return matmulRes;
}

std::shared_ptr<TensorImpl> permute(std::shared_ptr<TensorImpl> a,
                                    std::vector<size_t> dims) {
  size_t rank = a->getRank();
  if (dims.size() != rank || !isPermutation(dims)) {
    throw std::runtime_error("Incorrect dims value passed to permute");
  }
  Shape newShape(rank);
  std::vector<size_t> newStrides(rank);
  for (size_t i = 0; i < rank; ++i) {
    newShape[i] = a->m_shape[dims[i]];
    newStrides[i] = a->m_strides[dims[i]];
  }
  auto out = std::make_shared<TensorImpl>(newShape, newStrides, a->m_data);
  out->m_offset = a->m_offset;
  out->m_requiresGrad = GradMode::enabled && (a->m_requiresGrad);
  if (out->m_requiresGrad) {
    out->m_creator = std::make_unique<PermuteOp>(std::vector{a}, out, dims);
  }
  return out;
}

std::shared_ptr<TensorImpl> reduceSum(std::shared_ptr<TensorImpl> a,
                                      std::vector<int> dimInputs,
                                      bool keep_dims) {
  Shape outShape = a->m_shape;
  std::vector<int> normalizedDims;
  for (auto dimInput : dimInputs) {
    int dim = dimInput >= 0 ? dimInput : a->getRank() + dimInput;
    // TODO: check if `dim` invalid.
    normalizedDims.push_back(dim);
    outShape[dim] = 1;
  }
  auto out = std::make_shared<TensorImpl>(outShape);
  std::fill(out->m_data->begin(), out->m_data->end(), 0.0f);
  // Get the strides as if we are broadcasting `out` to the input's shape.
  // This sets stride value to 0 wherever `out`'s dimension size is 1.
  // Therefore, when iterating, we just need to iterate on the input's coords,
  // and use these strides (which match up with where we want to accumulate
  // values).
  auto broadcast_strides = getBroadcastStrides(out, a->m_shape);

  size_t total_elements = sizeFromShape(a->m_shape);
  std::vector<size_t> coords(a->getRank(), 0);

  for (size_t i = 0; i < total_elements; ++i) {
    // out's offset will always be 0, since it is a new tensor.
    // Somewhat of a hack since we use broadcasted strides but we're not
    // actually broadcasting anything.
    size_t offset_view =
        getPhysicalOffset(coords, broadcast_strides, out->m_offset);
    size_t offset_out = getPhysicalOffset(coords, a->m_strides, a->m_offset);
    (*out->m_data)[offset_view] += (*a->m_data)[offset_out];
    incrementCoords(coords, a->m_shape);
  }
  out->m_requiresGrad = GradMode::enabled && (a->m_requiresGrad);
  if (out->m_requiresGrad) {
    out->m_creator = std::make_unique<ReduceSumOp>(std::vector{a}, out);
  }
  // If `keep_dims` is false, performs a squeeze, so the backward pass of
  // `reduceSum` can assume that any reduced dimensions are set to 1.
  if (!keep_dims) {
    out = squeeze(out, normalizedDims);
  }
  return out;
}

std::shared_ptr<TensorImpl> reduceMax(std::shared_ptr<TensorImpl> a,
                                      int dimInput, bool keep_dim) {
  Shape outShape = a->m_shape;
  int dim = dimInput >= 0 ? dimInput : a->getRank() + dimInput;
  outShape[dim] = 1;
  auto out = std::make_shared<TensorImpl>(outShape);
  // Used for the backward pass.
  auto argmax = std::make_shared<TensorImpl>(outShape);
  std::fill(out->m_data->begin(), out->m_data->end(),
            -std::numeric_limits<float>::infinity());
  auto broadcast_strides = getBroadcastStrides(out, a->m_shape);

  size_t total_elements = sizeFromShape(a->m_shape);
  std::vector<size_t> coords(a->getRank(), 0);

  for (size_t i = 0; i < total_elements; ++i) {
    // out's offset will always be 0, since it is a new tensor.
    // Somewhat of a hack since we use broadcasted strides but we're not
    // actually broadcasting anything.
    size_t offset_view =
        getPhysicalOffset(coords, broadcast_strides, out->m_offset);
    size_t offset_out = getPhysicalOffset(coords, a->m_strides, a->m_offset);
    if ((*a->m_data)[offset_out] > (*out->m_data)[offset_view]) {
      (*out->m_data)[offset_view] = (*a->m_data)[offset_out];
      // Coordinate of the max value (the one that gets sent to the particular
      // location).
      (*argmax->m_data)[offset_view] = coords[dim];
    }
    incrementCoords(coords, a->m_shape);
  }
  out->m_requiresGrad = GradMode::enabled && (a->m_requiresGrad);
  if (out->m_requiresGrad) {
    out->m_creator =
        std::make_unique<ReduceMaxOp>(std::vector{a}, out, dim, argmax);
  }
  if (!keep_dim) {
    out = squeeze(out, {dim});
  }
  return out;
}

std::shared_ptr<TensorImpl> makeContiguous(std::shared_ptr<TensorImpl> a) {
  if (a->isContiguous()) {
    return a;
  }
  size_t total_elements = sizeFromShape(a->m_shape);
  std::vector<size_t> coords(a->m_shape.size(), 0);
  std::vector<float> newData(total_elements);
  for (size_t i = 0; i < total_elements; ++i) {
    size_t offset_a = getPhysicalOffset(coords, a->m_strides, a->m_offset);
    newData[i] = (*a->m_data)[offset_a];
    incrementCoords(coords, a->m_shape);
  }
  // contiguous, so ctor will calculate default strides.
  auto out = std::make_shared<TensorImpl>(a->m_shape, newData);
  out->m_requiresGrad = GradMode::enabled && (a->m_requiresGrad);
  if (out->m_requiresGrad) {
    out->m_creator = std::make_unique<MakeContiguousOp>(std::vector{a}, out);
  }
  return out;
}

std::shared_ptr<TensorImpl> reshape(std::shared_ptr<TensorImpl> a,
                                    Shape newShape) {
  // TODO: verify newShape.
  // Note: we use a weaker condition in terms of when a reshape view is
  // possible. Need to understand the stronger condition and potentially use it
  // in the future, so we don't unnecessarily make contiguous.
  auto a_cont = makeContiguous(a);
  auto out = std::make_shared<TensorImpl>(
      newShape, defaultStridesFromShape(newShape), a_cont->m_data);
  out->m_requiresGrad = GradMode::enabled && (a_cont->m_requiresGrad);
  if (out->m_requiresGrad) {
    out->m_creator = std::make_unique<ReshapeOp>(std::vector{a_cont}, out);
  }
  return out;
}

std::shared_ptr<TensorImpl> slice(std::shared_ptr<TensorImpl> a,
                                  std::vector<int> start, Shape size) {
  if (start.size() != size.size() || start.size() != a->getRank()) {
    throw std::runtime_error(
        "Invalid start/size lists, must be equal to rank of input");
  }
  // Calculate offset
  size_t newOffset = a->m_offset;
  for (size_t i = 0; i < a->getRank(); ++i) {
    newOffset += start[i] * a->m_strides[i];
  }
  auto out = std::make_shared<TensorImpl>(size, a->m_strides, a->m_data);
  out->m_offset = newOffset;
  out->m_requiresGrad = GradMode::enabled && (a->m_requiresGrad);
  if (out->m_requiresGrad) {
    out->m_creator = std::make_unique<SliceOp>(
        std::vector{a}, out, std::move(start), std::move(size));
  }
  return out;
}

std::shared_ptr<TensorImpl> relu(std::shared_ptr<TensorImpl> a) {
  auto out = std::make_shared<TensorImpl>(a->m_shape);
  elementwiseUnaryKernel(out, a, [](float val) { return std::max(val, 0.0f); });
  out->m_requiresGrad = GradMode::enabled && (a->m_requiresGrad);
  if (out->m_requiresGrad) {
    out->m_creator = std::make_unique<ReLUOp>(std::vector{a}, out);
  }
  return out;
}

std::shared_ptr<TensorImpl> geluApprox(std::shared_ptr<TensorImpl> a) {
  auto out = std::make_shared<TensorImpl>(a->m_shape);
  const float sqrt2Pi = 0.7978845608f;
  const float coeff = 0.044715f;

  elementwiseUnaryKernel(out, a, [=](float x) {
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(sqrt2Pi * (x + coeff * x3)));
  });

  out->m_requiresGrad = GradMode::enabled && a->m_requiresGrad;
  if (out->m_requiresGrad) {
    out->m_creator = std::make_unique<GeluOp>(std::vector{a}, out);
  }
  return out;
}

std::shared_ptr<TensorImpl> softmax(std::shared_ptr<TensorImpl> a, int dim) {
  // Do not need to have a specific SoftmaxOp since it is a combination of
  // several operations that have their own backward passes defined, so it will
  // just work.
  // In the future, this should be implemented as a fused kernel for better
  // numerical stability (in which case we need a SoftmaxOp).
  auto mx = reduceMax(a, dim, true);
  auto shifted = subtract(a, mx);
  auto expA = exp(shifted);
  auto sumExp = reduceSum(expA, {dim}, true);
  return divide(expA, sumExp);
}

std::shared_ptr<TensorImpl> gather(std::shared_ptr<TensorImpl> a, int dimInput,
                                   std::shared_ptr<TensorImpl> index) {
  if (a->getRank() != index->getRank()) {
    throw std::runtime_error{"Ranks of input and index tensors must be equal"};
  }
  if (index->m_requiresGrad) {
    throw std::runtime_error{"Index tensor cannot require a gradient"};
  }
  int dim = dimInput >= 0 ? dimInput : a->getRank() + dimInput;
  size_t totalElements = sizeFromShape(index->m_shape);
  auto out = std::make_shared<TensorImpl>(index->m_shape);
  std::vector<size_t> coords(a->getRank());
  for (int i = 0; i < totalElements; ++i) {
    size_t offset =
        getPhysicalOffset(coords, index->m_strides, index->m_offset);
    // TODO: If it's a negative float value we should throw an error.
    int val = static_cast<int>((*index->m_data)[offset]);
    // e.g., input[index[i][j][k]][j][k] if dim == 0, and in rank 3
    // TODO: is this ok being a size_t?
    size_t inputOffset = getPhysicalOffset(coords, a->m_strides, a->m_offset);
    inputOffset -= coords[dim] * a->m_strides[dim];
    inputOffset += val * a->m_strides[dim];
    size_t resOffset = getPhysicalOffset(coords, out->m_strides, out->m_offset);
    (*out->m_data)[resOffset] = (*a->m_data)[inputOffset];
    incrementCoords(coords, index->m_shape);
  }
  out->m_requiresGrad = GradMode::enabled && a->m_requiresGrad;
  if (out->m_requiresGrad) {
    out->m_creator =
        std::make_unique<GatherOp>(std::vector{a, index}, out, dim);
  }
  return out;
}

std::shared_ptr<TensorImpl> triu(std::shared_ptr<TensorImpl> a, int diagonal) {
  if (a->getRank() < 2) {
    throw std::runtime_error("triu only supported for rank >= 2");
  }
  auto out = std::make_shared<TensorImpl>(a->m_shape);
  size_t total_elements = sizeFromShape(a->m_shape);
  std::vector<size_t> coords(a->m_shape.size(), 0);
  size_t aRank = a->getRank();

  for (size_t i = 0; i < total_elements; ++i) {
    size_t offset_a = getPhysicalOffset(coords, a->m_strides, a->m_offset);
    size_t offset_res =
        getPhysicalOffset(coords, out->m_strides, out->m_offset);
    // j - i >= diagonal => kept, otherwise 0.
    int i_coord = static_cast<int>(coords[aRank - 2]);
    int j_coord = static_cast<int>(coords[aRank - 1]);
    (*out->m_data)[offset_res] =
        (j_coord - i_coord >= diagonal) ? (*a->m_data)[offset_a] : 0.0f;
    incrementCoords(coords, a->m_shape);
  }
  out->m_requiresGrad = GradMode::enabled && (a->m_requiresGrad);
  if (out->m_requiresGrad) {
    out->m_creator = std::make_unique<TriuOp>(std::vector{a}, out, diagonal);
  }
  return out;
}

std::shared_ptr<TensorImpl> maskedFill(std::shared_ptr<TensorImpl> a,
                                       std::shared_ptr<TensorImpl> mask,
                                       float val) {
  if (mask->m_requiresGrad) {
    throw std::runtime_error(
        "It is invalid for the mask to require a gradient");
  }
  auto target_shape_opt = getBroadcastShape(a->m_shape, mask->m_shape);
  if (!target_shape_opt || a->m_shape != *target_shape_opt) {
    throw std::runtime_error("The mask cannot be broadcasted to the input");
  }
  auto mask_bc = (mask->m_shape == *target_shape_opt)
                     ? mask
                     : broadcast(mask, *target_shape_opt);
  auto out = std::make_shared<TensorImpl>(a->m_shape);
  elementwiseBinaryKernel(out, a, mask_bc,
                          [&](float a, float b) { return b == 1 ? val : a; });
  out->m_requiresGrad = GradMode::enabled && a->m_requiresGrad;
  if (out->m_requiresGrad) {
    out->m_creator =
        std::make_unique<MaskedFillOp>(std::vector{a, mask}, out, val);
  }
  return out;
}

////////////////////////////////////////////////////////////////////////////////

// In-place operations. Used by the optimizer and should also be used in the
// future in `backward` implementations of several operations.

// Assumes `a` is the Tensor we are changing.
// Returns new `b` after broadcasting.
std::shared_ptr<TensorImpl> alignInputsForInPlace(
    std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b) {
  // Uses the same semantics as PyTorch; in particular, the shape of `a` cannot
  // change as a result of broadcasting.
  auto target_shape_opt = getBroadcastShape(a->m_shape, b->m_shape);
  if (!target_shape_opt || a->m_shape != *target_shape_opt) {
    throw std::runtime_error(
        "Tensors are not broadcast-compatible, or the in-place tensor's shape "
        "changes");
  }
  Shape& target_shape = *target_shape_opt;
  auto b_bc = (b->m_shape == target_shape) ? b : broadcast(b, target_shape);
  return b_bc;
}

void subtract_(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b) {
  if (GradMode::enabled) {
    throw std::runtime_error{
        "Gradient must be disabled when performing in-place operations"};
  }
  auto b_bc = alignInputsForInPlace(a, b);
  elementwiseBinaryKernel(a, a, b_bc, std::minus<float>());
}
