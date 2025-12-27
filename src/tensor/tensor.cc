/**
 *    Author:  Vishnu Satish
 *    Created: Dec 25, 2025
 **/

#include "tensor.h"

#include <assert.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <vector>

// Stores all the state necessary to perform an operation forward or
// backwards.
class Operation {
 public:
  virtual ~Operation() = default;

  Operation(std::vector<std::shared_ptr<TensorImpl>> parents,
            std::shared_ptr<TensorImpl> output)
      : m_parents(std::move(parents)), m_output(output) {}

  virtual std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) = 0;

  std::vector<std::shared_ptr<TensorImpl>> m_parents;
  std::weak_ptr<TensorImpl> m_output;
};

std::shared_ptr<TensorImpl> multiply(std::shared_ptr<TensorImpl> a,
                                     std::shared_ptr<TensorImpl> b);

std::shared_ptr<TensorImpl> add(std::shared_ptr<TensorImpl> a,
                                std::shared_ptr<TensorImpl> b);

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

size_t sizeFromShape(const std::vector<size_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), 1,
                         std::multiplies<size_t>());
}

std::vector<size_t> defaultStridesFromShape(const std::vector<size_t>& shape) {
  size_t ndim = shape.size();
  std::vector<size_t> strides(ndim, 1);
  if (ndim == 0) return strides;
  for (int i = ndim - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

TensorImpl::TensorImpl(const std::vector<size_t>& shape)
    : m_shape{shape},
      m_data{std::make_shared<Storage>(sizeFromShape(shape))},
      m_strides(defaultStridesFromShape(shape)) {}

std::shared_ptr<TensorImpl> TensorImpl::getGrad() const { return m_grad; }

void TensorImpl::fill_random() {
  float mn = 0;
  float mx = 10;

  std::random_device rd;
  std::mt19937 gen(rd());  // seeded each call
  std::uniform_real_distribution<float> dist(mn, mx);

  for (auto& x : *m_data) {
    x = dist(gen);
  }
}

void build_topo(Operation* op, std::set<Operation*>& visited,
                std::vector<Operation*>& topo_order) {
  if (!op || visited.contains(op)) return;
  visited.insert(op);
  for (auto& input : op->m_parents) {
    if (input->m_creator) {
      build_topo(input->m_creator.get(), visited, topo_order);
    }
  }
  topo_order.push_back(op);
}

// Initialize to all ones.
void TensorImpl::initialize_grad() {
  m_grad = std::make_shared<TensorImpl>(m_shape);
  std::fill(std::begin(*m_grad->m_data), std::end(*m_grad->m_data), 1.0);
}

void TensorImpl::accumulate_grad(std::shared_ptr<TensorImpl> new_grad) {
  if (!m_grad) {
    m_grad = new_grad;
  } else {
    // Need to be careful here, but I think this is OK.
    m_grad = add(m_grad, new_grad);
  }
}

void TensorImpl::backward() {
  if (!m_grad) {
    // set to all 1s, gradient of us wrt us is 1
    initialize_grad();
  }

  std::vector<Operation*> topo_order;
  std::set<Operation*> visited;

  if (this->m_creator) {
    build_topo(this->m_creator.get(), visited, topo_order);
  }

  std::reverse(topo_order.begin(), topo_order.end());

  for (auto op : topo_order) {
    std::shared_ptr<TensorImpl> outTensor = op->m_output.lock();
    auto inp_grads = op->backward(outTensor->m_grad);
    // Accumulate gradients for inputs.
    assert(inp_grads.size() == op->m_parents.size());
    for (size_t i = 0; i < op->m_parents.size(); ++i) {
      auto input = op->m_parents[i];
      auto calculated_grad = inp_grads[i];
      input->accumulate_grad(calculated_grad);
    }
  }
}

void TensorImpl::print(std::ostream& os) {
  // Case 0: Scalar (0-D)
  if (m_shape.empty()) {
    os << (*m_data)[0];
    return;
  }

  // Case 1: Vector (1-D)
  if (m_shape.size() == 1) {
    os << "[";
    for (int i = 0; i < m_shape[0]; ++i) {
      // Calculate index: start + i * stride
      int idx = 0 + (i * m_strides[0]);
      os << (*m_data)[idx];

      // Comma separation
      if (i < m_shape[0] - 1) os << ", ";
    }
    os << "]";
    return;
  }

  // Case 2: Matrix (2-D)
  if (m_shape.size() == 2) {
    os << "[\n";
    for (int i = 0; i < m_shape[0]; ++i) {  // Rows
      os << "  [";
      for (int j = 0; j < m_shape[1]; ++j) {  // Columns
        // Calculate index: start + row_offset + col_offset
        int idx = (i * m_strides[0]) + (j * m_strides[1]);
        os << (*m_data)[idx];

        if (j < m_shape[1] - 1) os << ", ";
      }
      os << "]";

      // Newline between rows, but not after the last one
      if (i < m_shape[0] - 1) os << ",\n";
    }
    os << "\n]";
    return;
  }

  os << "[Tensor with " << m_shape.size()
     << " dimensions (Print not supported yet)]";
}

/////////////////////////////////////////////////////////////////////////////////

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
    auto grad_op1 = multiply(grad_output, m_parents[1]);
    auto grad_op2 = multiply(grad_output, m_parents[0]);

    return {grad_op1, grad_op2};
  }
};

/////////////////////////////////////////////////////////////////////////////////

// Kernels/functions that perform computations and populate data so `backward`
// can run.

std::shared_ptr<TensorImpl> multiply(std::shared_ptr<TensorImpl> a,
                                     std::shared_ptr<TensorImpl> b) {
  auto out = std::make_shared<TensorImpl>(a->m_shape);

  for (size_t i = 0; i < out->m_data->size(); ++i)
    (*out->m_data)[i] = (*a->m_data)[i] * (*b->m_data)[i];

  out->m_creator = std::make_unique<MulOp>(std::vector{a, b}, out);

  return out;
}

std::shared_ptr<TensorImpl> add(std::shared_ptr<TensorImpl> a,
                                std::shared_ptr<TensorImpl> b) {
  auto out = std::make_shared<TensorImpl>(a->m_shape);

  for (size_t i = 0; i < out->m_data->size(); ++i)
    (*out->m_data)[i] = (*a->m_data)[i] + (*b->m_data)[i];

  out->m_creator = std::make_unique<AddOp>(std::vector{a, b}, out);

  return out;
}

/////////////////////////////////////////////////////////////////////////////////

Tensor::Tensor(const std::vector<size_t>& shape)
    : m_impl{std::make_shared<TensorImpl>(shape)} {}

Tensor::Tensor(const Tensor& other) : m_impl{other.m_impl} {}

Tensor::Tensor(std::shared_ptr<TensorImpl> otherImpl) : m_impl{otherImpl} {}

void Tensor::fill_random() { m_impl->fill_random(); }

void Tensor::print(std::ostream& os) { m_impl->print(os); }

Tensor Tensor::getGrad() const { return Tensor{m_impl->getGrad()}; }

Tensor Tensor::operator+(const Tensor& other) {
  return Tensor{add(m_impl, other.m_impl)};
}

Tensor Tensor::operator*(const Tensor& other) {
  return Tensor{multiply(m_impl, other.m_impl)};
}

void Tensor::backward() { m_impl->backward(); }
