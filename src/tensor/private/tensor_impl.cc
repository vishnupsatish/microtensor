/**
 *    Author:  Vishnu Satish
 *    Created: Dec 26, 2025
 **/

#include "tensor_impl.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "operation.h"
#include "shape.h"

namespace {

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

}  // namespace

TensorImpl::TensorImpl(const Shape& shape)
    : m_shape{shape},
      m_data{std::make_shared<Storage>(sizeFromShape(shape))},
      m_strides(defaultStridesFromShape(shape)) {}

TensorImpl::TensorImpl(const Shape& shape, const std::vector<size_t>& strides,
                       std::shared_ptr<Storage> data)
    : m_shape{shape}, m_data{data}, m_strides{strides} {}

std::shared_ptr<TensorImpl> TensorImpl::getGrad() const { return m_grad; }

void TensorImpl::fill_random() {
  float mn = 0;
  float mx = 10;

  std::random_device rd;
  std::mt19937 gen(rd());  // seeded each call
  std::uniform_real_distribution<float> dist(mn, mx);

  // TODO: must take into account the offset, and also that the storage could be
  // shared... not ideal. In general, operations that mutate tensor data is
  // unclean with this design.
  for (auto& x : *m_data) {
    x = dist(gen);
  }
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
    // Disable tracking creator.
    m_grad = add(m_grad, new_grad, false);
  }
}

void TensorImpl::backward() {
  if (!m_grad) {
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
    os << (*m_data)[m_offset];
    return;
  }

  std::vector<size_t> coords(m_shape.size(), 0);

  // Recursive helper to handle nested brackets and indentation
  auto recursive_print = [&](auto& self, size_t dim) -> void {
    // Base case: we've reached the innermost level, print the value
    if (dim == m_shape.size()) {
      os << (*m_data)[get_physical_offset(coords, m_strides, m_offset)];
      return;
    }

    os << "[";
    for (size_t i = 0; i < m_shape[dim]; ++i) {
      coords[dim] = i;
      self(self, dim + 1);

      // If there's another element in this dimension, add a comma
      if (i < m_shape[dim] - 1) {
        os << ", ";

        // If we are printing anything deeper than a 1D vector,
        // add a newline and indentation for readability
        if (m_shape.size() - dim > 1) {
          os << "\n";
          for (size_t j = 0; j <= dim; ++j) os << "  ";
        }
      }
    }
    os << "]";
  };

  recursive_print(recursive_print, 0);
}

size_t get_physical_offset(const std::vector<size_t>& coords,
                           const std::vector<size_t>& strides, size_t offset) {
  size_t out = offset;
  for (size_t i = 0; i < coords.size(); ++i) {
    out += coords[i] * strides[i];
  }
  return out;
}

void increment_coords(std::vector<size_t>& coords, const Shape& shape) {
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    coords[i]++;
    if (coords[i] < shape[i]) return;
    coords[i] = 0;
  }
}
