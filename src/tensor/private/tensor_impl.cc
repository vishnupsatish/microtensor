/**
 *    Author:  Vishnu Satish
 *    Created: Dec 26, 2025
 **/

#include "tensor_impl.h"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "operation.h"
#include "shape.h"

namespace {

void buildTopo(Operation* op, std::set<Operation*>& visited,
               std::vector<Operation*>& topo_order) {
  if (!op || visited.contains(op)) return;
  visited.insert(op);
  for (auto& input : op->m_parents) {
    if (input->m_creator) {
      buildTopo(input->m_creator.get(), visited, topo_order);
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

TensorImpl::TensorImpl(const Shape& shape, const std::vector<float>& data)
    : m_shape{shape},
      m_data{std::make_shared<Storage>(data)},
      m_strides(defaultStridesFromShape(shape)) {}

std::shared_ptr<TensorImpl> TensorImpl::getGrad() const { return m_grad; }

size_t TensorImpl::getRank() const { return m_shape.size(); }

void TensorImpl::fillRandom() {
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
void TensorImpl::initializeGrad() {
  m_grad = std::make_shared<TensorImpl>(m_shape);
  std::fill(std::begin(*m_grad->m_data), std::end(*m_grad->m_data), 1.0);
}

void TensorImpl::accumulateGrad(std::shared_ptr<TensorImpl> new_grad) {
  if (!m_grad) {
    m_grad = new_grad;
  } else {
    // Disable tracking creator.
    m_grad = add(m_grad, new_grad, false);
  }
}

void TensorImpl::backward() {
  if (!m_grad) {
    initializeGrad();
  }

  std::vector<Operation*> topo_order;
  std::set<Operation*> visited;

  if (this->m_creator) {
    buildTopo(this->m_creator.get(), visited, topo_order);
  }

  std::reverse(topo_order.begin(), topo_order.end());

  for (auto op : topo_order) {
    std::shared_ptr<TensorImpl> outTensor = op->m_output.lock();
    auto inp_grads = op->backward(outTensor->m_grad);
    assert(inp_grads.size() == op->m_parents.size());
    for (size_t i = 0; i < op->m_parents.size(); ++i) {
      auto input = op->m_parents[i];
      auto calculated_grad = inp_grads[i];
      // Very inefficient; allocates new memory for m_grad every time
      // we accumulate gradient. This is very suboptimal. Need to have an
      // in-place gradient accumulation kernel.
      input->accumulateGrad(calculated_grad);
    }
  }
}

// Note: written by AI.
void TensorImpl::print(std::ostream& os) {
  // Case 0: Scalar (0-D)
  if (m_shape.empty()) {
    os << (*m_data)[m_offset];
    return;
  }

  std::vector<size_t> coords(m_shape.size(), 0);

  auto recursive_print = [&](auto& self, size_t dim) -> void {
    if (dim == m_shape.size()) {
      os << (*m_data)[getPhysicalOffset(coords, m_strides, m_offset)];
      return;
    }
    os << "[";
    for (size_t i = 0; i < m_shape[dim]; ++i) {
      coords[dim] = i;
      self(self, dim + 1);
      if (i < m_shape[dim] - 1) {
        os << ", ";
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

// Used for differential testing.
// Format:
// rank
// shape[0] shape[1] ... shape[rank-1]
// data[0] data[1] ... data[total_elements-1]
void TensorImpl::dumpTensor(std::ostream& os) {
  os << m_shape.size() << "\n";
  for (size_t dim : m_shape) {
    os << dim << " ";
  }
  os << "\n";

  size_t total_elements = sizeFromShape(m_shape);
  std::vector<size_t> coords(m_shape.size(), 0);

  os << std::fixed << std::setprecision(6);  // 6 decimal places

  for (size_t i = 0; i < total_elements; ++i) {
    size_t offset = getPhysicalOffset(coords, m_strides, m_offset);
    os << (*m_data)[offset] << " ";
    incrementCoords(coords, m_shape);
  }
  os << "\n";
}

size_t getPhysicalOffset(const std::vector<size_t>& coords,
                         const std::vector<size_t>& strides, size_t offset) {
  // inefficient.
  size_t out = offset;
  for (size_t i = 0; i < coords.size(); ++i) {
    out += coords[i] * strides[i];
  }
  return out;
}

void incrementCoords(std::vector<size_t>& coords, const Shape& shape) {
  // also add an assertion to make sure we are not incrementing past the shape.
  assert(coords.size() == shape.size());
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    coords[i]++;
    if (coords[i] < shape[i]) return;
    coords[i] = 0;
  }
}

std::vector<size_t> getBroadcastStrides(std::shared_ptr<TensorImpl> a,
                                        const Shape& target) {
  std::vector<size_t> strides(target.size(), 0);
  auto it_target_dim = target.rbegin();
  auto it_new_stride = strides.rbegin();
  auto it_input_dim = a->m_shape.rbegin();
  auto it_input_stride = a->m_strides.rbegin();

  while (it_target_dim != target.rend()) {
    size_t current_target_dim = *it_target_dim;
    if (it_input_dim != a->m_shape.rend()) {
      size_t current_input_dim = *it_input_dim;
      size_t current_input_stride = *it_input_stride;
      if (current_input_dim == current_target_dim) {
        *it_new_stride = current_input_stride;
      } else if (current_input_dim == 1) {
        *it_new_stride = 0;
      }
      ++it_input_dim;
      ++it_input_stride;
    } else {
      *it_new_stride = 0;
    }
    ++it_target_dim;
    ++it_new_stride;
  }
  return strides;
}
