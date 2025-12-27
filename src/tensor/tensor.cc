/**
 *    Author:  Vishnu Satish
 *    Created: Dec 25, 2025
 **/

#include "tensor.h"

#include <iostream>
#include <memory>
#include <vector>

#include "private/operation.h"
#include "private/tensor_impl.h"

Tensor::Tensor(const Shape& shape)
    : m_impl{std::make_shared<TensorImpl>(shape)} {}

Tensor::Tensor(const Tensor& other) : m_impl{other.m_impl} {}

Tensor::Tensor(std::shared_ptr<TensorImpl> otherImpl) : m_impl{otherImpl} {}

Tensor::Tensor(const Shape& shape, const std::vector<float>& data)
    : m_impl{std::make_shared<TensorImpl>(shape, data)} {}

void Tensor::fill_random() { m_impl->fill_random(); }

void Tensor::print(std::ostream& os) { m_impl->print(os); }

Tensor Tensor::getGrad() const { return Tensor{m_impl->getGrad()}; }

Shape Tensor::getShape() const { return m_impl->m_shape; }

Tensor Tensor::operator+(const Tensor& other) {
  return Tensor{add(m_impl, other.m_impl)};
}

Tensor Tensor::operator*(const Tensor& other) {
  return Tensor{multiply(m_impl, other.m_impl)};
}

void Tensor::backward() { m_impl->backward(); }

void Tensor::dump_tensor(std::ostream& os) { m_impl->dump_tensor(os); }