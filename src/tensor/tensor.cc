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

void Tensor::fillRandom() { m_impl->fillRandom(); }

void Tensor::print(std::ostream& os) { m_impl->print(os); }

Tensor Tensor::getGrad() const { return Tensor{m_impl->getGrad()}; }

Shape Tensor::getShape() const { return m_impl->m_shape; }

Tensor Tensor::operator+(const Tensor& other) {
  return Tensor{add(m_impl, other.m_impl)};
}

Tensor Tensor::operator*(const Tensor& other) {
  return Tensor{multiply(m_impl, other.m_impl)};
}

Tensor Tensor::tanh() { return Tensor{::tanh(m_impl)}; }

Tensor Tensor::matmul(const Tensor& other) {
  return Tensor{::matmul(m_impl, other.m_impl)};
}

Tensor Tensor::permute(const std::vector<size_t>& dims) {
  return Tensor{::permute(m_impl, dims)};
}

Tensor Tensor::broadcast(const Shape& target) {
  return Tensor{::broadcast(m_impl, target)};
}

Tensor Tensor::reduceSum(const std::vector<int>& dims, bool keep_dims) {
  return Tensor{::reduceSum(m_impl, dims, keep_dims, true)};
}

void Tensor::backward() { m_impl->backward(); }

void Tensor::dumpTensor(std::ostream& os) { m_impl->dumpTensor(os); }